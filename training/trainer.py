import os
import time
import gc
import glob
from typing import Dict, Optional, List
import numpy as np
import torch
from training.selfplay import SelfPlayEngine
from training.replay_buffer import ReplayBuffer, Experience
from agent.policy_value_net import PolicyValueNet
from training.loss import AlphaZeroLoss


class AlphaZeroTrainer:
    """
    AlphaZero trainer with TD(n) for delayed rewards.

    v3.0 Changes:
    - TD(n) training for value network (not raw MC targets)
    - Root-relative value convention: V(s) = E[γ^T R | s]
    """

    def __init__(self,
                 network: PolicyValueNet,
                 mcts_coordinator,
                 selfplay_engine: SelfPlayEngine,
                 replay_buffer: ReplayBuffer,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: AlphaZeroLoss,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_compile: bool = False,
                 value_loss_decay: float = 0.95,
                 min_value_loss_weight: float = 0.1,
                 # NEW: TD(n) parameters
                 n_step_td: int = 5,
                 gamma: float = 0.99,
                 use_td_targets: bool = True):
        
        self.network = network.to(device)

        if use_compile and hasattr(torch, 'compile'):
            print("Optimizing network with torch.compile()...")
            try:
                self.network = torch.compile(self.network, mode='default')
            except Exception as e:
                print(f"torch.compile failed: {e}. Continuing without compilation.")

        self.mcts = mcts_coordinator
        self.selfplay = selfplay_engine
        self.buffer = replay_buffer
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

        self.value_loss_decay = value_loss_decay
        self.min_value_loss_weight = min_value_loss_weight
        self.initial_value_loss_weight = loss_fn.value_loss_weight

        # NEW: TD(n) configuration
        self.n_step_td = n_step_td
        self.gamma = gamma
        self.use_td_targets = use_td_targets
        
        print(f"TD(n) config: n_step={n_step_td}, gamma={gamma}, enabled={use_td_targets}")

        self.iteration = 0
        self.stats_history = []

    def _update_value_loss_weight(self):
        """Decay value loss weight to reduce overfitting."""
        new_weight = max(
            self.initial_value_loss_weight * (self.value_loss_decay ** self.iteration),
            self.min_value_loss_weight
        )
        self.loss_fn.value_loss_weight = new_weight
        return new_weight

    # ============================================
    # NEW: TD(n) Target Computation
    # ============================================
    
    def compute_td_targets(self, episode_data: Dict) -> List[Experience]:
        """
        Compute TD(n) targets from episode data and create Experiences.
        
        Root-relative convention: V(s_t) = E[γ^T R | s_t]
        
        TD(n) target:
            G_t^(n) = γ^n * V(s_{t+n})    if t+n < T  (bootstrap)
            G_t^(n) = γ^{T-t} * R         otherwise  (terminal)
        """
        states = episode_data['states']
        policies = episode_data['policies']
        value_estimates = episode_data['value_estimates']
        final_reward = episode_data['final_reward']
        T = len(states)
        
        if not self.use_td_targets:
            # Monte Carlo: γ^{T-t} * R for all states
            td_targets = [(self.gamma ** (T - 1 - t)) * final_reward for t in range(T)]
        else:
            # TD(n) with bootstrap
            td_targets = []
            gamma_n = self.gamma ** self.n_step_td
            
            for t in range(T):
                if t + self.n_step_td < T:
                    # Bootstrap from value estimate n steps ahead
                    # V(s_{t+n}) was stored during MCTS
                    bootstrap_value = value_estimates[t + self.n_step_td]
                    target = gamma_n * bootstrap_value
                else:
                    # Terminal: remaining discount to final reward
                    steps_to_end = T - 1 - t
                    target = (self.gamma ** steps_to_end) * final_reward
                
                td_targets.append(target)
        
        # Create Experience objects with TD targets (not raw reward!)
        experiences = []
        for t, (state, policy, td_target) in enumerate(zip(states, policies, td_targets)):
            experiences.append(Experience(
                graph_state=state,
                visit_distributions=policy,
                final_reward=td_target,  # This is the TD target, not raw R!
                step=t
            ))
        
        return experiences

    def train_step(self, batch_size: int) -> Dict:
        """Single training step with TD(n) targets."""
        if not self.buffer.is_ready():
            return {'skipped': True}
        
        self.network.train()
        
        # Sample batch (supports PER)
        if getattr(self.buffer, 'use_per', False):
            experiences, indices, is_weights = self.buffer.sample(batch_size)
            if not experiences:
                return {'skipped': True}
        else:
            experiences = self.buffer.sample(batch_size)
            if not experiences:
                return {'skipped': True}
            indices = None
            is_weights = None
        
        # Collate batch
        batched_graphs, target_policies_per_head, target_values = self.buffer.collate_batch(experiences)
        
        # Move to device
        batched_graphs = batched_graphs.to(self.device)
        target_policies_per_head = [t.to(self.device) for t in target_policies_per_head]
        target_values = target_values.to(self.device)  # These are TD(n) targets!
        if is_weights is not None:
            is_weights = is_weights.to(self.device).view(-1)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
            policy_logits_list, predicted_values = self.network(batched_graphs, head_id=None)

            # Compute loss against TD(n) targets; pass IS weights if available
            loss, stats = self.loss_fn(
                policy_logits_list,
                target_policies_per_head,
                predicted_values,
                target_values,
                sample_weights=is_weights
            )
            
            # L2 regularization
            l2_reg = sum(p.pow(2.0).sum() for p in self.network.parameters())
            loss = loss + self.loss_fn.l2_weight * l2_reg
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        stats['l2_reg'] = (self.loss_fn.l2_weight * l2_reg).item()
        stats['td_n_step'] = self.n_step_td if self.use_td_targets else -1
        # Update priorities in PER (use absolute value error on value head as priority)
        if indices is not None and getattr(self.buffer, 'use_per', False):
            with torch.no_grad():
                value_errors = (predicted_values.detach().squeeze(-1) - target_values.detach().squeeze(-1)).abs()
                priorities = (value_errors + 1e-6).cpu().numpy().tolist()
            try:
                self.buffer.update_priorities(indices, priorities)
            except Exception:
                pass

        return stats
    
    def train(self, 
              num_iterations: int,
              episodes_per_iteration: int = 10,
              train_steps_per_iteration: int = 100,
              batch_size: int = 32,
              eval_interval: int = 10,
              save_interval: int = 5,
              save_dir: str = 'checkpoints',
              auto_resume: bool = True):
        """Main training loop with TD(n)."""
        os.makedirs(save_dir, exist_ok=True)
        
        start_iteration = 0
        
        if auto_resume:
            latest_checkpoint = self._find_latest_checkpoint(save_dir)
            if latest_checkpoint:
                self.load_checkpoint(latest_checkpoint)
                start_iteration = self.iteration + 1
                print(f"Resuming from iteration {start_iteration}")
        
        for iteration in range(start_iteration, num_iterations):
            self.iteration = iteration
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")

            current_value_weight = self._update_value_loss_weight()
            print(f"Value loss weight: {current_value_weight:.4f}")

            # Self-play
            print("Generating self-play data...")
            start_time = time.time()
            
            # NEW: Get episode dicts with value estimates
            episode_dicts = self.selfplay.generate_episodes(episodes_per_iteration)
            
            # NEW: Process each episode with TD(n) targets
            all_experiences = []
            for ep_data in episode_dicts:
                experiences = self.compute_td_targets(ep_data)
                all_experiences.extend(experiences)
            
            self.buffer.add_episode(all_experiences)
            selfplay_time = time.time() - start_time
            
            print(f"Generated {len(episode_dicts)} episodes → {len(all_experiences)} experiences in {selfplay_time:.1f}s")
            print(f"Buffer size: {len(self.buffer)}/{self.buffer.max_size}")
            if self.use_td_targets:
                print(f"Using TD({self.n_step_td}) targets")
            
            # Training
            if self.buffer.is_ready():
                print(f"\nTraining for {train_steps_per_iteration} steps...")
                start_time = time.time()
                
                train_stats = []
                for step in range(train_steps_per_iteration):
                    stats = self.train_step(batch_size)
                    if 'skipped' not in stats:
                        train_stats.append(stats)
                    else:
                        print("Training step skipped: insufficient data in replay buffer.")
                
                train_time = time.time() - start_time
                
                if train_stats:
                    avg_stats = {
                        key: np.mean([s[key] for s in train_stats])
                        for key in train_stats[0].keys()
                    }

                    print(f"\nTraining Stats ({train_time:.1f}s):")
                    print(f"  Total Loss: {avg_stats['total_loss']:.4f}")
                    print(f"  Policy Loss: {avg_stats['policy_loss']:.4f}")
                    print(f"  Value Loss: {avg_stats['value_loss']:.4f}")
                    print(f"  Value MAE: {avg_stats['value_mae']:.4f}")
                    print(f"  TD(n): n={avg_stats.get('td_n_step', 'N/A')}")
                    print(f"  Throughput: {train_steps_per_iteration * batch_size / train_time:.1f} samples/sec")

                    self.stats_history.append(avg_stats)
            
            # Save checkpoint
            if (iteration + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{iteration+1}.pt')
                self.save_checkpoint(checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
            
            self._cleanup_memory()
    
    def save_checkpoint(self, path: str):
        """Save checkpoint with TD config."""
        network_to_save = self.network
        if hasattr(self.network, '_orig_mod'):
            network_to_save = self.network._orig_mod
        
        state_dict = network_to_save.state_dict()
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
            else:
                new_key = key
            clean_state_dict[new_key] = value
        
        torch.save({
            'iteration': self.iteration,
            'network_state_dict': clean_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'stats_history': self.stats_history,
            'value_loss_weight': self.loss_fn.value_loss_weight,
            # NEW: Save TD config
            'n_step_td': self.n_step_td,
            'gamma': self.gamma,
            'use_td_targets': self.use_td_targets
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint with TD config."""
        checkpoint = torch.load(path, map_location=self.device)
        
        network_to_load = self.network
        if hasattr(self.network, '_orig_mod'):
            network_to_load = self.network._orig_mod
        
        state_dict = checkpoint['network_state_dict']
        model_keys = set(network_to_load.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        if len(model_keys & ckpt_keys) == 0:
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[len('_orig_mod.'):]
                else:
                    new_key = key
                fixed_state_dict[new_key] = value
            state_dict = fixed_state_dict
        
        network_to_load.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.stats_history = checkpoint.get('stats_history', [])

        if 'value_loss_weight' in checkpoint:
            self.loss_fn.value_loss_weight = checkpoint['value_loss_weight']
        
        # NEW: Restore TD config
        self.n_step_td = checkpoint.get('n_step_td', self.n_step_td)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.use_td_targets = checkpoint.get('use_td_targets', self.use_td_targets)
        
        print(f"Loaded checkpoint from iteration {self.iteration}")
        print(f"TD config: n_step={self.n_step_td}, gamma={self.gamma}, enabled={self.use_td_targets}")
    
    def _find_latest_checkpoint(self, save_dir: str) -> Optional[str]:
        """Find latest checkpoint."""
        pattern = os.path.join(save_dir, 'checkpoint_*.pt')
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return None
        
        def get_iteration(path):
            basename = os.path.basename(path)
            try:
                return int(basename.replace('checkpoint_', '').replace('.pt', ''))
            except ValueError:
                return -1
        
        return max(checkpoints, key=get_iteration)
    
    def _cleanup_memory(self):
        """Memory cleanup."""
        if hasattr(self.mcts, 'clear_caches'):
            self.mcts.clear_caches()
        elif hasattr(self.mcts, 'reset_trees'):
            self.mcts.reset_trees()
        
        if hasattr(self.selfplay, 'evaluator') and hasattr(self.selfplay.evaluator, 'clear_cache'):
            self.selfplay.evaluator.clear_cache()
        
        gc.collect()
        
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()