import os
import time
import gc
import glob
from typing import Dict, Optional
import numpy as np
import torch
from training.selfplay import SelfPlayEngine
from training.replay_buffer import ReplayBuffer
from agent.policy_value_net import PolicyValueNet
from training.loss import AlphaZeroLoss


class AlphaZeroTrainer:
    """
    Optimized AlphaZero trainer.
    
    v2.0 Changes:
    - Single forward pass per batch (not H+1 per experience)
    - Efficient graph batching
    - ~10-50x faster than v1.0
    """
    
    def __init__(self,
                 network: PolicyValueNet,
                 mcts_coordinator,
                 selfplay_engine: SelfPlayEngine,
                 replay_buffer: ReplayBuffer,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: AlphaZeroLoss,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_compile: bool = True):
        self.network = network.to(device)
        
        # Optimize with torch.compile for 30-50% speedup (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            print("Optimizing network with torch.compile()...")
            try:
                self.network = torch.compile(self.network, mode='reduce-overhead')
            except Exception as e:
                print(f"torch.compile failed: {e}. Continuing without compilation.")
        
        self.mcts = mcts_coordinator
        self.selfplay = selfplay_engine
        self.buffer = replay_buffer
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        
        self.iteration = 0
        self.stats_history = []
    
    def train_step(self, batch_size: int) -> Dict:
        """
        OPTIMIZED: Single forward pass per batch.
        
        v1.0: B × (H+1) forward passes
        v2.0: 1 forward pass
        
        Speedup: ~(H+1) × B for typical batch sizes
        """
        if not self.buffer.is_ready():
            return {'skipped': True}
        
        self.network.train()
        
        # Sample batch
        experiences = self.buffer.sample(batch_size)
        
        # Collate into batched format (OPTIMIZED)
        batched_graphs, target_policies_per_head, target_values = self.buffer.collate_batch(experiences)
        
        # Move to device
        batched_graphs = batched_graphs.to(self.device)
        target_policies_per_head = [t.to(self.device) for t in target_policies_per_head]
        target_values = target_values.to(self.device)
        
        # SINGLE FORWARD PASS for entire batch (OPTIMIZED)
        with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):  # Mixed precision
            policy_logits_list, predicted_values = self.network(batched_graphs, head_id=None)
            
            # Compute loss
            loss, stats = self.loss_fn(
                policy_logits_list,
                target_policies_per_head,
                predicted_values,
                target_values
            )
            
            # L2 regularization
            l2_reg = sum(p.pow(2.0).sum() for p in self.network.parameters())
            loss = loss + self.loss_fn.l2_weight * l2_reg
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        stats['l2_reg'] = (self.loss_fn.l2_weight * l2_reg).item()
        
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
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        start_iteration = 0
        
        # Auto-resume from latest checkpoint
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
            
            # Self-play
            print("Generating self-play data...")
            start_time = time.time()
            experiences = self.selfplay.generate_episodes(episodes_per_iteration)
            self.buffer.add_episode(experiences)
            selfplay_time = time.time() - start_time
            
            print(f"Generated {len(experiences)} experiences in {selfplay_time:.1f}s")
            print(f"Buffer size: {len(self.buffer)}/{self.buffer.max_size}")
            
            # Training
            if self.buffer.is_ready():
                print(f"\nTraining for {train_steps_per_iteration} steps...")
                start_time = time.time()
                
                train_stats = []
                for step in range(train_steps_per_iteration):
                    stats = self.train_step(batch_size)
                    if 'skipped' not in stats:
                        train_stats.append(stats)
                
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
                    print(f"  Entropy: {avg_stats['entropy']:.4f}")
                    print(f"  Throughput: {train_steps_per_iteration * batch_size / train_time:.1f} samples/sec")
                    
                    self.stats_history.append(avg_stats)
            
            # Save checkpoint
            if (iteration + 1) % save_interval == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_{iteration+1}.pt')
                self.save_checkpoint(checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
            
            # Memory cleanup every iteration
            self._cleanup_memory()
    
    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        # Handle torch.compile wrapped models
        network_to_save = self.network
        if hasattr(self.network, '_orig_mod'):
            network_to_save = self.network._orig_mod
        
        # Get state dict and strip any _orig_mod prefix for compatibility
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
            'stats_history': self.stats_history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle torch.compile wrapped models
        network_to_load = self.network
        if hasattr(self.network, '_orig_mod'):
            network_to_load = self.network._orig_mod
        
        # Fix state dict key mismatches from torch.compile
        state_dict = checkpoint['network_state_dict']
        model_keys = set(network_to_load.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # Check if we need to strip or add _orig_mod prefix
        if len(model_keys & ckpt_keys) == 0:
            # Keys don't match - try fixing prefixes
            fixed_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    # Strip prefix
                    new_key = key[len('_orig_mod.'):]
                else:
                    new_key = key
                fixed_state_dict[new_key] = value
            state_dict = fixed_state_dict
        
        network_to_load.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.stats_history = checkpoint.get('stats_history', [])
        print(f"Loaded checkpoint from iteration {self.iteration}")
    
    def _find_latest_checkpoint(self, save_dir: str) -> Optional[str]:
        """Find the latest checkpoint in save_dir."""
        pattern = os.path.join(save_dir, 'checkpoint_*.pt')
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            return None
        
        # Extract iteration numbers and find max
        def get_iteration(path):
            basename = os.path.basename(path)
            try:
                return int(basename.replace('checkpoint_', '').replace('.pt', ''))
            except ValueError:
                return -1
        
        latest = max(checkpoints, key=get_iteration)
        return latest
    
    def _cleanup_memory(self):
        """Aggressive memory cleanup to prevent host memory buildup."""
        # Clear MCTS caches
        if hasattr(self.mcts, 'clear_caches'):
            self.mcts.clear_caches()
        elif hasattr(self.mcts, 'reset_trees'):
            self.mcts.reset_trees()
        
        # Clear evaluation cache if available
        if hasattr(self.selfplay, 'evaluator') and hasattr(self.selfplay.evaluator, 'clear_cache'):
            self.selfplay.evaluator.clear_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()