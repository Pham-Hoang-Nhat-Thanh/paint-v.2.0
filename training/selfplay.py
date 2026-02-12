import time
import gc
from typing import List, Dict
import numpy as np
from tqdm import trange
from training.replay_buffer import Experience


class SelfPlayEngine:
    """Self-play engine for generating training data."""
    
    def __init__(self,
                 mcts_coordinator,
                 initial_graph_fn: callable,
                 reward_fn: callable,
                 max_steps: int = 50,
                 num_simulations: int = 800,
                 mcts_batch_size: int = 16,  # NEW: batch size for MCTS leaf evaluation
                 temperature_schedule: callable = None):
        self.mcts = mcts_coordinator
        self.initial_graph_fn = initial_graph_fn
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.num_simulations = num_simulations
        self.mcts_batch_size = mcts_batch_size  # NEW

        if temperature_schedule is None:
            self.temperature_schedule = lambda step: max(0.1, 1.0 - step / max_steps)
        else:
            self.temperature_schedule = temperature_schedule
    
    def generate_episode(self) -> Dict:
        """
        Generate one self-play episode with value estimates for TD(n).
        
        Returns dict with:
        - states: list of graph states
        - policies: list of per-head visit distributions  
        - value_estimates: list of V(s) from network during MCTS
        - final_reward: terminal reward
        """
        graph = self.initial_graph_fn()
        trajectory = []
        episode_terminated_early = False
        
        with trange(self.max_steps, desc="[SelfPlay] Steps", unit="step") as pbar:
            for step in pbar:
                start = time.time()
                
                # ============================================
                # MODIFIED: Get both visits AND value estimates from MCTS
                # ============================================
                search_result = self.mcts.search(
                    graph, 
                    self.num_simulations, 
                    batch_size=self.mcts_batch_size
                )
                
                # search_result contains:
                # - visit_distributions: per-head MCTS policies
                # - root_value_estimate: V(s) from network at root (for TD bootstrap)
                visit_distributions = search_result['visits']
                root_value_estimate = search_result.get('root_value', 0.0)

                # Store graph copy
                graph_copy = graph.copy()
                trajectory.append({
                    'graph': graph_copy,
                    'visits': visit_distributions,
                    'step': step,
                    'value_estimate': root_value_estimate,
                })

                temperature = self.temperature_schedule(step)
                actions = self.mcts.select_actions(temperature)
                graph = self.mcts.batched_commit(graph, actions)
                
                # Clear cache on current graph to prevent memory buildup
                if hasattr(graph, 'clear_cache'):
                    graph.clear_cache()
                end = time.time()
                pbar.set_postfix({
                    'step_s': f"{end - start:.2f}",
                    'V(s)': f"{root_value_estimate:.3f}"  # Show value in progress
                })
                
                # Check termination: if graph has 0 edges (except on first step)
                if step > 0:
                    num_edges = graph.get_num_edges()
                    if num_edges == 0:
                        episode_terminated_early = True
                        pbar.close()
                        break
        
        # Apply heavy penalization if episode terminated early with 0 edges
        if episode_terminated_early:
            final_reward = 0.0  # Minimum value for sigmoid output range [0, 1]
        else:
            final_reward = self.reward_fn(graph)
        
        # NEW: Return structured data for TD(n) processing
        episode_data = {
            'states': [entry['graph'] for entry in trajectory],
            'policies': [entry['visits'] for entry in trajectory],
            'value_estimates': [entry['value_estimate'] for entry in trajectory],
            'final_reward': final_reward,
            'length': len(trajectory)
        }
        
        # Also return backward-compatible Experience objects
        experiences = [
            Experience(
                graph_state=entry['graph'],
                visit_distributions=entry['visits'],
                final_reward=final_reward,
                step=entry['step']
            )
            for entry in trajectory
        ]
        episode_data['experiences'] = experiences
        
        # Clear graph caches in trajectory before clearing
        for entry in trajectory:
            if hasattr(entry['graph'], 'clear_cache'):
                entry['graph'].clear_cache()
        
        # Clear trajectory to free memory
        trajectory.clear()
        
        # Clear MCTS tree after episode
        if hasattr(self.mcts, 'clear_caches'):
            self.mcts.clear_caches()
        elif hasattr(self.mcts, 'reset_trees'):
            self.mcts.reset_trees()
        
        return episode_data
    
    def generate_episodes(self, num_episodes: int) -> List[Dict]:
        """
        Generate multiple episodes with TD(n) data.
        
        Returns list of episode dicts (not flat Experience list).
        Trainer will process these with compute_td_targets().
        """
        all_episodes = []

        for i in trange(num_episodes, desc="[SelfPlay] Episodes", unit="ep"):
            episode_data = self.generate_episode()
            all_episodes.append(episode_data)

            if (i + 1) % 10 == 0:
                avg_reward = np.mean([ep['final_reward'] for ep in all_episodes[-10:]])
                avg_length = np.mean([ep['length'] for ep in all_episodes[-10:]])
                avg_value = np.mean([
                    np.mean(ep['value_estimates']) 
                    for ep in all_episodes[-10:] 
                    if ep['value_estimates']
                ])
                print(f"Episode {i+1}/{num_episodes}, "
                      f"Avg R: {avg_reward:.3f}, "
                      f"Avg Steps: {avg_length:.1f}, "
                      f"Avg V(s): {avg_value:.3f}")
                
                if hasattr(self, 'evaluator') and hasattr(self.evaluator, 'clear_cache'):
                    self.evaluator.clear_cache()
            
            if (i + 1) % 5 == 0:
                gc.collect()

        return all_episodes