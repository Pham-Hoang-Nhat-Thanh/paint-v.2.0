import time
import gc
from typing import List
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
                 temperature_schedule: callable = None):
        self.mcts = mcts_coordinator
        self.initial_graph_fn = initial_graph_fn
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.num_simulations = num_simulations
        
        if temperature_schedule is None:
            self.temperature_schedule = lambda step: max(0.1, 1.0 - step / max_steps)
        else:
            self.temperature_schedule = temperature_schedule
    
    def generate_episode(self) -> List[Experience]:
        """Generate one self-play episode."""
        graph = self.initial_graph_fn()
        trajectory = []
        episode_terminated_early = False
        
        with trange(self.max_steps, desc="[SelfPlay] Steps", unit="step") as pbar:
            for step in pbar:
                start = time.time()
                search_start = time.time()
                visit_distributions = self.mcts.search(graph, self.num_simulations)
                search_end = time.time()

                trajectory.append({
                    'graph': graph.copy(),
                    'visits': visit_distributions,
                    'step': step
                })

                temperature = self.temperature_schedule(step)
                actions = self.mcts.select_actions(temperature)
                graph = self.mcts.batched_commit(graph, actions)
                end = time.time()

                pbar.set_postfix({'search_s': f"{search_end - search_start:.2f}", 'step_s': f"{end - start:.2f}"})
                
                # Check termination: if graph has 0 edges (except on first step)
                if step > 0:
                    num_edges = graph.get_num_edges()
                    if num_edges == 0:
                        episode_terminated_early = True
                        pbar.close()
                        break
        
        # Apply heavy penalization if episode terminated early with 0 edges
        if episode_terminated_early:
            final_reward = - 1000.0  # Heavy penalization
        else:
            final_reward = self.reward_fn(graph)
        
        experiences = [
            Experience(
                graph_state=entry['graph'],
                visit_distributions=entry['visits'],
                final_reward=final_reward,
                step=entry['step']
            )
            for entry in trajectory
        ]
        
        # Clear trajectory to free memory
        trajectory.clear()
        
        # Clear MCTS tree after episode
        if hasattr(self.mcts, 'clear_caches'):
            self.mcts.clear_caches()
        elif hasattr(self.mcts, 'reset_trees'):
            self.mcts.reset_trees()
        
        return experiences
    
    def generate_episodes(self, num_episodes: int) -> List[Experience]:
        """Generate multiple episodes."""
        all_experiences = []

        for i in trange(num_episodes, desc="[SelfPlay] Episodes", unit="ep"):
            episode = self.generate_episode()
            all_experiences.extend(episode)

            if (i + 1) % 10 == 0:
                avg_reward = np.mean([exp.final_reward for exp in episode])
                print(f"Episode {i+1}/{num_episodes}, Avg Reward: {avg_reward:.3f}, Steps: {len(episode)}")
            
            # Periodic garbage collection every 5 episodes
            if (i + 1) % 5 == 0:
                gc.collect()

        return all_experiences
