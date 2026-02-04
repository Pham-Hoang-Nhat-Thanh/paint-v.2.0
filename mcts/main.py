"""
Multi-Head MCTS v5.0 - Synchronized State Evolution (Single Shared Tree)

Changes:
- Single shared tree structure with per-head statistics
- All heads search in lockstep: joint selection → shared expansion → multi-head backup
- Conceptually equivalent to one MCTS on the joint action space
- Each head maintains independent visit counts/Q-values for its local actions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
from abc import ABC, abstractmethod

# Import the synchronized engine instead of individual heads
from .fast_mcts import CythonSynchronizedMCTS


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, states: List[Any], head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        pass


class SimpleEvaluator(Evaluator):
    def __init__(self, policy_fn: Callable, value_fn: Callable):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
    
    def evaluate(self, states: List[Any], head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        policies = [self.policy_fn(state, head_id) for state, head_id in zip(states, head_ids)]
        values = [self.value_fn(state) for state in states]
        return policies, values


class BatchedEvaluator(Evaluator):
    def __init__(self, policy_fn: Callable, value_fn: Callable, max_batch_size: int = 64):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.max_batch_size = max_batch_size
    
    def evaluate(self, states: List[Any], head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        all_policies = []
        all_values = []
        
        for i in range(0, len(states), self.max_batch_size):
            batch_states = states[i:i+self.max_batch_size]
            batch_heads = head_ids[i:i+self.max_batch_size]
            policies = self.policy_fn(batch_states, batch_heads)
            values = self.value_fn(batch_states)
            all_policies.extend(policies)
            all_values.extend(values)
        
        return all_policies, all_values


class NetworkEvaluator(Evaluator):
    """
    Optimized evaluator that calls predict_batch ONCE to get both policies and values.
    This is the preferred evaluator for MCTS with neural network guidance.
    """
    def __init__(self, network):
        self.network = network
    
    def evaluate(self, states: List[Any], head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Single network call returns all head policies + values.
        For synchronized MCTS, states typically has length 1.
        """
        return self.network.predict_batch(states, head_ids)


class MultiHeadMCTS:
    def __init__(self, 
                 node_subsets: List[List[int]],
                 n_nodes: int,
                 c_puct: float = 1.0,
                 max_tree_nodes: int = 5000000,
                 **kwargs):
        """
        Initialize synchronized multi-head MCTS.
        
        Args:
            node_subsets: List of node subsets, one per head
            n_nodes: Total number of nodes in the graph
            c_puct: Exploration constant
            max_tree_nodes: Maximum nodes in the shared tree
        """
        self.n_heads = len(node_subsets)
        
        # Validate subsets
        for subset in node_subsets:
            if not subset:
                raise ValueError("Empty node subset")
            for idx in subset:
                if not (0 <= idx < n_nodes):
                    raise ValueError(f"Invalid node index {idx}")
        
        # Create the synchronized engine with single shared tree
        self.engine = CythonSynchronizedMCTS(
            node_subsets=node_subsets,
            n_nodes=n_nodes,
            c_puct=c_puct,
            max_tree_nodes=max_tree_nodes,
            **kwargs
        )
        
        self.evaluator = None
    
    def set_evaluator(self, 
                     policy_fn: Callable = None, 
                     value_fn: Callable = None,
                     evaluator: Evaluator = None,
                     network = None,
                     batched: bool = False):
        """
        Set the evaluator for neural network guidance.
        
        Preferred: pass network directly for optimal single-call evaluation.
        """
        if evaluator is not None:
            self.evaluator = evaluator
        elif network is not None:
            # Optimal: single predict_batch call for policies + values
            self.evaluator = NetworkEvaluator(network)
        elif policy_fn is not None and value_fn is not None:
            if batched:
                self.evaluator = BatchedEvaluator(policy_fn, value_fn)
            else:
                self.evaluator = SimpleEvaluator(policy_fn, value_fn)
        else:
            raise ValueError("Must provide evaluator, network, or (policy_fn, value_fn)")
    
    def search(self, state: Any, num_simulations: int) -> List[np.ndarray]:
        """
        Run synchronized MCTS search.
        
        All heads move in lockstep through the tree, maintaining per-head statistics
        at each node but sharing the same state evolution trajectory.
        
        Returns:
            List of visit distributions (one per head) at the root
        """
        if self.evaluator is None:
            raise ValueError("Must call set_evaluator before search")
        
        # Single call to the synchronized engine
        # This runs the lockstep algorithm entirely in Cython
        return self.engine.search(state, self.evaluator, num_simulations)
    
    def select_actions(self, temperature: float = 1.0) -> List[Optional[Tuple[int, int]]]:
        """
        Select actions for all heads based on their respective visit counts at root.
        
        Returns:
            List of actions (as (u,v) tuples), one per head
        """
        return self.engine.select_actions(temperature)
    
    def batched_commit(self, 
                      state: Any,
                      actions: List[Optional[Tuple[int, int]]],
                      use_advanced_resolution: bool = False) -> Any:
        """
        Apply actions from all heads to create the next state.
        
        Since heads operate on the same state with (mostly) disjoint action sets,
        we apply all valid actions to produce the next architecture.
        """
        if use_advanced_resolution:
            return self._batched_commit_advanced(state, actions)
        else:
            return self._batched_commit_simple(state, actions)
    
    def _batched_commit_simple(self, state: Any, 
                               actions: List[Optional[Tuple[int, int]]]) -> Any:
        """Simple greedy commit: apply actions in priority order by visit count."""
        new_state = state.copy()
        toggle = new_state.toggle_edge
        
        # Get priorities from the shared tree's per-head statistics
        action_priorities = []
        root = self.engine.root
        
        for head_id, action in enumerate(actions):
            if action is not None:
                head_children = root.get_head_children(head_id)
                if action in head_children:
                    child = head_children[action]
                    n = child.visit_count + child.virtual_loss
                    q = (child.total_value - child.virtual_loss) / n if n > 0 else 0.0
                    priority = child.visit_count * max(0.0, q)
                    action_priorities.append((priority, head_id, action))
        
        # Sort by priority (visit count * Q-value)
        action_priorities.sort(key=lambda x: x[0], reverse=True)
        
        # Apply in order, checking for conflicts
        applied = set()
        for _, _, (u, v) in action_priorities:
            edge = (u, v)
            reverse_edge = (v, u)
            
            if edge not in applied and reverse_edge not in applied:
                if toggle(u, v):
                    applied.add(edge)
        
        return new_state
    
    def _batched_commit_advanced(self, state: Any,
                                 actions: List[Optional[Tuple[int, int]]]) -> Any:
        """Advanced commit with conflict resolution."""
        new_state = state.copy()
        
        action_list = []
        root = self.engine.root
        
        for head_id, action in enumerate(actions):
            if action is not None:
                head_children = root.get_head_children(head_id)
                if action in head_children:
                    child = head_children[action]
                    n = child.visit_count + child.virtual_loss
                    q = (child.total_value - child.virtual_loss) / n if n > 0 else 0.0
                    priority = child.visit_count * max(0.0, q)
                    action_list.append((priority, head_id, action))
        
        action_list.sort(key=lambda x: x[0], reverse=True)
        
        applied = []
        for priority, head_id, action in action_list:
            test_state = new_state.copy()
            if test_state.toggle_edge(*action):
                # Check for conflicts with higher priority actions
                conflicts = False
                for other_priority, _, other_action in action_list:
                    if other_action != action and other_action not in [a for _, _, a in applied]:
                        if not test_state.is_valid_add(*other_action) and other_priority > priority:
                            conflicts = True
                            break
                
                if not conflicts:
                    new_state = test_state
                    applied.append((priority, head_id, action))
        
        return new_state
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get training statistics for all heads."""
        return self.engine.get_training_data()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return self.engine.get_stats()
    
    def clear_caches(self):
        """Reset the tree for a new episode."""
        self.engine.clear_caches()
    
    def reset_trees(self):
        """Reset the shared tree."""
        self.engine.clear_caches()