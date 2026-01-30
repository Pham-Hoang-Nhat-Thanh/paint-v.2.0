"""
Multi-Head MCTS v4.0 - Cross-Head Batching (Sequential per Head)

Changes:
- Batching happens ACROSS heads, not within them
- Each head runs sequential MCTS: select → [batch eval with other heads] → backup
- Preserves AlphaZero-style correctness per head
- Maximizes GPU utilization via cross-head batching
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Set
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

from .fast_mcts import CythonHeadMCTS as HeadMCTS


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


class SimulationStep:
    """Container for one simulation step across all heads."""
    def __init__(self):
        self.pending_heads: List[int] = []
        self.leaf_indices: List[int] = []
        self.states: List[Any] = []
        self.is_roots: List[bool] = []
        self.results_ready = threading.Event()
        self.policies: Optional[List[np.ndarray]] = None
        self.values: Optional[List[float]] = None


class MultiHeadMCTS:
    def __init__(self, 
                 node_subsets: List[List[int]],
                 n_nodes: int,
                 c_puct: float = 1.0,
                 num_threads: int = 1,
                 parallel_heads: bool = True,
                 max_concurrent_heads: Optional[int] = None,
                 **head_kwargs):
        self.n_heads = len(node_subsets)
        self.num_threads = num_threads
        self.parallel_heads = parallel_heads
        self.max_concurrent_heads = max_concurrent_heads
        self.evaluator = None
        
        # Validate and create heads
        for subset in node_subsets:
            if not subset:
                raise ValueError("Empty node subset")
            for idx in subset:
                if not (0 <= idx < n_nodes):
                    raise ValueError(f"Invalid node index {idx}")
        
        self.heads = [
            HeadMCTS(head_id=i, node_indices=subset, c_puct=c_puct, **head_kwargs)
            for i, subset in enumerate(node_subsets)
        ]
        
        # Threading for head parallelism
        self._head_executor = None
        if parallel_heads and self.n_heads > 1:
            max_workers = min(max_concurrent_heads or self.n_heads, self.n_heads)
            self._head_executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Synchronization primitives for cross-head batching
        self._step_barrier = threading.Barrier(self.n_heads) if parallel_heads else None
        self._current_step: Optional[SimulationStep] = None
        self._step_lock = threading.Lock()
    
    def set_evaluator(self, 
                     policy_fn: Callable = None, 
                     value_fn: Callable = None,
                     evaluator: Evaluator = None,
                     batched: bool = False):
        if evaluator is not None:
            self.evaluator = evaluator
        elif policy_fn is not None and value_fn is not None:
            if batched:
                self.evaluator = BatchedEvaluator(policy_fn, value_fn)
            else:
                self.evaluator = SimpleEvaluator(policy_fn, value_fn)
        else:
            raise ValueError("Must provide evaluator or (policy_fn, value_fn)")
    
    def search(self, state: Any, num_simulations: int) -> List[np.ndarray]:
        """
        Run MCTS with cross-head batching.
        Each head runs sequential simulations, but evaluations are batched across heads.
        """
        if self.evaluator is None:
            raise ValueError("Must call set_evaluator before search")
        
        print(f"Starting cross-head batched search: {self.n_heads} heads, "
              f"{num_simulations} simulations each, "
              f"batch size = {self.n_heads} (one per head)")
        
        if not self.parallel_heads or self.n_heads == 1:
            # Sequential mode: simple loop
            for sim in range(num_simulations):
                self._run_simulation_sequential(state)
        else:
            # Parallel mode: barrier synchronization
            self._run_simulation_parallel(state, num_simulations)
        
        # Extract visit distributions
        results = []
        for head in self.heads:
            visits = np.zeros(head.n_actions, dtype=np.float32)
            head._engine.get_visit_distribution(visits)
            results.append(visits)
        
        return results
    
    def _run_simulation_sequential(self, state: Any):
        """Run one simulation step across all heads sequentially."""
        # Phase 1: Select
        selections = []
        for head_id, head in enumerate(self.heads):
            leaf_idx, features, is_root = head.select_leaf(state)
            selections.append((head_id, leaf_idx, features, is_root))
        
        # Phase 2: Batch Evaluate
        states = [s[2] for s in selections]
        head_ids = [s[0] for s in selections]
        policies, values = self.evaluator.evaluate(states, head_ids)
        
        # Phase 3: Expand and Backup (sequential per head, immediately)
        for i, (head_id, leaf_idx, features, is_root) in enumerate(selections):
            head = self.heads[head_id]
            head.expand_and_backup(leaf_idx, policies[i], values[i], state, is_root)
    
    def _run_simulation_parallel(self, state: Any, num_simulations: int):
        """Run simulations with heads in parallel, batching evaluations."""
        
        def head_worker(head_id: int, head: HeadMCTS, local_state: Any, n_sims: int):
            for sim in range(n_sims):
                # Step 1: Select (head acquires virtual loss)
                leaf_idx, features, is_root = head.select_leaf(local_state)
                
                # Register with global batch
                with self._step_lock:
                    if self._current_step is None:
                        self._current_step = SimulationStep()
                    step = self._current_step
                    step.pending_heads.append(head_id)
                    step.leaf_indices.append(leaf_idx)
                    step.states.append(features)
                    step.is_roots.append(is_root)
                    is_last = (len(step.pending_heads) == self.n_heads)
                
                if is_last:
                    # Last head to arrive: do batch evaluation
                    policies, values = self.evaluator.evaluate(step.states, step.pending_heads)
                    step.policies = policies
                    step.values = values
                    step.results_ready.set()
                else:
                    # Wait for evaluation
                    step.results_ready.wait()
                
                # Step 2: Get result and backup immediately (sequential per head)
                idx_in_batch = step.pending_heads.index(head_id)
                policy = step.policies[idx_in_batch]
                value = step.values[idx_in_batch]
                
                head.expand_and_backup(leaf_idx, policy, value, local_state, is_root)
                
                # Step 3: Barrier to ensure all heads finish before next iteration
                try:
                    self._step_barrier.wait()
                except threading.BrokenBarrierError:
                    pass
                
                # Clean up step object after all threads pass barrier
                with self._step_lock:
                    if head_id == 0:  # One head resets
                        self._current_step = None
        
        # Launch workers
        futures = []
        for head_id, head in enumerate(self.heads):
            fut = self._head_executor.submit(head_worker, head_id, head, state, num_simulations)
            futures.append(fut)
        
        # Wait for completion
        for fut in futures:
            fut.result()
    
    def select_actions(self, temperature: float = 1.0) -> List[Optional[Tuple[int, int]]]:
        return [head.select_action(temperature) for head in self.heads]
    
    def batched_commit(self, 
                      state: Any,
                      actions: List[Optional[Tuple[int, int]]],
                      use_advanced_resolution: bool = False) -> Any:
        if use_advanced_resolution:
            return self._batched_commit_advanced(state, actions)
        else:
            return self._batched_commit_simple(state, actions)
    
    def _batched_commit_simple(self, state: Any, 
                               actions: List[Optional[Tuple[int, int]]]) -> Any:
        new_state = state.copy()
        toggle = new_state.toggle_edge
        
        action_priorities = []
        for head_id, action in enumerate(actions):
            if action is not None:
                head = self.heads[head_id]
                root = head.root
                if root and action in root.children:
                    child = root.children[action]
                    n = child.visit_count + child.virtual_loss
                    q = (child.total_value - child.virtual_loss) / n if n > 0 else 0.0
                    priority = child.visit_count * max(0.0, q)
                    action_priorities.append((priority, head_id, action))
        
        action_priorities.sort(key=lambda x: x[0], reverse=True)
        
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
        new_state = state.copy()
        
        action_list = []
        for head_id, action in enumerate(actions):
            if action is not None:
                head = self.heads[head_id]
                root = head.root
                if root and action in root.children:
                    child = root.children[action]
                    n = child.visit_count + child.virtual_loss
                    q = (child.total_value - child.virtual_loss) / n if n > 0 else 0.0
                    priority = child.visit_count * max(0.0, q)
                    action_list.append((priority, head_id, action))
        
        action_list.sort(key=lambda x: x[0], reverse=True)
        
        applied = []
        for priority, head_id, action in action_list:
            test_state = new_state.copy()
            if test_state.toggle_edge(*action):
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
    
    def clear_caches(self):
        for head in self.heads:
            head._valid_cache.clear()
            head._policy_cache.clear()
    
    def reset_trees(self):
        for head in self.heads:
            with head._lock:
                head._engine.clear()
                head._valid_cache.clear()
                head._policy_cache.clear()
                head.total_simulations = 0
                head._current_root_hash = None
    
    def __del__(self):
        if self._head_executor is not None:
            self._head_executor.shutdown(wait=False)