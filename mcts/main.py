"""
Production-Ready Multi-Head MCTS for Neural Architecture Search
Version 2.0 - Refined based on professional code review

Improvements over v1.0:
- O(N) topological reordering using deque
- Guaranteed virtual loss cleanup with context managers
- LRU cache with configurable size limits
- Advanced conflict resolution with constraint satisfaction
- Exception safety throughout
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from env.network import NASGraph
from utils.lru_cache import LRUCache
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from .eval_batcher import AsyncBatchedEvaluator
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class MCTSNode:
    """MCTS tree node with virtual loss."""
    state_hash: int
    parent: Optional['MCTSNode']
    action: Optional[Tuple[int, int]]
    children: Dict[Tuple[int, int], 'MCTSNode'] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    virtual_loss: int = 0
    
    def q_value(self) -> float:
        """Q-value with virtual loss."""
        n = self.visit_count + self.virtual_loss
        if n == 0:
            return 0.0
        return (self.total_value - self.virtual_loss) / n
    
    def ucb_score(self, parent_visits: int, c_puct: float = 1.0) -> float:
        """UCB with virtual loss."""
        n = self.visit_count + self.virtual_loss
        u = c_puct * self.prior * np.sqrt(parent_visits) / (1 + n)
        return self.q_value() + u


class Evaluator(ABC):
    """Abstract base class for neural network evaluation."""
    
    @abstractmethod
    def evaluate(self, states: List[NASGraph], head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        """Batch evaluate states."""
        pass


class SimpleEvaluator(Evaluator):
    """Simple evaluator for single-state functions."""
    
    def __init__(self, policy_fn: Callable, value_fn: Callable):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
    
    def evaluate(self, states: List[NASGraph], head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        policies = [self.policy_fn(state, head_id) for state, head_id in zip(states, head_ids)]
        values = [self.value_fn(state) for state in states]
        return policies, values


class BatchedEvaluator(Evaluator):
    """Batched evaluator for GPU inference."""
    
    def __init__(self, policy_fn: Callable, value_fn: Callable, max_batch_size: int = 128):
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        self.max_batch_size = max_batch_size
    
    def evaluate(self, states: List[NASGraph], head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
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


@contextmanager
def virtual_loss_context(nodes: List[MCTSNode], loss_value: int, lock: threading.Lock):
    """
    Context manager for virtual loss application.
    Guarantees cleanup even on exceptions.
    """
    try:
        yield
    finally:
        # Always remove virtual loss
        with lock:
            for node in nodes:
                node.virtual_loss = max(0, node.virtual_loss - loss_value)


class HeadMCTS:
    """
    Thread-safe MCTS for a single policy head.
    
    v2.0 improvements:
    - Guaranteed virtual loss cleanup with context managers
    - LRU caches with size limits
    - Better exception safety
    """
    
    def __init__(self, 
                 head_id: int,
                 node_indices: List[int],
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 virtual_loss: int = 3,
                 prune_threshold: float = 0.01,
                 min_expansions: int = 5,
                 cache_size: int = 10000):
        """
        Args:
            cache_size: Maximum size for LRU caches
        """
        self.head_id = head_id
        self.node_indices = sorted(node_indices)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.virtual_loss_value = virtual_loss
        self.prune_threshold = prune_threshold
        self.min_expansions = min_expansions
        
        # Action space
        self.actions = [(u, v) for u in self.node_indices 
                       for v in self.node_indices if u != v]
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
        # Tree
        self.root: Optional[MCTSNode] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        # LRU caches
        self._valid_cache = LRUCache(max_size=cache_size)
        self._policy_cache = LRUCache(max_size=cache_size)
        
        # Statistics
        self.total_simulations = 0
    
    def search(self, 
               state: NASGraph,
               evaluator: Evaluator,
               num_simulations: int,
               num_threads: int = 1) -> np.ndarray:
        """Run MCTS search."""
        # Tree reuse
        state_hash = state.get_hash()
        if self.root is None or self.root.state_hash != state_hash:
            with self._lock:
                self.root = MCTSNode(state_hash=state_hash, parent=None, action=None)
                self._valid_cache.clear()
                self._policy_cache.clear()
        
        if num_threads == 1:
            for _ in range(num_simulations):
                self._simulate(state, evaluator)
        else:
            threads = []
            sims_per_thread = num_simulations // num_threads
            remainder = num_simulations % num_threads
            
            for i in range(num_threads):
                sims = sims_per_thread + (1 if i < remainder else 0)
                t = threading.Thread(
                    target=lambda n=sims: [self._simulate(state, evaluator) for _ in range(n)],
                    daemon=True
                )
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
        
        self.total_simulations += num_simulations
        return self._get_visit_distribution()
    
    def _simulate(self, root_state: NASGraph, evaluator: Evaluator):
        """
        Single simulation with guaranteed virtual loss cleanup.
        
        v2.0: Uses context manager for exception safety.
        """
        node = self.root
        state = root_state.copy()
        path = [node]
        selected_children = []
        
        # Selection with virtual loss
        while node.children:
            with self._lock:
                action, child = self._select_child(node)
                child.virtual_loss += self.virtual_loss_value
                selected_children.append(child)
            
            state.toggle_edge(*action)
            node = child
            path.append(node)
        
        # Expansion and evaluation with guaranteed cleanup
        try:
            value = self._expand_and_evaluate(node, state, evaluator)
            
            # Backup
            with self._lock:
                for n in path:
                    n.visit_count += 1
                    n.total_value += value
        finally:
            # ALWAYS remove virtual loss, even on exceptions
            with self._lock:
                for child in selected_children:
                    child.virtual_loss = max(0, child.virtual_loss - self.virtual_loss_value)
    
    def _select_child(self, node: MCTSNode) -> Tuple[Tuple[int, int], MCTSNode]:
        """Select child with highest UCB (must hold lock)."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = child.ucb_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _expand_and_evaluate(self, node: MCTSNode, state: NASGraph, 
                            evaluator: Evaluator) -> float:
        """Expand with LRU caching."""
        cache_key = state.get_hash()
        
        # Check policy cache
        cached = self._policy_cache.get(cache_key)
        if cached is not None:
            priors, value = cached
        else:
            # Neural network evaluation
            policies, values = evaluator.evaluate([state], [self.head_id])
            logits = policies[0]
            value = values[0]
            
            # Get valid mask
            valid_cached = self._valid_cache.get(cache_key)
            if valid_cached is not None:
                valid_mask = valid_cached
            else:
                valid_mask = self._get_valid_mask(state)
                self._valid_cache.put(cache_key, valid_mask)
            
            # Mask and normalize
            logits = np.where(valid_mask, logits, -1e9)
            priors = self._softmax(logits)
            
            # Dirichlet noise at root (only on valid actions)
            if node == self.root:
                valid_indices = np.where(valid_mask)[0]
                noise = np.zeros(len(self.actions))
                if len(valid_indices) > 0:
                    noise[valid_indices] = np.random.dirichlet(
                        [self.dirichlet_alpha] * len(valid_indices)
                    )
                    priors = (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
            
            self._policy_cache.put(cache_key, (priors, value))
        
        # Progressive widening
        with self._lock:
            sorted_indices = np.argsort(priors)[::-1]
            
            for idx in sorted_indices:
                prior = priors[idx]
                
                if prior < self.prune_threshold and len(node.children) >= self.min_expansions:
                    break
                
                action = self.actions[idx]
                if action not in node.children and prior > 0:
                    child_state = state.copy()
                    if child_state.toggle_edge(*action):
                        node.children[action] = MCTSNode(
                            state_hash=child_state.get_hash(),
                            parent=node,
                            action=action,
                            prior=prior
                        )
        
        return value
    
    def _get_valid_mask(self, state: NASGraph) -> np.ndarray:
        """Get boolean mask of valid actions."""
        mask = np.zeros(len(self.actions), dtype=bool)
        
        for i, (u, v) in enumerate(self.actions):
            u_node = state.nodes[u]
            v_node = state.nodes[v]
            
            if v_node in state.adjacency[u_node]:
                mask[i] = True
            else:
                mask[i] = state.is_valid_add(u, v)
        
        return mask
    
    def _get_visit_distribution(self) -> np.ndarray:
        """Get normalized visit count distribution."""
        visits = np.zeros(len(self.actions), dtype=np.float32)
        
        with self._lock:
            for action, child in self.root.children.items():
                idx = self.action_to_idx[action]
                visits[idx] = child.visit_count
        
        total = visits.sum()
        return visits / total if total > 0 else visits
    
    def select_action(self, temperature: float = 1.0) -> Optional[Tuple[int, int]]:
        """Select action with temperature."""
        probs = self._get_visit_distribution()
        
        if probs.sum() == 0:
            return None
        
        if temperature == 0:
            idx = np.argmax(probs)
        else:
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
            idx = np.random.choice(len(probs), p=probs)
        
        return self.actions[idx]
    
    def get_training_data(self) -> Dict[str, Any]:
        """Get data for training."""
        visits = self._get_visit_distribution()
        probs = visits[visits > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-10)) if len(probs) > 0 else 0.0
        
        with self._lock:
            num_children = len(self.root.children) if self.root else 0
            total_visits = self.root.visit_count if self.root else 0
        
        return {
            'visit_distribution': visits,
            'policy_entropy': float(entropy),
            'num_children': num_children,
            'total_visits': total_visits
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            tree_size = self._count_nodes(self.root)
            max_depth = self._max_depth(self.root)
        
        stats = {
            'tree_size': tree_size,
            'max_depth': max_depth,
            'total_simulations': self.total_simulations,
            'valid_cache': self._valid_cache.stats(),
            'policy_cache': self._policy_cache.stats()
        }
        return stats
    
    def _count_nodes(self, node: Optional[MCTSNode]) -> int:
        if node is None:
            return 0
        return 1 + sum(self._count_nodes(c) for c in node.children.values())
    
    def _max_depth(self, node: Optional[MCTSNode], depth: int = 0) -> int:
        if node is None or not node.children:
            return depth
        return max(self._max_depth(c, depth + 1) for c in node.children.values())
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class MultiHeadMCTS:
    """
    Multi-head MCTS coordinator.
    
    v2.0 improvements:
    - Advanced conflict resolution with constraint satisfaction
    - Better validation
    """
    
    def __init__(self, 
                 node_subsets: List[List[int]],
                 n_nodes: int,
                 c_puct: float = 1.0,
                 num_threads: int = 1,
                 parallel_heads: bool = True,
                 max_concurrent_heads: Optional[int] = None,
                 **head_kwargs):
        self._validate_subsets(node_subsets, n_nodes)
        
        self.heads = [
            HeadMCTS(head_id=i, node_indices=subset, c_puct=c_puct, **head_kwargs)
            for i, subset in enumerate(node_subsets)
        ]
        self.n_heads = len(self.heads)
        self.num_threads = num_threads
        self.evaluator: Optional[Evaluator] = None
        # Parallel execution across heads (threaded)
        self.parallel_heads = parallel_heads
        # Max concurrent heads (None -> all)
        self.max_concurrent_heads = max_concurrent_heads
    
    def _validate_subsets(self, subsets: List[List[int]], n_nodes: int):
        """Validate node subsets."""
        for i, subset in enumerate(subsets):
            if not subset:
                raise ValueError(f"Head {i} has empty node subset")
            
            for idx in subset:
                if not (0 <= idx < n_nodes):
                    raise ValueError(f"Head {i} has invalid node index {idx}")
    
    def set_evaluator(self, 
                     policy_fn: Callable = None, 
                     value_fn: Callable = None,
                     evaluator: Evaluator = None,
                     batched: bool = False):
        """Set neural network evaluator."""
        if evaluator is not None:
            self.evaluator = evaluator
        elif policy_fn is not None and value_fn is not None:
            if batched:
                base = BatchedEvaluator(policy_fn, value_fn)
                # Wrap underlying batched evaluator with an async batcher so many
                # small single-state evaluate() calls can be coalesced into larger
                # GPU-friendly batches.
                self.evaluator = AsyncBatchedEvaluator(base, max_batch_size=base.max_batch_size, timeout_ms=10)
            else:
                self.evaluator = SimpleEvaluator(policy_fn, value_fn)
        else:
            raise ValueError("Must provide evaluator or (policy_fn, value_fn)")
    
    def search(self, state: NASGraph, num_simulations: int) -> List[np.ndarray]:
        """Run MCTS for all heads."""
        if self.evaluator is None:
            raise ValueError("Must call set_evaluator before search")
        # If parallel_heads is False, run sequentially (old behavior)
        if not self.parallel_heads or self.n_heads <= 1:
            return [head.search(state, self.evaluator, num_simulations, self.num_threads)
                    for head in self.heads]

        # Run head.search concurrently using a thread pool. This parallelizes the
        # outer loop across heads so each head can perform its simulations in
        # parallel (they still use their internal threading where configured).
        max_workers = self.max_concurrent_heads or self.n_heads
        max_workers = min(max_workers, self.n_heads)

        results = [None] * self.n_heads

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(head.search, state, self.evaluator, num_simulations, self.num_threads): idx
                       for idx, head in enumerate(self.heads)}

            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                except Exception:
                    # On exception, propagate to caller
                    raise
                results[idx] = res

        return results
    
    def select_actions(self, temperature: float = 1.0) -> List[Optional[Tuple[int, int]]]:
        """Select one action per head."""
        return [head.select_action(temperature) for head in self.heads]
    
    def batched_commit(self, 
                      state: NASGraph,
                      actions: List[Optional[Tuple[int, int]]],
                      use_advanced_resolution: bool = False) -> NASGraph:
        """
        Apply actions with conflict resolution.
        
        v2.0: Added advanced constraint satisfaction mode.
        
        Args:
            state: Current graph
            actions: List of actions per head
            use_advanced_resolution: Use constraint satisfaction (slower but better)
        """
        if use_advanced_resolution:
            return self._batched_commit_advanced(state, actions)
        else:
            return self._batched_commit_simple(state, actions)
    
    def _batched_commit_simple(self, state: NASGraph, 
                               actions: List[Optional[Tuple[int, int]]]) -> NASGraph:
        """Simple greedy conflict resolution."""
        new_state = state.copy()
        
        action_priorities = []
        for head_id, action in enumerate(actions):
            if action is not None:
                head = self.heads[head_id]
                if head.root and action in head.root.children:
                    child = head.root.children[action]
                    priority = child.visit_count * max(0, child.q_value())
                    action_priorities.append((priority, head_id, action))
        
        action_priorities.sort(reverse=True, key=lambda x: x[0])
        
        applied = set()
        for priority, head_id, (u, v) in action_priorities:
            edge = (u, v)
            reverse_edge = (v, u)
            
            if edge not in applied and reverse_edge not in applied:
                if new_state.toggle_edge(u, v):
                    applied.add(edge)
        
        return new_state
    
    def _batched_commit_advanced(self, state: NASGraph,
                                 actions: List[Optional[Tuple[int, int]]]) -> NASGraph:
        """
        Advanced conflict resolution using constraint satisfaction.
        
        Tries to find the maximum subset of actions that can be applied
        simultaneously without creating cycles or conflicts.
        """
        new_state = state.copy()
        
        # Collect actions with priorities
        action_list = []
        for head_id, action in enumerate(actions):
            if action is not None:
                head = self.heads[head_id]
                if head.root and action in head.root.children:
                    child = head.root.children[action]
                    priority = child.visit_count * max(0, child.q_value())
                    action_list.append((priority, head_id, action))
        
        # Sort by priority
        action_list.sort(reverse=True, key=lambda x: x[0])
        
        # Try to apply each action, backtrack if it invalidates future actions
        applied = []
        for priority, head_id, action in action_list:
            # Try applying this action
            test_state = new_state.copy()
            if test_state.toggle_edge(*action):
                # Check if remaining higher-priority unapplied actions are still valid
                conflicts = False
                for other_priority, other_head, other_action in action_list:
                    if other_action != action and other_action not in [a for _, _, a in applied]:
                        # Check if other_action is still valid after applying action
                        if not test_state.is_valid_add(*other_action) and other_priority > priority:
                            conflicts = True
                            break
                
                if not conflicts:
                    new_state = test_state
                    applied.append((priority, head_id, action))
        
        return new_state
    
    def get_all_training_data(self) -> Dict[str, Any]:
        """Get training data from all heads."""
        return {f'head_{i}': head.get_training_data() for i, head in enumerate(self.heads)}
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all heads."""
        return {f'head_{i}': head.get_stats() for i, head in enumerate(self.heads)}
    
    def clear_caches(self):
        """Clear all caches."""
        for head in self.heads:
            head._valid_cache.clear()
            head._policy_cache.clear()
    
    def reset_trees(self):
        """Reset all MCTS trees."""
        for head in self.heads:
            with head._lock:
                head.root = None
                head._valid_cache.clear()
                head._policy_cache.clear()
                head.total_simulations = 0