"""
Fast MCTS - Cython Backend  
Version 4.0: Synchronized State Evolution (Single Shared Tree)
All heads share one tree structure with per-head statistics at each node.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import threading

try:
    from . import mcts_fast
    HAS_CYTHON = True
except ImportError as e:
    HAS_CYTHON = False
    raise ImportError(f"Cython required: {e}")

from env.network_fast import CythonNASGraph


class _CHeadChildProxy:
    """Proxy for a child node's per-head statistics."""
    __slots__ = ('_engine', '_node_idx', '_head_id', '_action_idx')
    
    def __init__(self, engine, node_idx: int, head_id: int, action_idx: int):
        self._engine = engine
        self._node_idx = node_idx
        self._head_id = head_id
        self._action_idx = action_idx
    
    @property
    def visit_count(self) -> int:
        return self._engine.get_head_action_visits(self._node_idx, self._head_id, self._action_idx)
    
    @property
    def virtual_loss(self) -> int:
        return self._engine.get_head_action_virtual_loss(self._node_idx, self._head_id, self._action_idx)
    
    @property
    def total_value(self) -> float:
        return self._engine.get_head_action_total_value(self._node_idx, self._head_id, self._action_idx)


class _CJointRootProxy:
    """
    Root proxy providing per-head views into the shared tree.
    """
    __slots__ = ('_engine', '_head_actions', '_children_cache')
    
    def __init__(self, engine, head_actions: List[List[Tuple[int, int]]]):
        self._engine = engine
        self._head_actions = head_actions  # List of action tuples per head
        self._children_cache = {}  # (head_id, action) -> proxy
    
    def get_head_children(self, head_id: int) -> Dict[Tuple[int, int], '_CHeadChildProxy']:
        """Get children dict for specific head at root."""
        if head_id not in self._children_cache:
            self._children_cache[head_id] = {}
            root_idx = self._engine.root_idx
            n_children = self._engine.root_num_children
            
            for i in range(n_children):
                child_idx = self._engine.root_child_idx(i)
                # Get the action this head took to reach this child
                action_idx = self._engine.get_head_action_at_node(child_idx, head_id)
                if action_idx >= 0 and action_idx < len(self._head_actions[head_id]):
                    action = self._head_actions[head_id][action_idx]
                    # Pass root_idx as parent, since that's where the action stats are stored
                    self._children_cache[head_id][action] = _CHeadChildProxy(
                        self._engine, root_idx, head_id, action_idx
                    )
        
        return self._children_cache[head_id]


class CythonSynchronizedMCTS:
    """
    Multi-head MCTS with synchronized state evolution.
    All heads share a single tree structure; they move in lockstep through joint trajectories.
    """
    
    def __init__(self, 
                 node_subsets: List[List[int]],
                 n_nodes: int,
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 virtual_loss: int = 3,
                 max_tree_nodes: int = 5000000,
                 **kwargs):
        
        self.n_heads = len(node_subsets)
        self.n_nodes = n_nodes
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        # Build action spaces for each head
        self.head_actions: List[List[Tuple[int, int]]] = []
        for subset in node_subsets:
            actions = [(u, v) for u in sorted(subset) 
                          for v in sorted(subset) if u != v]
            self.head_actions.append(actions)
        
        self.n_actions_per_head = [len(a) for a in self.head_actions]
        
        # Single shared Cython engine
        self._engine = mcts_fast.MCTSEngine(
            max_nodes=max_tree_nodes,
            n_heads=self.n_heads,
            c_puct=c_puct,
            virtual_loss=virtual_loss,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )
        
        # Pass all action spaces to Cython
        self._engine.set_head_action_spaces(self.head_actions)
        self._engine.initialize_tree(self.n_actions_per_head)
        
        # Threading lock for tree access (though tree updates are mostly in Cython)
        self._lock = threading.Lock()
        self.total_simulations = 0
    
    @property
    def root(self) -> _CJointRootProxy:
        """Get root proxy providing per-head child access."""
        return _CJointRootProxy(self._engine, self.head_actions)
    
    def search(self, state: CythonNASGraph, evaluator, num_simulations: int) -> List[np.ndarray]:
        """
        Run synchronized search: all heads move together through num_simulations joint trajectories.
        Returns list of visit distributions, one per head.
        """
        with self._lock:
            self._engine.search(state, evaluator, num_simulations)
            self.total_simulations += num_simulations
            
            # Extract per-head visit distributions from shared tree root
            results = []
            for h in range(self.n_heads):
                visits = np.zeros(self.n_actions_per_head[h], dtype=np.float64)
                self._engine.get_visit_distribution_for_head(h, visits)
                results.append(visits.astype(np.float32))
            
            return results
    
    def select_actions(self, temperature: float = 1.0) -> List[Optional[Tuple[int, int]]]:
        """Select action for each head based on their respective visit counts at root."""
        actions = []
        for h in range(self.n_heads):
            visits = np.zeros(self.n_actions_per_head[h], dtype=np.float64)
            self._engine.get_visit_distribution_for_head(h, visits)
            
            if visits.sum() == 0:
                actions.append(None)
                continue
            
            if temperature == 0:
                idx = int(np.argmax(visits))
            else:
                probs = visits ** (1.0 / temperature)
                probs /= probs.sum()
                idx = np.random.choice(len(probs), p=probs)
            
            actions.append(self.head_actions[h][idx])
        
        return actions
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get training stats for all heads."""
        results = []
        for h in range(self.n_heads):
            visits = np.zeros(self.n_actions_per_head[h], dtype=np.float64)
            self._engine.get_visit_distribution_for_head(h, visits)
            
            probs = visits[visits > 0]
            entropy = -np.sum(probs * np.log(probs + 1e-10)) if len(probs) > 0 else 0.0
            
            results.append({
                'visit_distribution': visits.astype(np.float32),
                'policy_entropy': float(entropy),
                'num_children': self._engine.root_num_children_for_head(h),
                'total_visits': int(visits.sum())
            })
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'tree_size': self._engine.size,
            'total_simulations': self.total_simulations,
            'n_heads': self.n_heads
        }
    
    def clear_caches(self):
        """Reset tree for new episode."""
        self._engine.clear_tree()
        self.total_simulations = 0


# Backward compatibility: HeadMCTS now refers to the synchronized system
# If main.py expects individual heads, we provide a shim or use this directly
HeadMCTS = CythonSynchronizedMCTS