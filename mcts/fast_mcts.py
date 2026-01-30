"""
Fast MCTS - Cython Backend
Version 3.0: Cross-Head Batching Support (Sequential per Head)
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


class _CChildProxy:
    """Lazy proxy for tree statistics."""
    __slots__ = ('_engine', '_idx', '_q_cached')
    
    def __init__(self, engine, idx: int):
        self._engine = engine
        self._idx = idx
        self._q_cached = None
    
    @property
    def visit_count(self) -> int:
        return self._engine.node_visits(self._idx)
    
    @property
    def virtual_loss(self) -> int:
        return self._engine.node_virtual_loss(self._idx)
    
    @property
    def total_value(self) -> float:
        return self._engine.node_total_value(self._idx)
    
    def q_value(self) -> float:
        if self._q_cached is None:
            self._q_cached = self._engine.node_q(self._idx)
        return self._q_cached


class _CRootProxy:
    """Root proxy for MultiHeadMCTS compatibility."""
    __slots__ = ('_engine', '_actions', '_children_cache')
    
    def __init__(self, engine, actions):
        self._engine = engine
        self._actions = actions
        self._children_cache = None
    
    @property
    def children(self) -> Dict[Tuple[int, int], '_CChildProxy']:
        if self._children_cache is None:
            self._children_cache = {}
            n_children = self._engine.root_num_children
            
            for i in range(n_children):
                child_idx = self._engine.root_child_idx(i)
                action_id = self._engine.node_action_id(child_idx)
                action = self._actions[action_id]
                self._children_cache[action] = _CChildProxy(self._engine, child_idx)
        return self._children_cache
    
    @property
    def visit_count(self) -> int:
        return self._engine.root_visits


class CythonHeadMCTS:
    """
    Head MCTS with sequential simulation steps for cross-head batching.
    """
    
    def __init__(self, 
                 head_id: int,
                 node_indices: List[int],
                 c_puct: float = 1.0,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 virtual_loss: int = 3,
                 cache_size: int = 10000,
                 max_tree_nodes: int = 5000000,
                 **kwargs):
        
        self.head_id = head_id
        self.node_indices = sorted(node_indices)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.virtual_loss_value = virtual_loss
        
        # Action space
        self.actions = tuple((u, v) for u in self.node_indices 
                            for v in self.node_indices if u != v)
        self.n_actions = len(self.actions)
        self.action_to_idx = {a: i for i, a in enumerate(self.actions)}
        
        # Cython engine
        self._engine = mcts_fast.MCTSEngine(
            max_nodes=max_tree_nodes,
            c_puct=c_puct,
            virtual_loss=virtual_loss,
            n_actions=self.n_actions,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon
        )
        
        # Pass action mapping to C
        self._engine.set_action_space(self.actions)
        
        # Compatibility mocks
        self._valid_cache = type('MockCache', (), {'clear': lambda: None})()  
        self._policy_cache = type('MockCache', (), {'clear': lambda: None})()
        self._lock = threading.Lock()
        
        self.total_simulations = 0
        self._current_root_hash = None
    
    @property
    def root(self) -> Optional[_CRootProxy]:
        if self._engine.size == 0:
            return None
        return _CRootProxy(self._engine, self.actions)
    
    def select_leaf(self, state: CythonNASGraph):
        """
        Select leaf for evaluation.
        Returns: (leaf_idx, features, is_root)
        """
        return self._engine.select_leaf(state)
    
    def expand_and_backup(self, leaf_idx: int, policy: np.ndarray, value: float, 
                         state: CythonNASGraph, is_root: bool = False):
        """
        Expand leaf and backup value. Sequential: updates tree immediately.
        """
        self._engine.expand_and_backup(leaf_idx, policy, value, state, is_root)
        self.total_simulations += 1
    
    def select_action(self, temperature: float = 1.0) -> Optional[Tuple[int, int]]:
        """Greedy or sampling selection based on visit counts."""
        if self._engine.size == 0:
            return None
        
        visits = np.zeros(self.n_actions, dtype=np.float64)
        self._engine.get_visit_distribution(visits)
        
        if visits.sum() == 0:
            return None
        
        if temperature == 0:
            idx = int(np.argmax(visits))
        else:
            probs = visits ** (1.0 / temperature)
            probs /= probs.sum()
            idx = np.random.choice(len(probs), p=probs)
        
        return self.actions[idx]
    
    def get_training_data(self) -> Dict[str, Any]:
        """Get training stats."""
        visits = np.zeros(self.n_actions, dtype=np.float64)
        self._engine.get_visit_distribution(visits)
        
        probs = visits[visits > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-10)) if len(probs) > 0 else 0.0
        
        return {
            'visit_distribution': visits.astype(np.float32),
            'policy_entropy': float(entropy),
            'num_children': self._engine.root_num_children,
            'total_visits': self._engine.root_visits
        }
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'tree_size': self._engine.size,
            'total_simulations': self.total_simulations
        }
    
    def clear_caches(self):
        """Reset for new episode."""
        self._engine.clear()
        self._current_root_hash = None


# Backward compatibility alias
HeadMCTS = CythonHeadMCTS