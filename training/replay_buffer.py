import torch
from torch_geometric.data import Data, Batch
from typing import List, Tuple
import numpy as np
import random
import gc
from collections import deque
import math
from dataclasses import dataclass


@dataclass
class Experience:
    """Single experience tuple."""
    graph_state: any  # NASGraph
    visit_distributions: List[np.ndarray]
    final_reward: float
    step: int


class SumTree:
    """Binary sum tree for efficient prioritized sampling."""

    def __init__(self, capacity: int):
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity <<= 1
        # tree size = 2*capacity
        self.tree = [0.0] * (2 * self.capacity)

    def _propagate(self, idx: int, change: float):
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] += change
            parent //= 2

    def update(self, leaf_idx: int, value: float):
        # leaf_idx in [0, capacity-1]
        tree_idx = leaf_idx + self.capacity
        change = value - self.tree[tree_idx]
        self.tree[tree_idx] = value
        self._propagate(tree_idx, change)

    def total(self) -> float:
        return self.tree[1]

    def get(self, s: float) -> int:
        # Traverse the tree to find the leaf for cumulative sum s
        idx = 1
        while idx < self.capacity:
            left = idx * 2
            if self.tree[left] >= s:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        return idx - self.capacity


class ReplayBuffer:
    """
    Optimized experience replay buffer with optional Prioritized Experience Replay (PER).
    
    When `use_per` is True this implements a SumTree-based PER with circular eviction.
    """
    
    def __init__(self, max_size: int = 100000, min_size: int = 1000, use_per: bool = True,
                 per_alpha: float = 0.6, per_beta_start: float = 0.4, per_beta_frames: int = 100000):
        """
        If `use_per` is True, buffer uses a SumTree for prioritized sampling.
        Default hyperparameters follow common PER settings.
        """
        self.max_size = int(max_size)
        self.min_size = int(min_size)
        self.use_per = bool(use_per)
        self.per_alpha = float(per_alpha)
        self.per_beta_start = float(per_beta_start)
        self.per_beta_frames = int(per_beta_frames)

        # storage for experiences (circular)
        self._data = [None] * self.max_size
        self._pos = 0
        self._size = 0

        # SumTree for priorities
        if self.use_per:
            self._tree = SumTree(self.max_size)
            self._max_priority = 1.0

        self._add_count = 0
        self._cleanup_interval = 1000  # Cleanup every N additions
    
    def add(self, experience: Experience):
        # Clear graph caches before storing to reduce memory
        if hasattr(experience.graph_state, '_reach_cache'):
            experience.graph_state._reach_cache.clear()
        if hasattr(experience.graph_state, '_pyg_tensors'):
            experience.graph_state._pyg_tensors.clear()
        if hasattr(experience.graph_state, '_invalidate_caches'):
            experience.graph_state._invalidate_caches()

        # Store in circular buffer
        self._data[self._pos] = experience

        # Set priority for new sample to max_priority (so new samples are seen)
        if self.use_per:
            self._tree.update(self._pos, float(self._max_priority ** self.per_alpha))

        self._pos = (self._pos + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

        self._add_count += 1

        # Periodic cleanup
        if self._add_count % self._cleanup_interval == 0:
            gc.collect()
    
    def add_episode(self, episode: List[Experience]):
        for exp in episode:
            self.add(exp)
    
    def sample(self, batch_size: int, beta: float = None):
        """
        If PER is enabled, returns (experiences, indices, is_weights).
        Otherwise returns a list of experiences (legacy behavior).
        """
        if self._size < self.min_size:
            return [] if not self.use_per else ([], [], None)

        if not self.use_per:
            k = min(batch_size, self._size)
            # sample without replacement
            indices = random.sample(range(self._size), k)
            experiences = [self._data[i] for i in indices]
            return experiences

        # PER sampling
        beta = self.per_beta_start if beta is None else float(beta)
        # Anneal beta linearly based on add_count
        beta = min(1.0, beta + (1.0 - beta) * (self._add_count / max(1, self.per_beta_frames)))

        experiences = []
        indices = []
        is_weights = []

        total = self._tree.total()
        if total <= 0:
            # fallback to uniform sampling
            k = min(batch_size, self._size)
            indices = random.sample(range(self._size), k)
            experiences = [self._data[i] for i in indices]
            is_weights = [1.0] * len(experiences)
            return experiences, indices, torch.tensor(is_weights, dtype=torch.float32)

        segment = total / float(batch_size)
        min_prob = math.inf
        probs = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.random() * (b - a) + a
            idx = self._tree.get(s)
            # if index outside current size, clamp
            if idx >= self._size:
                idx = self._size - 1
            indices.append(idx)
            experiences.append(self._data[idx])
            p = self._tree.tree[self._tree.capacity + idx]
            probs.append(p)
            if p > 0:
                min_prob = min(min_prob, p / total)

        probs = np.array(probs, dtype=np.float32)
        # Convert per-leaf priority back to sampling probability
        sampling_probs = probs / (total + 1e-12)

        # Importance-sampling weights
        is_weights = (self._size * sampling_probs) ** (-beta)
        # Normalize
        is_weights = is_weights / (is_weights.max() + 1e-12)

        return experiences, indices, torch.tensor(is_weights, dtype=torch.float32)
    
    def collate_batch(self, experiences: List[Experience]) -> Tuple[Batch, List[List[np.ndarray]], torch.Tensor]:
        """
        Convert experiences to batched format.
        
        v2.0: Creates a single batched graph for efficient processing.
        
        Returns:
            (batched_graphs, target_policies_per_head, target_values)
        """
        data_list = []
        target_policies_per_head = [[] for _ in range(len(experiences[0].visit_distributions))]
        target_values = []
        
        for exp in experiences:
            graph = exp.graph_state
            
            # Convert to Data object
            edges, n_nodes = graph.to_sparse_features()
            if len(edges) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edges.T, dtype=torch.long)
            
            node_types = torch.zeros(n_nodes, dtype=torch.long)
            node_types[graph.n_input:graph.n_input+graph.n_hidden] = 1
            node_types[graph.n_input+graph.n_hidden:] = 2
            
            data = Data(x=node_types, edge_index=edge_index)
            data_list.append(data)
            
            # Targets
            for head_idx, visits in enumerate(exp.visit_distributions):
                target_policies_per_head[head_idx].append(torch.tensor(visits, dtype=torch.float32))
            
            target_values.append(exp.final_reward)
        
        # Batch graphs
        batched_graphs = Batch.from_data_list(data_list)
        
        # Stack targets
        target_policies_per_head = [torch.stack(head_targets) for head_targets in target_policies_per_head]
        
        # Value head uses sigmoid activation, so targets must be in [0, 1]
        target_values_arr = np.array(target_values, dtype=np.float32)
        target_values_arr = np.clip(target_values_arr, 0.0, 1.0)
        target_values = torch.tensor(target_values_arr, dtype=torch.float32).unsqueeze(1)
        
        return batched_graphs, target_policies_per_head, target_values

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for provided leaf indices (indices correspond to positions in the circular buffer)."""
        if not self.use_per:
            return
        for idx, pr in zip(indices, priorities):
            pr_adj = float(max(1e-6, pr))
            self._tree.update(idx, pr_adj ** self.per_alpha)
            self._max_priority = max(self._max_priority, pr_adj)
    
    def __len__(self):
        return int(self._size)
    
    def is_ready(self):
        return int(self._size) >= self.min_size
    
    def clear(self):
        """Clear all experiences and force garbage collection."""
        self._data = [None] * self.max_size
        self._pos = 0
        self._size = 0
        self._add_count = 0
        if self.use_per:
            self._tree = SumTree(self.max_size)
            self._max_priority = 1.0
        gc.collect()