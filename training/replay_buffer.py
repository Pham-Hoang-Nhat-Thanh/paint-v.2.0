import torch
from torch_geometric.data import Data, Batch
from typing import List, Tuple
import numpy as np
import random
from collections import deque
from dataclasses import dataclass


@dataclass
class Experience:
    """Single experience tuple."""
    graph_state: any  # NASGraph
    visit_distributions: List[np.ndarray]
    final_reward: float
    step: int


class ReplayBuffer:
    """
    Optimized experience replay buffer.
    
    v2.0: Supports efficient batching with custom collate function.
    """
    
    def __init__(self, max_size: int = 100000, min_size: int = 1000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
        self.min_size = min_size
    
    def add(self, experience: Experience):
        self.buffer.append(experience)
    
    def add_episode(self, episode: List[Experience]):
        for exp in episode:
            self.add(exp)
    
    def sample(self, batch_size: int) -> List[Experience]:
        if len(self.buffer) < self.min_size:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
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
        target_values = torch.tensor(target_values, dtype=torch.float32).unsqueeze(1)
        
        return batched_graphs, target_policies_per_head, target_values
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self):
        return len(self.buffer) >= self.min_size