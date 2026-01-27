import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from agent.encoder import GraphEncoder
from agent.communication import CommunicationHub
from agent.policy_head import PolicyHead
from agent.value_head import ValueHead


class PolicyValueNet(nn.Module):
    """
    Optimized policy-value network for NAS.
    
    v2.0 Changes:
    - Single forward pass returns ALL policy heads + value
    - Supports batched graph processing
    - Efficient MCTS evaluation interface
    """
    
    def __init__(self, 
                 node_subsets: List[List[int]],
                 n_nodes: int,
                 node_embed_dim: int = 128,
                 head_embed_dim: int = 128,
                 encoder_layers: int = 4):
        super().__init__()
        
        self.node_subsets = node_subsets
        self.n_nodes = n_nodes
        self.num_heads = len(node_subsets)
        self.node_embed_dim = node_embed_dim
        self.head_embed_dim = head_embed_dim
        
        # Encoder
        self.encoder = GraphEncoder(
            hidden_dim=node_embed_dim,
            num_layers=encoder_layers
        )
        
        # Head extractors
        self.extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_embed_dim * 2, head_embed_dim),
                nn.LayerNorm(head_embed_dim),
                nn.ReLU()
            ) for _ in range(self.num_heads)
        ])
        
        # Communication hub
        self.communication_hub = CommunicationHub(
            head_embed_dim=head_embed_dim,
            num_heads=self.num_heads
        )
        
        # Policy heads
        self.policy_heads = nn.ModuleList([
            PolicyHead(
                head_embed_dim=head_embed_dim,
                num_actions=len(subset) * (len(subset) - 1)
            ) for subset in node_subsets
        ])
        
        # Value head
        self.value_head = ValueHead(node_embed_dim=node_embed_dim)
    
    def forward(self, batched_data: Batch, 
                head_id: Optional[int] = None) -> Tuple[Optional[List[torch.Tensor]], torch.Tensor]:
        """
        OPTIMIZED: Single forward pass for all heads.
        
        Args:
            batched_data: PyTorch Geometric Batch object
            head_id: If specified, only return this head's policy (for MCTS)
        
        Returns:
            (policy_logits_list, values) where:
            - policy_logits_list: List of [batch_size, num_actions] or None
            - values: [batch_size, 1]
        """
        # Ensure batched_data tensors are on the same device as model
        device = next(self.parameters()).device
        if hasattr(batched_data, 'to'):
            batched_data = batched_data.to(device)

        # Encode entire batched graph
        node_embeddings = self.encoder(batched_data.x, batched_data.edge_index)
        
        # Extract per-head embeddings for each graph in batch
        batch_size = batched_data.num_graphs
        head_embeddings_per_graph = [[] for _ in range(self.num_heads)]
        
        for graph_idx in range(batch_size):
            # Get node embeddings for this graph
            mask = batched_data.batch == graph_idx
            graph_node_embeddings = node_embeddings[mask]
            
            # Extract for each head
            for head_idx, (extractor, subset) in enumerate(zip(self.extractors, self.node_subsets)):
                # Get embeddings for nodes in this head's subset
                subset_embeddings = graph_node_embeddings[subset]
                
                # Pool
                mean_emb = subset_embeddings.mean(dim=0, keepdim=True)
                max_emb = subset_embeddings.max(dim=0, keepdim=True)[0]
                pooled = torch.cat([mean_emb, max_emb], dim=-1)
                
                # Extract
                head_emb = extractor(pooled)
                head_embeddings_per_graph[head_idx].append(head_emb)
        
        # Stack into batch: [batch_size, head_embed_dim] per head
        head_embeddings = [torch.cat(embs, dim=0) for embs in head_embeddings_per_graph]
        
        # Inter-head communication
        global_contexts = self.communication_hub(head_embeddings)
        
        # Policy logits
        if head_id is not None:
            # Single head (for MCTS evaluation)
            policy_logits = [self.policy_heads[head_id](
                head_embeddings[head_id],
                global_contexts[head_id]
            )]
        else:
            # All heads (for training)
            policy_logits = [
                self.policy_heads[i](head_embeddings[i], global_contexts[i])
                for i in range(self.num_heads)
            ]
        
        # Value
        values = self.value_head(node_embeddings, batched_data.batch)
        
        return policy_logits, values
    
    def predict_batch(self, graphs: List, head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Optimized batch prediction for MCTS evaluator.
        
        v2.0: Convert all graphs once, then batch by head_id.
        """
        self.eval()
        
        with torch.no_grad():
            # Convert ALL graphs to Data objects ONCE (expensive operation)
            data_list = []
            # Use cached torch tensors on the model device when available to avoid repeated tensor creation
            device = next(self.parameters()).device
            for graph in graphs:
                tensors = graph.to_torch_tensors(device)
                data = Data(x=tensors['x'], edge_index=tensors['edge_index'])
                data_list.append(data)
            
            # Group by head_id (cheap operation - just indices)
            head_to_indices = {}
            for idx, head_id in enumerate(head_ids):
                if head_id not in head_to_indices:
                    head_to_indices[head_id] = []
                head_to_indices[head_id].append(idx)
            
            # Initialize results
            all_policies = [None] * len(graphs)
            all_values = [None] * len(graphs)
            
            # Process each unique head_id (cheap rebatching of already-converted Data objects)
            for head_id, indices in head_to_indices.items():
                # Extract sub-list (cheap - just indexing)
                sub_data_list = [data_list[i] for i in indices]
                
                # Batch this subset (relatively cheap - data already converted)
                sub_batched = Batch.from_data_list(sub_data_list)
                
                # Forward pass
                policy_logits, values = self.forward(sub_batched, head_id=head_id)
                
                # Store results
                for i, idx in enumerate(indices):
                    all_policies[idx] = policy_logits[0][i].cpu().numpy()
                    all_values[idx] = values[i].item()
            
            return all_policies, all_values