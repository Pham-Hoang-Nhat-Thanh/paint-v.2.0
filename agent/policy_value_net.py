import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from agent.encoder import GraphEncoder
from agent.communication import CommunicationHub
from agent.policy_head import PolicyHead
from agent.value_head import ValueHead
from agent.extractor import HeadExtractor


class PolicyValueNet(nn.Module):
    """
    Complete NAS network following spec EXACTLY.
    
    v2.1 Corrections:
    ✅ Extractors process local subgraph structure (not just pooling)
    ✅ Comprehensive residual connections throughout
    ✅ Dropout, LayerNorm for overfitting prevention
    ✅ Proper architectural depth
    """
    
    def __init__(self, 
                 node_subsets: List[List[int]],
                 n_nodes: int,
                 node_embed_dim: int = 128,
                 head_embed_dim: int = 128,
                 encoder_layers: int = 4,
                 extractor_layers: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            node_subsets: List of node indices for each head
            n_nodes: Total number of nodes
            node_embed_dim: Node embedding dimension
            head_embed_dim: Head embedding dimension
            encoder_layers: Depth of main encoder
            extractor_layers: Depth of per-head extractors
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.node_subsets = node_subsets
        self.n_nodes = n_nodes
        self.num_heads = len(node_subsets)
        self.node_embed_dim = node_embed_dim
        self.head_embed_dim = head_embed_dim
        
        # Main encoder (processes full graph)
        self.encoder = GraphEncoder(
            hidden_dim=node_embed_dim,
            num_layers=encoder_layers,
            dropout=dropout
        )
        
        # Per-head extractors (process local subgraphs)
        self.extractors = nn.ModuleList([
            HeadExtractor(
                node_embed_dim=node_embed_dim,
                head_embed_dim=head_embed_dim,
                num_layers=extractor_layers,
                dropout=dropout
            ) for _ in range(self.num_heads)
        ])
        
        # Communication hub
        self.communication_hub = CommunicationHub(
            head_embed_dim=head_embed_dim,
            num_heads=self.num_heads,
            dropout=dropout
        )
        
        # Policy heads
        self.policy_heads = nn.ModuleList([
            PolicyHead(
                head_embed_dim=head_embed_dim,
                num_actions=len(subset) * (len(subset) - 1),
                dropout=dropout
            ) for subset in node_subsets
        ])
        
        # Value head
        self.value_head = ValueHead(
            node_embed_dim=node_embed_dim,
            dropout=dropout
        )
    
    def forward(self, batched_data: Batch, 
                head_id: Optional[int] = None) -> Tuple[Optional[List[torch.Tensor]], torch.Tensor]:
        """
        Single forward pass following spec architecture.
        
        Flow:
        1. Encoder: Full graph → node embeddings
        2. Extractors: Process local subgraph per head → head embeddings
        3. Hub: Inter-head communication → global contexts
        4. Policy heads: Local + global → action logits
        5. Value head: Node embeddings → value
        """
        # Ensure data is on same device as model
        device = next(self.parameters()).device
        if hasattr(batched_data, 'to'):
            try:
                batched_data = batched_data.to(device)
            except Exception:
                pass

        # Defensive moves
        if hasattr(batched_data, 'x'):
            try:
                batched_data.x = batched_data.x.to(device)
            except Exception:
                pass
        if hasattr(batched_data, 'edge_index'):
            try:
                batched_data.edge_index = batched_data.edge_index.to(device)
            except Exception:
                pass

        # 1. Encode full graph
        node_embeddings = self.encoder(batched_data.x, batched_data.edge_index)
        
        # 2. Extract local subgraph representations per head
        head_embeddings = []
        for head_idx, (extractor, subset) in enumerate(zip(self.extractors, self.node_subsets)):
            subset_tensor = torch.tensor(subset, dtype=torch.long, device=node_embeddings.device)
            
            # Process local subgraph structure
            head_emb = extractor(
                node_embeddings,
                subset_tensor,
                batched_data.edge_index,
                batched_data.batch
            )
            head_embeddings.append(head_emb)
        
        # 3. Inter-head communication
        global_contexts = self.communication_hub(head_embeddings)
        
        # 4. Policy predictions
        if head_id is not None:
            # Single head (MCTS evaluation)
            policy_logits = [self.policy_heads[head_id](
                head_embeddings[head_id],
                global_contexts[head_id]
            )]
        else:
            # All heads (training)
            policy_logits = [
                self.policy_heads[i](head_embeddings[i], global_contexts[i])
                for i in range(self.num_heads)
            ]
        
        # 5. Value prediction
        values = self.value_head(node_embeddings, batched_data.batch)
        
        return policy_logits, values
    
    def predict_batch(self, graphs: List, head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        """Batch prediction for MCTS."""
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            data_list = []
            for graph in graphs:
                # Support passing a zero-copy adjacency view from the MCTS engine
                if isinstance(graph, dict) and 'adj' in graph:
                    adj = graph['adj']
                    n_nodes = int(graph['n_nodes'])
                    n_input = int(graph['n_input'])
                    n_hidden = int(graph['n_hidden'])

                    # Use numpy nonzero (C-level) to get edge indices quickly
                    rows, cols = np.nonzero(adj)
                    if rows.size == 0:
                        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    else:
                        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long, device=device)

                    node_types = torch.zeros(n_nodes, dtype=torch.long, device=device)
                    node_types[n_input:n_input+n_hidden] = 1
                    node_types[n_input+n_hidden:] = 2

                    data = Data(x=node_types, edge_index=edge_index)
                    data_list.append(data)
                else:
                    edges, n_nodes = graph.to_sparse_features()

                    if len(edges) == 0:
                        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    else:
                        edge_index = torch.tensor(edges.T, dtype=torch.long, device=device)

                    node_types = torch.zeros(n_nodes, dtype=torch.long, device=device)
                    node_types[graph.n_input:graph.n_input+graph.n_hidden] = 1
                    node_types[graph.n_input+graph.n_hidden:] = 2

                    data = Data(x=node_types, edge_index=edge_index)
                    data_list.append(data)
            
            # Group by head_id
            head_to_indices = {}
            for idx, head_id in enumerate(head_ids):
                if head_id not in head_to_indices:
                    head_to_indices[head_id] = []
                head_to_indices[head_id].append(idx)
            
            all_policies = [None] * len(graphs)
            all_values = [None] * len(graphs)
            
            for head_id, indices in head_to_indices.items():
                sub_data_list = [data_list[i] for i in indices]
                sub_batched = Batch.from_data_list(sub_data_list)
                # Ensure tensors are on the model device (redundant but safe)
                try:
                    sub_batched = sub_batched.to(device)
                except Exception:
                    if hasattr(sub_batched, 'x'):
                        sub_batched.x = sub_batched.x.to(device)
                    if hasattr(sub_batched, 'edge_index'):
                        sub_batched.edge_index = sub_batched.edge_index.to(device)
                
                policy_logits, values = self.forward(sub_batched, head_id=head_id)
                
                for i, idx in enumerate(indices):
                    all_policies[idx] = policy_logits[0][i].cpu().numpy()
                    all_values[idx] = values[i].item()
            
            return all_policies, all_values
