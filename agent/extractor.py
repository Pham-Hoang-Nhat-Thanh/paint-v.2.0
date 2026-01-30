import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from torch_geometric.nn import GATv2Conv


class HeadExtractor(nn.Module):
    """
    Extractor that processes the LOCAL SUBGRAPH structure for a head.
    
    This is the CORRECT implementation as per spec:
    - Extracts subgraph induced by nodes in S_i
    - Processes local structure and relations
    - Creates head embedding e_i that captures subgraph topology
    """
    
    def __init__(self, 
                 node_embed_dim: int,
                 head_embed_dim: int,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        """
        Args:
            node_embed_dim: Dimension of node embeddings from encoder
            head_embed_dim: Output dimension for head embedding
            num_layers: Number of GAT layers for subgraph processing
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.node_embed_dim = node_embed_dim
        self.head_embed_dim = head_embed_dim
        
        # Local subgraph processing (mini-GNN)
        self.local_gat_layers = nn.ModuleList()
        self.local_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.local_gat_layers.append(
                GATv2Conv(
                    in_channels=node_embed_dim,
                    out_channels=node_embed_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    add_self_loops=True
                )
            )
            self.local_norms.append(nn.LayerNorm(node_embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Aggregation to create head embedding
        self.aggregator = nn.Sequential(
            nn.Linear(node_embed_dim * 3, head_embed_dim * 2),  # mean + max + sum
            nn.LayerNorm(head_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_embed_dim * 2, head_embed_dim),
            nn.LayerNorm(head_embed_dim),
            nn.ReLU()
        )
    
    def forward(self, 
                node_embeddings: torch.Tensor,
                subset_indices: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Process local subgraph structure.
        
        Args:
            node_embeddings: [total_nodes, node_embed_dim]
            subset_indices: [subset_size] indices of nodes in this head's subset
            edge_index: [2, total_edges] full graph edges
            batch: [total_nodes] batch assignment
        
        Returns:
            Head embedding [batch_size, head_embed_dim]
        """
        batch_size = batch.max().item() + 1
        head_embeddings = []
        
        for graph_idx in range(batch_size):
            # Get nodes for this graph
            graph_mask = batch == graph_idx
            graph_node_indices = torch.where(graph_mask)[0]
            
            # Map subset indices to this graph's nodes
            local_subset = graph_node_indices[subset_indices]
            
            # Extract subgraph induced by subset
            subset_edge_index, subset_edge_attr = subgraph(
                local_subset,
                edge_index,
                relabel_nodes=True,
                num_nodes=node_embeddings.size(0)
            )
            
            # Get embeddings for subset nodes
            subset_embeddings = node_embeddings[local_subset]
            
            # Process local subgraph structure with GAT layers
            x = subset_embeddings
            for gat, norm in zip(self.local_gat_layers, self.local_norms):
                x_residual = x
                x = gat(x, subset_edge_index)
                x = self.dropout(x)
                x = x + x_residual  # Residual connection
                x = norm(x)
                x = F.relu(x)
            
            # Aggregate processed subgraph: mean + max + sum
            mean_emb = x.mean(dim=0, keepdim=True)
            max_emb = x.max(dim=0, keepdim=True)[0]
            sum_emb = x.sum(dim=0, keepdim=True)
            
            # Combine aggregations
            combined = torch.cat([mean_emb, max_emb, sum_emb], dim=-1)
            
            # Project to head embedding
            head_emb = self.aggregator(combined)
            head_embeddings.append(head_emb)
        
        return torch.cat(head_embeddings, dim=0)

