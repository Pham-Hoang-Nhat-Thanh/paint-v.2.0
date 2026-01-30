import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GraphEncoder(nn.Module):
    """
    Graph encoder with proper residual connections and regularization.
    """
    
    def __init__(self, 
                 node_input_dim: int = 16,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node type embedding
        self.node_embedding = nn.Embedding(3, node_input_dim)
        
        # Initial projection to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            # GAT layer
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    add_self_loops=True
                )
            )
            
            # Normalization and dropout
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
    def forward(self, node_types: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_types: [total_nodes] node type indices
            edge_index: [2, total_edges] edge list
        
        Returns:
            Node embeddings [total_nodes, hidden_dim]
        """
        # Initial embedding
        x = self.node_embedding(node_types)
        x = self.input_proj(x)
        
        # GAT layers with residual connections
        for gat, norm, dropout in zip(self.gat_layers, self.layer_norms, self.dropouts):
            x_residual = x
            
            # GAT convolution
            x = gat(x, edge_index)
            x = dropout(x)
            
            # Residual connection
            x = x + x_residual
            
            # Post-normalization
            x = norm(x)
            x = F.relu(x)
        
        return x
