import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GraphEncoder(nn.Module):
    """
    Graph encoder using GAT (Graph Attention Networks).
    Supports batched graph processing.
    """
    
    def __init__(self, 
                 node_input_dim: int = 16,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.node_input_dim = node_input_dim
        self.hidden_dim = hidden_dim
        
        # Node type embedding
        self.node_embedding = nn.Embedding(3, node_input_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        in_dim = node_input_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim // num_heads if i < num_layers - 1 else hidden_dim
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=num_heads if i < num_layers - 1 else 1,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )
            in_dim = hidden_dim
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_types: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_types: [total_nodes] node type indices (batched)
            edge_index: [2, total_edges] edge list (batched)
        
        Returns:
            Node embeddings [total_nodes, hidden_dim]
        """
        # Ensure inputs are on the same device as model parameters
        device = self.node_embedding.weight.device
        if node_types.device != device:
            node_types = node_types.to(device)
        if edge_index.device != device:
            edge_index = edge_index.to(device)

        x = self.node_embedding(node_types)
        
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_residual = x
            x = gat(x, edge_index)
            x = self.dropout(x)
            
            if x.shape[-1] == x_residual.shape[-1]:
                x = x + x_residual
            
            x = norm(x)
            x = F.relu(x)
        
        return x

