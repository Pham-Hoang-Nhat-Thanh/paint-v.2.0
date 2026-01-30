import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F


class ValueHead(nn.Module):
    """
    Value head with residual connections and regularization.
    """
    
    def __init__(self, 
                 node_embed_dim: int, 
                 hidden_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_embed_dim * 2, hidden_dim),  # mean + max
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks
        self.residual_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.residual_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_embeddings: [total_nodes, dim]
            batch: [total_nodes] batch assignment
        
        Returns:
            Values [batch_size, 1]
        """
        # Pool graph
        mean_pool = global_mean_pool(node_embeddings, batch)
        max_pool = global_max_pool(node_embeddings, batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)
        
        # Input projection
        x = self.input_proj(graph_embedding)
        
        # Residual block 1
        residual = x
        x = self.residual_block1(x)
        x = F.relu(x + residual)
        
        # Residual block 2 (dimension change, no residual)
        x = self.residual_block2(x)
        x = F.relu(x)
        
        # Output
        value = self.output_proj(x)
        
        return value