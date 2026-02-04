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
        
        # Residual block 1 (post-norm)
        self.residual_block1_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.residual_block1_norm = nn.LayerNorm(hidden_dim)
        
        # Residual block 2 with dimension reduction (post-norm)
        self.residual_block2_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        self.residual_block2_proj = nn.Linear(hidden_dim, hidden_dim // 2)  # Projection for residual
        self.residual_block2_norm = nn.LayerNorm(hidden_dim // 2)
        
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
        
        # Residual block 1 (post-norm)
        residual = x
        x = self.residual_block1_fc(x)
        x = self.residual_block1_norm(x + residual)
        x = F.relu(x)
        
        # Residual block 2 (with projection for dimension change)
        residual = self.residual_block2_proj(x)
        x = self.residual_block2_fc(x)
        x = self.residual_block2_norm(x + residual)
        x = F.relu(x)
        
        # Output
        value = self.output_proj(x)
        
        return value