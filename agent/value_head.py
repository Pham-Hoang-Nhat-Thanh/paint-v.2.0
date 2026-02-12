import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn.functional as F


class ValueHead(nn.Module):
    """
    Simplified value head with reduced capacity to prevent overfitting.

    Changes from v1:
    - Reduced hidden dimensions (256 -> 128)
    - Removed one residual block
    - Increased dropout (0.2 -> 0.4)
    - Simpler architecture to prevent memorization
    """

    def __init__(self,
                 node_embed_dim: int,
                 hidden_dim: int = 128,  # Reduced from 256
                 dropout: float = 0.4):   # Increased from 0.2
        super().__init__()

        # Input projection with higher dropout
        self.input_proj = nn.Sequential(
            nn.Linear(node_embed_dim * 2, hidden_dim),  # mean + max
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Single residual block (removed extra block for reduced capacity)
        self.residual_block_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.residual_block_norm = nn.LayerNorm(hidden_dim)

        # Output projection - Sigmoid for [0,1] range matching accuracy
        # Added extra dropout before final layer
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout),  # Extra dropout
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
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

        # Single residual block (post-norm)
        residual = x
        x = self.residual_block_fc(x)
        x = self.residual_block_norm(x + residual)
        x = F.relu(x)

        # Output
        value = self.output_proj(x)

        return value