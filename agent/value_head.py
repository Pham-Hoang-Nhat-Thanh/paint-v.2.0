import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool


class ValueHead(nn.Module):
    """Shared value head."""
    
    def __init__(self, node_embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(node_embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        mean_pool = global_mean_pool(node_embeddings, batch)
        max_pool = global_max_pool(node_embeddings, batch)
        graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)
        return self.mlp(graph_embedding)