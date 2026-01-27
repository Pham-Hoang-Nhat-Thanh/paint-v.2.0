import torch
import torch.nn as nn


class PolicyHead(nn.Module):
    """Policy head for a single MCTS head."""
    
    def __init__(self, head_embed_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        
        self.num_actions = num_actions
        
        self.mlp = nn.Sequential(
            nn.Linear(head_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, head_embedding: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        combined = head_embedding + global_context
        return self.mlp(combined)
