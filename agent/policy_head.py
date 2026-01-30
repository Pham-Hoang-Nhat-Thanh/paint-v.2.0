import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyHead(nn.Module):
    """
    Policy head with residual connections and dropout.
    """
    
    def __init__(self, 
                 head_embed_dim: int, 
                 num_actions: int, 
                 hidden_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_actions = num_actions
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(head_embed_dim, hidden_dim),
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, head_embedding: torch.Tensor, global_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head_embedding: [batch_size, head_embed_dim]
            global_context: [batch_size, head_embed_dim]
        
        Returns:
            Logits [batch_size, num_actions]
        """
        # Combine local and global
        x = head_embedding + global_context
        
        # Input projection
        x = self.input_proj(x)
        
        # Residual block 1
        residual = x
        x = self.residual_block1(x)
        x = F.relu(x + residual)
        
        # Residual block 2
        residual = x
        x = self.residual_block2(x)
        x = F.relu(x + residual)
        
        # Output
        logits = self.output_proj(x)
        
        return logits