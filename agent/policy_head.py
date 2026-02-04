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
        
        # Fusion layer for combining local and global context
        self.fusion = nn.Sequential(
            nn.Linear(head_embed_dim * 2, head_embed_dim),
            nn.LayerNorm(head_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(head_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Residual blocks (post-norm is more efficient)
        self.residual_block1_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.residual_block1_norm = nn.LayerNorm(hidden_dim)
        
        self.residual_block2_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.residual_block2_norm = nn.LayerNorm(hidden_dim)
        
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
        # Combine local and global via learned fusion
        x = self.fusion(torch.cat([head_embedding, global_context], dim=-1))
        
        # Input projection
        x = self.input_proj(x)
        
        # Residual block 1 (post-norm is more efficient)
        residual = x
        x = self.residual_block1_fc(x)
        x = self.residual_block1_norm(x + residual)
        x = F.relu(x)
        
        # Residual block 2
        residual = x
        x = self.residual_block2_fc(x)
        x = self.residual_block2_norm(x + residual)
        x = F.relu(x)
        
        # Output
        logits = self.output_proj(x)
        
        return logits