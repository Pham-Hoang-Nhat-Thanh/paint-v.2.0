import torch
import torch.nn as nn
from typing import List


class CommunicationHub(nn.Module):
    """
    Inter-head communication with residual connections and regularization.
    
    Optimized v3.0:
    - Fast tensor operations for single-batch case
    - Reduced list/loop overhead
    """
    
    def __init__(self, 
                 head_embed_dim: int, 
                 num_heads: int, 
                 num_attention_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.head_embed_dim = head_embed_dim
        self.num_heads = num_heads
        
        # Pre-normalization
        self.pre_norm = nn.LayerNorm(head_embed_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=head_embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Simplified post-attention
        self.post_attention = nn.Sequential(
            nn.LayerNorm(head_embed_dim),
            nn.Linear(head_embed_dim, head_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, head_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Returns list of global contexts for API compatibility.
        """
        stacked = torch.stack(head_embeddings, dim=1)  # [batch, num_heads, dim]
        result = self.forward_stacked(stacked)
        return [result[:, i, :] for i in range(self.num_heads)]
    
    def forward_stacked(self, stacked: torch.Tensor) -> torch.Tensor:
        """
        Fast path operating on stacked tensor directly.
        
        Args:
            stacked: [batch, num_heads, dim]
        Returns:
            [batch, num_heads, dim]
        """
        normed = self.pre_norm(stacked)
        
        # Self-attention
        attended, _ = self.attention(normed, normed, normed)
        attended = self.dropout(attended)
        attended = attended + normed
        
        # Post-attention (fused reshape)
        batch_size = stacked.size(0)
        attended_flat = attended.reshape(-1, self.head_embed_dim)
        attended_processed = self.post_attention(attended_flat)
        attended_processed = attended_processed.view(batch_size, self.num_heads, self.head_embed_dim)
        
        return attended_processed + attended
