import torch
import torch.nn as nn
from typing import List


class CommunicationHub(nn.Module):
    """
    Inter-head communication with residual connections and regularization.
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
        
        # Post-attention processing with residual
        self.post_attention = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_embed_dim * 2, head_embed_dim * 2),
                nn.LayerNorm(head_embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_embed_dim * 2, head_embed_dim),
                nn.LayerNorm(head_embed_dim)
            ) for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, head_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            head_embeddings: List of [batch_size, head_embed_dim]
        
        Returns:
            List of [batch_size, head_embed_dim] global contexts
        """
        # Pre-normalization
        normed_embeddings = [self.pre_norm(emb) for emb in head_embeddings]
        
        # Stack for attention
        stacked = torch.stack(normed_embeddings, dim=1)  # [batch, num_heads, dim]
        
        # Cross-attention with residual
        attended, _ = self.attention(stacked, stacked, stacked)
        attended = self.dropout(attended)
        attended = attended + stacked  # Residual connection
        
        # Per-head global context
        global_contexts = []
        for i in range(self.num_heads):
            # Combine local and global
            local = head_embeddings[i]  # Original (not normed)
            global_i = attended[:, i, :]
            
            combined = torch.cat([local, global_i], dim=-1)
            
            # Process with residual
            context = self.post_attention[i](combined)
            context = context + local  # Residual connection
            
            global_contexts.append(context)
        
        return global_contexts
