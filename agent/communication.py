import torch
import torch.nn as nn
from typing import List


class CommunicationHub(nn.Module):
    """Inter-head communication using multi-head attention."""
    
    def __init__(self, head_embed_dim: int, num_heads: int, num_attention_heads: int = 4):
        super().__init__()
        
        self.head_embed_dim = head_embed_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=head_embed_dim,
            num_heads=num_attention_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_embed_dim * 2, head_embed_dim),
                nn.LayerNorm(head_embed_dim),
                nn.ReLU()
            ) for _ in range(num_heads)
        ])
    
    def forward(self, head_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            head_embeddings: List of [batch_size, head_embed_dim]
        
        Returns:
            List of [batch_size, head_embed_dim] global contexts
        """
        stacked = torch.stack(head_embeddings, dim=1)  # [batch, num_heads, dim]
        attended, _ = self.attention(stacked, stacked, stacked)
        
        global_contexts = []
        for i in range(self.num_heads):
            local = head_embeddings[i]
            global_i = attended[:, i, :]
            combined = torch.cat([local, global_i], dim=-1)
            context = self.output_proj[i](combined)
            global_contexts.append(context)
        
        return global_contexts
