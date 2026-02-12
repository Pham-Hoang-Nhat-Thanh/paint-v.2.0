import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class MultiHeadExtractor(nn.Module):
    """
    Fused extractor that processes ALL heads in a SINGLE batched GPU operation.
    
    v3.2: Pool + Per-head projections only (no GAT - encoder handles message passing)
    - Subset selection and pooling
    - Per-head MLP projections (preserves head-specific capacity)
    - Maximum speed, clean architecture
    
    v3.3: Optimized projection layer
    - Replaced per-head projection loop with batched LinearBatch operations
    - ~2x faster inference through vectorized matrix multiplication
    """
    
    def __init__(self,
                 node_subsets: List[List[int]],
                 node_embed_dim: int = 128,
                 head_embed_dim: int = 128,
                 num_layers: int = 1,  # Kept for API compatibility, ignored
                 num_attention_heads: int = 4,  # Kept for API compatibility, ignored
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_heads = len(node_subsets)
        self.node_embed_dim = node_embed_dim
        self.head_embed_dim = head_embed_dim
        
        # Store subset sizes
        self.subset_sizes = [len(s) for s in node_subsets]
        
        self.dropout = nn.Dropout(dropout)
        
        # OPTIMIZED: Replace per-head projections with batched operations
        # First projection: [num_heads, node_embed_dim*3, head_embed_dim*2]
        self.proj1_weights = nn.Parameter(
            torch.randn(self.num_heads, node_embed_dim * 3, head_embed_dim * 2) / (node_embed_dim * 3) ** 0.5
        )
        self.proj1_bias = nn.Parameter(torch.zeros(self.num_heads, head_embed_dim * 2))
        self.norm1 = nn.LayerNorm(head_embed_dim * 2)
        
        # Second projection: [num_heads, head_embed_dim*2, head_embed_dim]
        self.proj2_weights = nn.Parameter(
            torch.randn(self.num_heads, head_embed_dim * 2, head_embed_dim) / (head_embed_dim * 2) ** 0.5
        )
        self.proj2_bias = nn.Parameter(torch.zeros(self.num_heads, head_embed_dim))
        self.norm2 = nn.LayerNorm(head_embed_dim)
        
        # Precomputed tensors
        self._subset_tensors: Optional[List[torch.Tensor]] = None
        self._all_subset_nodes: Optional[torch.Tensor] = None
        self._head_batch: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        
        # Store node_subsets for lazy init
        self._node_subsets = node_subsets
    
    def _ensure_precomputed(self, device: torch.device):
        """Lazily initialize precomputed tensors."""
        if self._device == device and self._all_subset_nodes is not None:
            return
        
        # Convert subsets to tensors
        self._subset_tensors = [
            torch.tensor(s, dtype=torch.long, device=device) 
            for s in self._node_subsets
        ]
        
        # Concatenate all subset nodes: [total_subset_nodes]
        self._all_subset_nodes = torch.cat(self._subset_tensors)
        
        # Create head batch assignment: which head does each node belong to
        head_assignments = []
        for head_idx, subset in enumerate(self._node_subsets):
            head_assignments.extend([head_idx] * len(subset))
        self._head_batch = torch.tensor(head_assignments, dtype=torch.long, device=device)
        
        self._device = device
    
    def forward_all_heads_fused(self,
                                 node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Process ALL heads in a single fused operation.
        edge_index is ignored (kept for API compatibility).
        
        Args:
            node_embeddings: [num_nodes, node_embed_dim] - already processed by encoder
            edge_index: ignored
        
        Returns:
            head_embeddings: [num_heads, head_embed_dim]
        """
        device = node_embeddings.device
        
        self._ensure_precomputed(device)
        
        # 1. Gather all subset node embeddings in one operation
        all_subset_embeddings = node_embeddings[self._all_subset_nodes]  # [total_subset_nodes, dim]
        
        # 2. Pool per head using scatter operations (non-inplace for gradient safety)
        expand_idx = self._head_batch.unsqueeze(1).expand(-1, self.node_embed_dim)
        
        # sum pooling
        sum_emb = torch.zeros(self.num_heads, self.node_embed_dim, device=device, dtype=all_subset_embeddings.dtype)
        sum_emb = sum_emb.scatter_add(0, expand_idx, all_subset_embeddings)
        
        # count for mean
        ones = torch.ones_like(self._head_batch, dtype=all_subset_embeddings.dtype)
        counts = torch.zeros(self.num_heads, device=device, dtype=all_subset_embeddings.dtype)
        counts = counts.scatter_add(0, self._head_batch, ones)
        
        # mean pooling
        mean_emb = sum_emb / counts.unsqueeze(1).clamp(min=1)
        
        # max pooling - use functional scatter_reduce
        max_emb = torch.zeros(self.num_heads, self.node_embed_dim, device=device, dtype=all_subset_embeddings.dtype)
        max_emb = torch.scatter_reduce(max_emb, 0, expand_idx, all_subset_embeddings, reduce='amax', include_self=False)
        
        # 3. OPTIMIZED: Batched per-head projection using einsum
        combined = torch.cat([mean_emb, max_emb, sum_emb], dim=-1)  # [num_heads, dim*3]
        
        # First projection layer (batched): [num_heads, dim*3] @ [num_heads, dim*3, dim*2]
        # Use einsum for efficient batched matmul: "ij,ijk->ik"
        proj1_out = torch.einsum('ij,ijk->ik', combined, self.proj1_weights) + self.proj1_bias  # [num_heads, dim*2]
        proj1_out = self.norm1(proj1_out)
        proj1_out = F.relu(proj1_out)
        proj1_out = self.dropout(proj1_out)
        
        # Second projection layer (batched): [num_heads, dim*2] @ [num_heads, dim*2, dim]
        head_embeddings = torch.einsum('ij,ijk->ik', proj1_out, self.proj2_weights) + self.proj2_bias  # [num_heads, dim]
        head_embeddings = self.norm2(head_embeddings)
        head_embeddings = F.relu(head_embeddings)
        
        return head_embeddings
    
    def forward(self,
                node_embeddings: torch.Tensor,
                edge_index: torch.Tensor = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Main forward pass - fully vectorized for both single and batched graphs.
        
        Returns:
            For single graph: [1, num_heads, head_embed_dim]
            For batched: [batch_size, num_heads, head_embed_dim]
        """
        device = node_embeddings.device
        self._ensure_precomputed(device)
        
        if batch is None or batch.max().item() == 0:
            # Single graph - most common case in MCTS
            head_embs = self.forward_all_heads_fused(node_embeddings)
            return head_embs.unsqueeze(0)  # [1, num_heads, head_embed_dim]
        
        # Batched path - fully vectorized
        batch_size = batch.max().item() + 1

        # Compute graph offsets
        nodes_per_graph = torch.bincount(batch, minlength=batch_size)
        graph_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            nodes_per_graph.cumsum(0)[:-1]
        ])
        
        # For each (graph, head) pair, we need the subset node indices
        # all_subset_nodes is [total_subset_nodes] where total = sum of all head subset sizes
        # We need to offset these for each graph
        total_subset_nodes = self._all_subset_nodes.size(0)
        
        # Create [batch_size, total_subset_nodes] indices
        # graph_subset_indices[g, i] = graph_offsets[g] + all_subset_nodes[i]
        graph_subset_indices = graph_offsets.unsqueeze(1) + self._all_subset_nodes.unsqueeze(0)
        # Flatten to [batch_size * total_subset_nodes]
        flat_indices = graph_subset_indices.reshape(-1)
        
        # Gather all embeddings at once
        all_embeddings = node_embeddings[flat_indices]  # [batch_size * total_subset_nodes, dim]
        
        # Create combined batch-head assignment
        # For each node, we need (graph_idx, head_idx)
        # graph_batch: [batch_size * total_subset_nodes] - which graph
        # head_batch: [batch_size * total_subset_nodes] - which head
        graph_batch = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, total_subset_nodes).reshape(-1)
        head_batch_expanded = self._head_batch.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        
        # Combined index: graph_idx * num_heads + head_idx
        combined_idx = graph_batch * self.num_heads + head_batch_expanded
        num_combined = batch_size * self.num_heads
        
        # Scatter pooling across combined (graph, head) pairs
        # Use torch_scatter for gradient-safe operations
        expand_idx = combined_idx.unsqueeze(1).expand(-1, self.node_embed_dim)
        
        # sum pooling (needed for mean calculation)
        sum_emb = torch.zeros(num_combined, self.node_embed_dim, device=device, dtype=all_embeddings.dtype)
        sum_emb = sum_emb.scatter_add(0, expand_idx, all_embeddings)
        
        # count for mean
        ones = torch.ones(combined_idx.size(0), device=device, dtype=all_embeddings.dtype)
        counts = torch.zeros(num_combined, device=device, dtype=all_embeddings.dtype)
        counts = counts.scatter_add(0, combined_idx, ones)
        
        # mean pooling
        mean_emb = sum_emb / counts.unsqueeze(1).clamp(min=1)
        
        # max pooling - use scatter_reduce (non-inplace version)
        # Note: scatter_reduce with reduce='amax' is differentiable in PyTorch 2.0+
        max_emb = torch.zeros(num_combined, self.node_embed_dim, device=device, dtype=all_embeddings.dtype)
        max_emb = torch.scatter_reduce(max_emb, 0, expand_idx, all_embeddings, reduce='amax', include_self=False)
        
        # Reshape to [batch_size, num_heads, dim]
        mean_emb = mean_emb.view(batch_size, self.num_heads, self.node_embed_dim)
        max_emb = max_emb.view(batch_size, self.num_heads, self.node_embed_dim)
        sum_emb = sum_emb.view(batch_size, self.num_heads, self.node_embed_dim)
        
        # Combine: [batch_size, num_heads, dim*3]
        combined = torch.cat([mean_emb, max_emb, sum_emb], dim=-1)
        
        # Expand proj1_weights to match batch size: [batch_size, num_heads, dim*3, dim*2]
        # Process each head's data through its projection
        # combined: [batch_size, num_heads, dim*3]
        # proj1_weights: [num_heads, dim*3, dim*2]
        # Result: [batch_size, num_heads, dim*2]
        
        # Use einsum: "bni,nij->bnj" where b=batch, n=heads, i=input_dim, j=output_dim
        proj1_out = torch.einsum('bni,nij->bnj', combined, self.proj1_weights) + self.proj1_bias.unsqueeze(0)
        proj1_out = self.norm1(proj1_out)
        proj1_out = F.relu(proj1_out)
        proj1_out = self.dropout(proj1_out)
        
        # Second projection: [batch_size, num_heads, dim*2] -> [batch_size, num_heads, dim]
        result = torch.einsum('bni,nij->bnj', proj1_out, self.proj2_weights) + self.proj2_bias.unsqueeze(0)
        result = self.norm2(result)
        result = F.relu(result)
        
        return result

