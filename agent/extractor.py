import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.utils import subgraph
from typing import Optional, List


class HeadExtractor(nn.Module):
    """
    Extractor that processes the LOCAL SUBGRAPH structure for a head.
    
    Optimized v3.0:
    - Fast single-graph path without subgraph() call
    - Precomputed edge masks for subset extraction
    - Minimal tensor allocation in hot path
    """
    
    def __init__(self, 
                 node_embed_dim: int,
                 head_embed_dim: int,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        
        self.node_embed_dim = node_embed_dim
        self.head_embed_dim = head_embed_dim
        self.num_layers = num_layers
        
        # Local subgraph processing with GAT
        self.local_gat_layers = nn.ModuleList()
        self.local_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.local_gat_layers.append(
                GATv2Conv(
                    in_channels=node_embed_dim,
                    out_channels=node_embed_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    add_self_loops=True
                )
            )
            self.local_norms.append(nn.LayerNorm(node_embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Aggregation to create head embedding
        self.aggregator = nn.Sequential(
            nn.Linear(node_embed_dim * 3, head_embed_dim * 2),
            nn.LayerNorm(head_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_embed_dim * 2, head_embed_dim),
            nn.LayerNorm(head_embed_dim),
            nn.ReLU()
        )
        
        # Cache for precomputed subset edge indices (single graph case)
        self._cached_subset_edges: Optional[torch.Tensor] = None
        self._cached_subset_indices: Optional[torch.Tensor] = None
    
    def precompute_subset_edges(self, subset_indices: torch.Tensor, 
                                 edge_index: torch.Tensor, 
                                 num_nodes: int) -> torch.Tensor:
        """
        Precompute and cache the subset edge index for a fixed graph structure.
        Call this once when graph structure is known to avoid runtime subgraph extraction.
        """
        device = edge_index.device
        subset_set = set(subset_indices.cpu().tolist())
        
        # Create node mapping: original -> local index
        node_map = {orig: local for local, orig in enumerate(subset_indices.cpu().tolist())}
        
        # Filter edges where both endpoints are in subset
        src, dst = edge_index[0], edge_index[1]

        src_cpu = src.cpu().tolist()
        dst_cpu = dst.cpu().tolist()
        new_edges = []
        for i, (s, d) in enumerate(zip(src_cpu, dst_cpu)):
            if s in subset_set and d in subset_set:
                new_edges.append([node_map[s], node_map[d]])
        
        if new_edges:
            self._cached_subset_edges = torch.tensor(new_edges, dtype=torch.long, device=device).T
        else:
            self._cached_subset_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        self._cached_subset_indices = subset_indices.clone()
        
        return self._cached_subset_edges
    
    def forward_single_fast(self, 
                            node_embeddings: torch.Tensor,
                            subset_indices: torch.Tensor,
                            edge_index: torch.Tensor) -> torch.Tensor:
        """
        Ultra-fast path for single graph inference.
        Avoids subgraph() overhead by using precomputed or inline edge filtering.
        
        Returns:
            Head embedding [1, head_embed_dim]
        """
        device = node_embeddings.device
        
        # Get subset node embeddings directly
        subset_embeddings = node_embeddings[subset_indices]  # [subset_size, dim]
        
        # Fast inline edge extraction (avoids subgraph() overhead)
        # Check if we have cached edges for this subset
        if (self._cached_subset_edges is not None and 
            self._cached_subset_indices is not None and
            self._cached_subset_indices.device == device and
            torch.equal(self._cached_subset_indices, subset_indices)):
            subset_edge_index = self._cached_subset_edges
        else:
            # Inline fast edge filtering using tensor operations
            subset_size = subset_indices.size(0)
            
            # Create a mapping tensor: node_id -> local_idx (or -1 if not in subset)
            max_node = max(node_embeddings.size(0), subset_indices.max().item() + 1)
            node_to_local = torch.full((max_node,), -1, dtype=torch.long, device=device)
            node_to_local[subset_indices] = torch.arange(subset_size, device=device)
            
            # Map edges
            src_local = node_to_local[edge_index[0]]
            dst_local = node_to_local[edge_index[1]]
            
            # Keep only edges where both endpoints are in subset
            valid_mask = (src_local >= 0) & (dst_local >= 0)
            subset_edge_index = torch.stack([src_local[valid_mask], dst_local[valid_mask]], dim=0)
        
        # Process with GAT layers
        x = subset_embeddings
        for gat, norm in zip(self.local_gat_layers, self.local_norms):
            x_residual = x
            x = gat(x, subset_edge_index)
            x = self.dropout(x)
            x = x + x_residual
            x = norm(x)
            x = F.relu(x)
        
        # Aggregate (no scatter needed for single graph)
        mean_emb = x.mean(dim=0, keepdim=True)
        max_emb = x.max(dim=0, keepdim=True)[0]
        sum_emb = x.sum(dim=0, keepdim=True)
        
        combined = torch.cat([mean_emb, max_emb, sum_emb], dim=-1)
        return self.aggregator(combined)
    
    def forward(self, 
                node_embeddings: torch.Tensor,
                subset_indices: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Process local subgraph structure.
        Automatically uses fast path for single-graph case.
        """
        batch_size = batch.max().item() + 1
        
        # Fast path for single graph (most common in MCTS)
        if batch_size == 1:
            return self.forward_single_fast(node_embeddings, subset_indices, edge_index)
        
        # Batched path for training
        return self._forward_batched(node_embeddings, subset_indices, edge_index, batch)
    
    def _forward_batched(self, 
                         node_embeddings: torch.Tensor,
                         subset_indices: torch.Tensor,
                         edge_index: torch.Tensor,
                         batch: torch.Tensor) -> torch.Tensor:
        """Batched processing for training."""
        batch_size = batch.max().item() + 1
        subset_size = subset_indices.size(0)
        device = node_embeddings.device
        
        # Compute offsets
        nodes_per_graph = torch.bincount(batch, minlength=batch_size)
        graph_offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), 
                                   nodes_per_graph.cumsum(0)[:-1]])
        
        # Compute all local subset indices
        local_subsets = graph_offsets.unsqueeze(1) + subset_indices.unsqueeze(0)
        all_subset_nodes = local_subsets.reshape(-1)
        subset_batch = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, subset_size).reshape(-1)
        
        # Extract combined subgraph
        combined_subset_edge_index, _ = subgraph(
            all_subset_nodes, edge_index,
            relabel_nodes=True, num_nodes=node_embeddings.size(0)
        )
        
        subset_embeddings = node_embeddings[all_subset_nodes]
        
        x = subset_embeddings
        for gat, norm in zip(self.local_gat_layers, self.local_norms):
            x_residual = x
            x = gat(x, combined_subset_edge_index)
            x = self.dropout(x)
            x = x + x_residual
            x = norm(x)
            x = F.relu(x)
        
        mean_emb = global_mean_pool(x, subset_batch)
        max_emb = global_max_pool(x, subset_batch)
        sum_emb = torch.zeros(batch_size, x.size(1), device=device)
        sum_emb.scatter_add_(0, subset_batch.unsqueeze(1).expand(-1, x.size(1)), x)
        
        combined = torch.cat([mean_emb, max_emb, sum_emb], dim=-1)
        return self.aggregator(combined)


class FusedMultiHeadExtractor(nn.Module):
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
                                 node_embeddings: torch.Tensor,
                                 edge_index: torch.Tensor = None) -> torch.Tensor:
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
            head_embs = self.forward_all_heads_fused(node_embeddings, edge_index)
            return head_embs.unsqueeze(0)  # [1, num_heads, head_embed_dim]
        
        # Batched path - fully vectorized
        batch_size = batch.max().item() + 1
        num_nodes_per_graph = node_embeddings.size(0) // batch_size  # Assumes equal size graphs
        
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
        
        # OPTIMIZED: Batched per-head projections using einsum
        # Reshape combined to [batch_size, num_heads, node_embed_dim*3]
        batch_size_actual = combined.shape[0]
        
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

