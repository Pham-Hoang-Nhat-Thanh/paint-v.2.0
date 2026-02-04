import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from torch_geometric.data import Data, Batch
from agent.encoder import GraphEncoder
from agent.communication import CommunicationHub
from agent.policy_head import PolicyHead
from agent.value_head import ValueHead
from agent.extractor import FusedMultiHeadExtractor


class PolicyValueNet(nn.Module):
    """
    Complete NAS network following spec EXACTLY.
    
    v3.0 Optimizations:
    - Fast single-graph inference path
    - Fused extractor operations
    - torch.compile support
    - Minimal tensor allocation in hot path
    """
    
    def __init__(self, 
                 node_subsets: List[List[int]],
                 n_nodes: int,
                 node_embed_dim: int = 128,
                 head_embed_dim: int = 128,
                 encoder_layers: int = 4,
                 extractor_layers: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            node_subsets: List of node indices for each head
            n_nodes: Total number of nodes
            node_embed_dim: Node embedding dimension
            head_embed_dim: Head embedding dimension
            encoder_layers: Depth of main encoder
            extractor_layers: Depth of extractor
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.node_subsets = node_subsets
        self.n_nodes = n_nodes
        self.num_heads = len(node_subsets)
        self.node_embed_dim = node_embed_dim
        self.head_embed_dim = head_embed_dim
        
        # Main encoder
        self.encoder = GraphEncoder(
            hidden_dim=node_embed_dim,
            num_layers=encoder_layers,
            dropout=dropout
        )
        
        # FUSED multi-head extractor (processes ALL heads in single GPU call)
        self.fused_extractor = FusedMultiHeadExtractor(
            node_subsets=node_subsets,
            node_embed_dim=node_embed_dim,
            head_embed_dim=head_embed_dim,
            num_layers=extractor_layers,
            dropout=dropout
        )
        
        # Communication hub
        self.communication_hub = CommunicationHub(
            head_embed_dim=head_embed_dim,
            num_heads=self.num_heads,
            dropout=dropout
        )
        
        # Policy heads
        self.policy_heads = nn.ModuleList([
            PolicyHead(
                head_embed_dim=head_embed_dim,
                num_actions=len(subset) * (len(subset) - 1),
                dropout=dropout
            ) for subset in node_subsets
        ])
        
        # Value head
        self.value_head = ValueHead(
            node_embed_dim=node_embed_dim,
            dropout=dropout
        )
        
        # Cached tensors (preallocated)
        self._subset_tensors: Optional[List[torch.Tensor]] = None
        self._single_batch: Optional[torch.Tensor] = None
        self._device: Optional[torch.device] = None
        
        # Compilation flag
        self._compiled = False
    
    def compile_for_inference(self, force: bool = False):
        """
        Enable torch.compile for faster inference.
        
        Note: torch.compile has known issues with PyTorch Geometric modules.
        This method is currently a no-op for stability. Set force=True to attempt
        compilation anyway (may cause segfaults).
        
        Args:
            force: If True, attempt compilation despite known issues.
        """
        if not force:
            # torch.compile + PyG causes segfaults; skip by default
            import warnings
            warnings.warn(
                "compile_for_inference() skipped: torch.compile has compatibility issues "
                "with PyTorch Geometric. The model is already optimized for single-graph "
                "inference. Use force=True to attempt compilation anyway.",
                UserWarning
            )
            return
        
        if not hasattr(torch, 'compile') or self._compiled:
            return
        
        try:
            # Only compile MLP-based components (avoid PyG modules)
            self.communication_hub = torch.compile(self.communication_hub, mode='reduce-overhead')
            self.value_head = torch.compile(self.value_head, mode='reduce-overhead')
            for i in range(len(self.policy_heads)):
                self.policy_heads[i] = torch.compile(self.policy_heads[i], mode='reduce-overhead')
            self._compiled = True
        except Exception as e:
            import warnings
            warnings.warn(f"torch.compile failed: {e}. Continuing without compilation.", UserWarning)
    
    def _ensure_cache(self, device: torch.device, n_nodes: int):
        """Ensure cached tensors are on correct device."""
        if self._device != device or self._subset_tensors is None:
            self._subset_tensors = [
                torch.tensor(subset, dtype=torch.long, device=device) 
                for subset in self.node_subsets
            ]
            self._single_batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
            self._device = device
        elif self._single_batch is not None and self._single_batch.size(0) != n_nodes:
            self._single_batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
    
    def forward(self, batched_data: Batch, 
                head_id: Optional[int] = None) -> Tuple[Optional[List[torch.Tensor]], torch.Tensor]:
        """
        Optimized forward pass with FUSED head extraction.
        All 39 heads processed in a single GPU operation.
        """
        device = next(self.parameters()).device
        batched_data = batched_data.to(device)
        n_nodes = batched_data.x.size(0)
        
        # Ensure cache is ready
        self._ensure_cache(device, n_nodes)
        
        # Check if single graph (fast path)
        is_single = (not hasattr(batched_data, 'batch') or 
                     batched_data.batch is None or 
                     batched_data.batch.max().item() == 0)
        
        # 1. Encode full graph
        node_embeddings = self.encoder(batched_data.x, batched_data.edge_index)
        
        # 2. FUSED extraction - ALL heads in ONE GPU call
        if is_single:
            batch = self._single_batch[:n_nodes] if self._single_batch.size(0) >= n_nodes else \
                    torch.zeros(n_nodes, dtype=torch.long, device=device)
            # [1, num_heads, head_embed_dim]
            head_embeddings = self.fused_extractor(node_embeddings, batched_data.edge_index, batch=None)
        else:
            batch = batched_data.batch
            # [batch_size, num_heads, head_embed_dim]
            head_embeddings = self.fused_extractor(node_embeddings, batched_data.edge_index, batch=batch)
        
        # 3. Communication hub (stacked tensor path)
        global_contexts = self.communication_hub.forward_stacked(head_embeddings)  # [batch, num_heads, dim]
        
        # 4. Policy and value
        if head_id is not None:
            policy_logits = [self.policy_heads[head_id](
                head_embeddings[:, head_id, :],
                global_contexts[:, head_id, :]
            )]
        else:
            policy_logits = [
                self.policy_heads[i](head_embeddings[:, i, :], global_contexts[:, i, :])
                for i in range(self.num_heads)
            ]
        
        if is_single:
            values = self.value_head(node_embeddings, batch)
        else:
            values = self.value_head(node_embeddings, batched_data.batch)
        
        return policy_logits, values
    
    def predict_batch(self, graphs: List, head_ids: List[int]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Batch prediction for MCTS.
        Uses no_grad for compatibility with training context.
        
        Special case: When called with a single graph, returns ALL head policies
        in one forward pass (efficient for MCTS node expansion).
        
        OPTIMIZED v3.3: Reduced redundant tensor allocations and conversions
        """
        self.eval()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            num_graphs = len(graphs)
            
            # Fast path: single state evaluation - return ALL head policies
            # This is the common case in MCTS (evaluate one state, get all head policies)
            if num_graphs == 1:
                return self._predict_all_heads_single_state(graphs[0], device)
            
            # Fast path: same graph repeated for all heads (legacy interface)
            if num_graphs == self.num_heads and len(set(head_ids)) == self.num_heads:
                first_graph = graphs[0]
                all_same = True
                for g in graphs[1:]:
                    if g is not first_graph:
                        if isinstance(g, dict) and isinstance(first_graph, dict):
                            if g.get('adj') is not first_graph.get('adj'):
                                all_same = False
                                break
                        else:
                            all_same = False
                            break
                
                if all_same:
                    return self._predict_all_heads_single_state(first_graph, device)
            
            # Preallocate result lists
            all_policies = [None] * num_graphs
            all_values = [None] * num_graphs
            
            # OPTIMIZED: Convert graphs to Data objects once, reuse for all head_ids
            # Group by unique graph
            graph_to_indices = {}  # graph_id -> list of (batch_idx, head_id)
            graph_id_to_data = {}  # graph_id -> Data object
            
            for idx, (graph, head_id) in enumerate(zip(graphs, head_ids)):
                graph_id = id(graph)  # Use object identity as key
                if graph_id not in graph_to_indices:
                    graph_to_indices[graph_id] = []
                    # Convert graph once
                    if isinstance(graph, dict) and 'adj' in graph:
                        adj = graph['adj']
                        n_nodes = int(graph['n_nodes'])
                        n_input = int(graph['n_input'])
                        n_hidden = int(graph['n_hidden'])
                        
                        rows, cols = np.nonzero(adj)
                        if rows.size == 0:
                            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                        else:
                            edge_index = torch.as_tensor(
                                np.vstack((rows, cols)), dtype=torch.long, device=device
                            )
                        
                        node_types = torch.zeros(n_nodes, dtype=torch.long, device=device)
                        node_types[n_input:n_input+n_hidden] = 1
                        node_types[n_input+n_hidden:] = 2
                    else:
                        edges, n_nodes = graph.to_sparse_features()
                        
                        if len(edges) == 0:
                            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                        else:
                            edge_index = torch.as_tensor(edges.T, dtype=torch.long, device=device)
                        
                        node_types = torch.zeros(n_nodes, dtype=torch.long, device=device)
                        node_types[graph.n_input:graph.n_input+graph.n_hidden] = 1
                        node_types[graph.n_input+graph.n_hidden:] = 2
                    
                    graph_id_to_data[graph_id] = Data(x=node_types, edge_index=edge_index)
                
                graph_to_indices[graph_id].append((idx, head_id))
            
            # Process each unique graph
            for graph_id, entries in graph_to_indices.items():
                data = graph_id_to_data[graph_id]
                
                # Group by head_id for this graph
                head_to_batch_indices = {}
                for batch_idx, head_id in entries:
                    if head_id not in head_to_batch_indices:
                        head_to_batch_indices[head_id] = []
                    head_to_batch_indices[head_id].append(batch_idx)
                
                # Forward pass once per unique head_id in this graph
                for head_id, batch_indices in head_to_batch_indices.items():
                    policy_logits, values = self.forward(data.to(device), head_id=head_id)
                    
                    for batch_idx in batch_indices:
                        all_policies[batch_idx] = policy_logits[0][0].cpu().numpy()
                        all_values[batch_idx] = values[0].item()
            
            return all_policies, all_values
    
    def _predict_all_heads_single_state(self, graph, device) -> Tuple[List[np.ndarray], List[float]]:
        """Fast path: single state, get all head policies in one forward pass."""
        if isinstance(graph, dict) and 'adj' in graph:
            adj = graph['adj']
            n_nodes = int(graph['n_nodes'])
            n_input = int(graph['n_input'])
            n_hidden = int(graph['n_hidden'])
            
            rows, cols = np.nonzero(adj)
            if rows.size == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            else:
                edge_index = torch.as_tensor(
                    np.vstack((rows, cols)), dtype=torch.long, device=device
                )
            
            node_types = torch.zeros(n_nodes, dtype=torch.long, device=device)
            node_types[n_input:n_input+n_hidden] = 1
            node_types[n_input+n_hidden:] = 2
        else:
            edges, n_nodes = graph.to_sparse_features()
            
            if len(edges) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            else:
                edge_index = torch.as_tensor(edges.T, dtype=torch.long, device=device)
            
            node_types = torch.zeros(n_nodes, dtype=torch.long, device=device)
            node_types[graph.n_input:graph.n_input+graph.n_hidden] = 1
            node_types[graph.n_input+graph.n_hidden:] = 2
        
        data = Data(x=node_types, edge_index=edge_index).to(device)
        
        # Single forward pass with head_id=None gets ALL policies
        policy_logits_list, values = self.forward(data, head_id=None)
        
        # Convert to numpy - policy_logits_list has one tensor per head
        all_policies = [p[0].cpu().numpy() for p in policy_logits_list]
        value = values[0].item()
        all_values = [value] * self.num_heads
        
        return all_policies, all_values
