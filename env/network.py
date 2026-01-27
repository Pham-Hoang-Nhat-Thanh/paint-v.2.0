from collections import deque
from typing import List, Tuple
import numpy as np
from utils.lru_cache import LRUCache
import torch

class Node:
    """Lightweight node for neural network DAG."""
    __slots__ = ('idx', 'node_type')
    
    def __init__(self, idx: int, node_type: str):
        self.idx = idx
        self.node_type = node_type
    
    def __repr__(self):
        return f"N{self.idx}"
    
    def __eq__(self, other):
        return isinstance(other, Node) and self.idx == other.idx
    
    def __hash__(self):
        return self.idx


class NASGraph:
    """
    Optimized DAG for Neural Architecture Search.
    
    Improvements in v2.0:
    - O(N) topological reordering using deque
    - Cleaner reachability caching
    """
    
    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        if n_input <= 0 or n_hidden < 0 or n_output <= 0:
            raise ValueError("Invalid graph dimensions")
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_nodes = n_input + n_hidden + n_output
        
        # Create nodes (immutable)
        self.nodes = tuple(Node(i, self._get_type(i)) for i in range(self.n_nodes))
        
        # Sparse adjacency
        self.adjacency = {node: set() for node in self.nodes}
        
        # Topological ordering (using deque for O(1) operations)
        self.topo_order = deque(self.nodes)
        self.position = {node: i for i, node in enumerate(self.nodes)}
        
        # Zobrist hashing
        self._init_zobrist()
        self.current_hash = 0
        
        # Reachability cache (LRU)
        self._reach_cache = LRUCache(max_size=5000)
        # Cached sparse features and PyG Data representation to avoid repeated Python conversion
        self._edges_np = None
        self._pyg_data = None
        # Cached torch tensors per device (keyed by device string)
        self._pyg_tensors = {}
    
    def _get_type(self, idx: int) -> str:
        if idx < self.n_input:
            return 'input'
        elif idx < self.n_input + self.n_hidden:
            return 'hidden'
        return 'output'
    
    def _init_zobrist(self):
        """Initialize Zobrist hash table."""
        rng = np.random.RandomState(42)
        self.zobrist = {}
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    self.zobrist[(i, j)] = rng.randint(1, 2**62, dtype=np.int64)
    
    def copy(self) -> 'NASGraph':
        """Efficient copy with shared immutable data."""
        g = NASGraph.__new__(NASGraph)
        g.n_input = self.n_input
        g.n_hidden = self.n_hidden
        g.n_output = self.n_output
        g.n_nodes = self.n_nodes
        g.nodes = self.nodes  # Shared
        g.zobrist = self.zobrist  # Shared
        
        # Deep copy mutable state
        g.adjacency = {n: s.copy() for n, s in self.adjacency.items()}
        g.topo_order = self.topo_order.copy()
        g.position = self.position.copy()
        g.current_hash = self.current_hash
        g._reach_cache = LRUCache(max_size=5000)
        g._edges_np = None
        g._pyg_data = None
        g._pyg_tensors = {}
        return g
    
    def is_valid_add(self, u_idx: int, v_idx: int) -> bool:
        """Check if adding edge u->v is valid (O(1) typical case)."""
        if u_idx == v_idx:
            return False

        # Disallow edges into input nodes and out of output nodes
        if v_idx < self.n_input:
            return False
        if u_idx >= self.n_input + self.n_hidden:
            return False
        
        u, v = self.nodes[u_idx], self.nodes[v_idx]
        if v in self.adjacency[u]:
            return False
        
        # Fast path: topological order check
        if self.position[u] < self.position[v]:
            return True
        
        # Check cache
        cache_key = (self.current_hash, u_idx, v_idx)
        cached = self._reach_cache.get(cache_key)
        if cached is not None:
            return not cached
        
        # Slow path: reachability
        reachable = self._reachable(v, u)
        self._reach_cache.put(cache_key, reachable)
        return not reachable
    
    def add_edge(self, u_idx: int, v_idx: int) -> bool:
        """
        Add edge with O(N) topological reordering.
        
        Improvement: Uses deque for O(1) append/remove operations.
        """
        if u_idx == v_idx:
            return False

        # Disallow edges into input nodes and out of output nodes
        if v_idx < self.n_input:
            return False
        if u_idx >= self.n_input + self.n_hidden:
            return False

        u, v = self.nodes[u_idx], self.nodes[v_idx]
        if v in self.adjacency[u]:
            return False
        
        # Fast path
        if self.position[u] < self.position[v]:
            self.adjacency[u].add(v)
            self.current_hash ^= self.zobrist[(u_idx, v_idx)]
            self._reach_cache.clear()
            return True
        
        # Check cycle
        if self._reachable(v, u):
            return False
        
        # Reorder: O(N) using deque
        u_pos = self.position[u]
        affected = [n for n in self._reachable_nodes(v) if self.position[n] <= u_pos]
        
        if affected:
            affected_set = set(affected)
            # Build new order efficiently
            new_order = deque()
            u_found = False
            
            for node in self.topo_order:
                if node not in affected_set:
                    new_order.append(node)
                    if node == u:
                        u_found = True
                        # Insert affected nodes after u
                        new_order.extend(affected)
            
            self.topo_order = new_order
            self.position = {n: i for i, n in enumerate(new_order)}
        
        self.adjacency[u].add(v)
        self.current_hash ^= self.zobrist[(u_idx, v_idx)]
        # Invalidate cached sparse features / pyg data and reachability cache
        self._reach_cache.clear()
        self._edges_np = None
        self._pyg_data = None
        self._pyg_tensors.clear()
        return True
    
    def remove_edge(self, u_idx: int, v_idx: int) -> bool:
        """Remove edge."""
        u, v = self.nodes[u_idx], self.nodes[v_idx]
        if v in self.adjacency[u]:
            self.adjacency[u].remove(v)
            self.current_hash ^= self.zobrist[(u_idx, v_idx)]
            # Invalidate cached sparse features / pyg data and reachability cache
            self._reach_cache.clear()
            self._edges_np = None
            self._pyg_data = None
            self._pyg_tensors.clear()
            return True
        return False
    
    def toggle_edge(self, u_idx: int, v_idx: int) -> bool:
        """Toggle edge."""
        u, v = self.nodes[u_idx], self.nodes[v_idx]
        if v in self.adjacency[u]:
            return self.remove_edge(u_idx, v_idx)
        return self.add_edge(u_idx, v_idx)
    
    def _reachable(self, start: Node, target: Node) -> bool:
        """Check if target is reachable from start."""
        if start == target:
            return True
        visited = set([start])
        stack = [start]
        while stack:
            cur = stack.pop()
            for child in self.adjacency[cur]:
                if child == target:
                    return True
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
        return False
    
    def _reachable_nodes(self, start: Node) -> List[Node]:
        """Get all nodes reachable from start in topological order."""
        visited = set([start])
        stack = [start]
        while stack:
            cur = stack.pop()
            for child in self.adjacency[cur]:
                if child not in visited:
                    visited.add(child)
                    stack.append(child)
        return [n for n in self.topo_order if n in visited]
    
    def get_hash(self) -> int:
        return self.current_hash
    
    def get_num_edges(self) -> int:
        return sum(len(children) for children in self.adjacency.values())
    
    def to_sparse_features(self) -> Tuple[np.ndarray, int]:
        """Return edge list for GNN input."""
        # Return a cached numpy array of edges when available to avoid repeated Python loops
        if self._edges_np is not None:
            return self._edges_np, self.n_nodes

        edges = []
        for u in self.nodes:
            # iterate adjacency set (Python-level) but only once per distinct graph
            for v in self.adjacency[u]:
                edges.append([u.idx, v.idx])

        if not edges:
            self._edges_np = np.zeros((0, 2), dtype=np.int32)
        else:
            self._edges_np = np.array(edges, dtype=np.int32)

        return self._edges_np, self.n_nodes

    def to_pyg_data(self):
        """Return a cached PyG `Data`-like lightweight dict to avoid repeated tensor creation.

        This avoids rebuilding node type arrays and edge index tensors on every evaluator call.
        The returned object is a simple dict with keys `x` (np.ndarray) and `edge_index` (np.ndarray).
        """
        if self._pyg_data is not None:
            return self._pyg_data

        edges, n = self.to_sparse_features()
        # Node types as numpy array (0=input,1=hidden,2=output)
        node_types = np.zeros(n, dtype=np.int64)
        node_types[self.n_input:self.n_input+self.n_hidden] = 1
        node_types[self.n_input+self.n_hidden:] = 2

        # Edge index as shape (2, E)
        if edges.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index = edges.T.astype(np.int64)

        self._pyg_data = {'x': node_types, 'edge_index': edge_index}
        return self._pyg_data

    def to_torch_tensors(self, device: torch.device):
        """Return cached PyG tensors on the requested device.

        Caches tensors per-device to avoid repeated CPU->GPU copies.
        """
        dev_key = str(device)
        if dev_key in self._pyg_tensors:
            return self._pyg_tensors[dev_key]

        pyg = self.to_pyg_data()

        # Create tensors directly on the target device
        if pyg['edge_index'].size == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.tensor(pyg['edge_index'], dtype=torch.long, device=device)

        x = torch.tensor(pyg['x'], dtype=torch.long, device=device)

        self._pyg_tensors[dev_key] = {'x': x, 'edge_index': edge_index}
        return self._pyg_tensors[dev_key]
