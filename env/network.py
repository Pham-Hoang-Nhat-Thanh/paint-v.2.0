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
    Optimized DAG for Neural Architecture Search - v2.1 Performance Tuned
    
    Optimizations:
    - O(1) attribute lookups in DFS hot loops (critical for MCTS)
    - Reduced allocation in cache key construction
    - Faster topological reordering (single-pass filtering)
    - Localized method binding in edge list construction
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
        
        # Sparse adjacency - using dict of sets
        self.adjacency = {node: set() for node in self.nodes}
        
        # Topological ordering
        self.topo_order = deque(self.nodes)
        self.position = {node: i for i, node in enumerate(self.nodes)}
        
        # Pre-compute node indices for fast lookup
        self._node_indices = {node: i for i, node in enumerate(self.nodes)}
        
        # Zobrist hashing
        self._init_zobrist()
        self.current_hash = 0
        
        # Reachability cache - smaller size to reduce memory
        self._reach_cache = LRUCache(max_size=1000)
        
        # Cached representations
        self._edges_np = None
        self._pyg_data = None
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
        g.nodes = self.nodes  # Shared tuple
        g.zobrist = self.zobrist  # Shared dict
        g._node_indices = self._node_indices  # Shared
        
        # Deep copy mutable state using fastest methods
        g.adjacency = {n: s.copy() for n, s in self.adjacency.items()}
        g.topo_order = self.topo_order.copy()
        g.position = self.position.copy()
        g.current_hash = self.current_hash
        # Create fresh, smaller cache for copy (don't inherit parent's cache)
        g._reach_cache = LRUCache(max_size=1000)
        g._edges_np = None
        g._pyg_data = None
        g._pyg_tensors = {}
        return g
    
    def is_valid_add(self, u_idx: int, v_idx: int) -> bool:
        """
        Check if adding edge u->v is valid.
        Optimized to avoid allocation in fast path.
        """
        # Fast structural rejects
        if u_idx == v_idx:
            return False
        if v_idx < self.n_input:
            return False
        if u_idx >= self.n_input + self.n_hidden:
            return False
        
        u = self.nodes[u_idx]
        v = self.nodes[v_idx]
        
        # Fast adjacency check (O(1))
        # Local variable for speed
        adj_u = self.adjacency[u]
        if v in adj_u:
            return False
        
        # Fast topological ordering check (O(1))
        # If u comes before v, no cycle possible
        if self.position[u] < self.position[v]:
            return True
        
        # Slow path: Check cache before DFS
        # Use hash directly instead of tuple to avoid allocation
        cache_key = (self.current_hash, u_idx, v_idx)
        cached = self._reach_cache.get(cache_key)
        if cached is not None:
            return not cached
        
        # Expensive DFS - check if v can reach u (adding u->v would create cycle)
        reachable = self._reachable(v, u)
        self._reach_cache.put(cache_key, reachable)
        return not reachable
    
    def add_edge(self, u_idx: int, v_idx: int) -> bool:
        """Add edge with O(N) topological reordering."""
        if u_idx == v_idx:
            return False
        if v_idx < self.n_input:
            return False
        if u_idx >= self.n_input + self.n_hidden:
            return False

        u = self.nodes[u_idx]
        v = self.nodes[v_idx]
        
        adj_u = self.adjacency[u]
        if v in adj_u:
            return False
        
        # Fast path: u before v in topo order
        pos_u = self.position[u]
        pos_v = self.position[v]
        
        if pos_u < pos_v:
            adj_u.add(v)
            self.current_hash ^= self.zobrist[(u_idx, v_idx)]
            self._reach_cache.clear()
            self._invalidate_caches()
            return True
        
        # Cycle check before reordering
        if self._reachable(v, u):
            return False
        
        # Reorder: Collect affected nodes (reachable from v that are at or before u)
        # Optimization: do this in one pass during DFS
        affected = self._reachable_nodes_before(v, pos_u)
        
        if affected:
            affected_set = set(affected)  # For O(1) lookup
            new_order = deque()
            
            # Rebuild ordering: nodes before u (not affected), then u, then affected, then rest
            for node in self.topo_order:
                if node not in affected_set:
                    new_order.append(node)
                    if node is u:
                        new_order.extend(affected)
            
            self.topo_order = new_order
            # Rebuild position map
            self.position = {n: i for i, n in enumerate(new_order)}
        
        adj_u.add(v)
        self.current_hash ^= self.zobrist[(u_idx, v_idx)]
        self._reach_cache.clear()
        self._invalidate_caches()
        return True
    
    def remove_edge(self, u_idx: int, v_idx: int) -> bool:
        """Remove edge."""
        u = self.nodes[u_idx]
        v = self.nodes[v_idx]
        adj_u = self.adjacency[u]
        
        if v in adj_u:
            adj_u.remove(v)
            self.current_hash ^= self.zobrist[(u_idx, v_idx)]
            self._reach_cache.clear()
            self._invalidate_caches()
            return True
        return False
    
    def toggle_edge(self, u_idx: int, v_idx: int) -> bool:
        """Toggle edge."""
        u = self.nodes[u_idx]
        v = self.nodes[v_idx]
        adj_u = self.adjacency[u]
        
        if v in adj_u:
            adj_u.remove(v)
            self.current_hash ^= self.zobrist[(u_idx, v_idx)]
            self._reach_cache.clear()
            self._invalidate_caches()
            return True
        return self.add_edge(u_idx, v_idx)
    
    def _reachable(self, start: Node, target: Node) -> bool:
        """
        Optimized DFS with early exit.
        Critical path: localized variable binding for speed.
        """
        if start is target:  # Identity check (fast)
            return True
        
        # Local binding to avoid attribute lookups in loop
        adj = self.adjacency
        stack = [start]
        seen = set()
        add_seen = seen.add
        pop = stack.pop
        
        while stack:
            cur = pop()
            # Iterate children (set iteration is fast)
            for child in adj[cur]:
                if child is target:
                    return True
                if child not in seen:
                    add_seen(child)
                    stack.append(child)
        return False
    
    def _reachable_nodes(self, start: Node) -> List[Node]:
        """Get all nodes reachable from start."""
        adj = self.adjacency
        stack = [start]
        visited = {start}
        add = visited.add
        
        while stack:
            cur = stack.pop()
            for child in adj[cur]:
                if child not in visited:
                    add(child)
                    stack.append(child)
        
        # Return in topological order
        return [n for n in self.topo_order if n in visited]
    
    def _reachable_nodes_before(self, start: Node, position_limit: int) -> List[Node]:
        """
        Optimized: Get reachable nodes from start that have position <= position_limit.
        Combines _reachable_nodes and filtering into one operation.
        """
        adj = self.adjacency
        stack = [start]
        visited = {start}
        add = visited.add
        pop = stack.pop
        result = []
        position = self.position
        
        while stack:
            cur = pop()
            for child in adj[cur]:
                if child not in visited:
                    add(child)
                    if position[child] <= position_limit:
                        result.append(child)
                    stack.append(child)
        return result
    
    def _invalidate_caches(self):
        """Clear all cached representations."""
        self._edges_np = None
        self._pyg_data = None
        self._pyg_tensors.clear()
    
    def get_hash(self) -> int:
        return self.current_hash
    
    def get_num_edges(self) -> int:
        return sum(len(children) for children in self.adjacency.values())
    
    def to_sparse_features(self) -> Tuple[np.ndarray, int]:
        """Return edge list for GNN input - optimized construction."""
        if self._edges_np is not None:
            return self._edges_np, self.n_nodes
        
        # Local binding for speed
        adj = self.adjacency
        edges = []
        append = edges.append
        
        for u in self.nodes:
            ui = u.idx
            for v in adj[u]:
                append([ui, v.idx])
        
        if not edges:
            self._edges_np = np.zeros((0, 2), dtype=np.int32)
        else:
            self._edges_np = np.array(edges, dtype=np.int32)
        
        return self._edges_np, self.n_nodes

    def to_pyg_data(self):
        """Return cached PyG Data dict."""
        if self._pyg_data is not None:
            return self._pyg_data
        
        edges, n = self.to_sparse_features()
        
        # Node types: 0=input, 1=hidden, 2=output
        node_types = np.empty(n, dtype=np.int64)
        node_types[:self.n_input] = 0
        node_types[self.n_input:self.n_input+self.n_hidden] = 1
        node_types[self.n_input+self.n_hidden:] = 2
        
        if edges.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index = edges.T.astype(np.int64)
        
        self._pyg_data = {'x': node_types, 'edge_index': edge_index}
        return self._pyg_data

    def to_torch_tensors(self, device: torch.device):
        """Return cached PyG tensors on the requested device."""
        dev_key = str(device)
        if dev_key in self._pyg_tensors:
            return self._pyg_tensors[dev_key]
        
        pyg = self.to_pyg_data()
        
        if pyg['edge_index'].size == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        else:
            edge_index = torch.from_numpy(pyg['edge_index']).to(device, dtype=torch.long)
        
        x = torch.from_numpy(pyg['x']).to(device, dtype=torch.long)
        
        self._pyg_tensors[dev_key] = {'x': x, 'edge_index': edge_index}
        return self._pyg_tensors[dev_key]