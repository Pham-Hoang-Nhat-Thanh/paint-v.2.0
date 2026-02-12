# network_fast.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memset
import numpy as np
cimport numpy as cnp

cdef class CythonNASGraph:
    
    def __init__(self, int n_input, int n_hidden, int n_output):
        cdef int i, n
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_nodes = n_input + n_hidden + n_output
        self._max_nodes = self.n_nodes
        
        from .network import Node
        self.nodes = tuple(Node(i, self._get_type(i)) for i in range(self.n_nodes))
        
        n = self.n_nodes
        adj_np = np.zeros((n, n), dtype=np.uint8)
        types_np = np.empty(n, dtype=np.uint8)
        topo_np = np.arange(n, dtype=np.int32)
        pos_np = np.arange(n, dtype=np.int32)
        
        self._adj_matrix = adj_np
        self._node_types = types_np
        self._topo_order = topo_np
        self._position = pos_np
        
        for i in range(n_input):
            self._node_types[i] = 0
        for i in range(n_input, n_input + n_hidden):
            self._node_types[i] = 1
        for i in range(n_input + n_hidden, n):
            self._node_types[i] = 2
            
        self._hash = 0
        self._adj_cache = {}
        self._adj_dirty = True
        
        import random
        random.seed(42)
        self._zobrist = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    self._zobrist[(i, j)] = random.randint(1, 2**62)
    
    cpdef str _get_type(self, int idx):
        if idx < self.n_input:
            return 'input'
        elif idx < self.n_input + self.n_hidden:
            return 'hidden'
        return 'output'
    
    @property
    def adjacency(self):
        cdef int i, j
        cdef set neighbor_set
        
        if self._adj_dirty:
            self._adj_cache = {}
            for node in self.nodes:
                self._adj_cache[node] = set()
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    if self._adj_matrix[i, j]:
                        self._adj_cache[self.nodes[i]].add(self.nodes[j])
            self._adj_dirty = False
        return self._adj_cache
    
    @property 
    def topo_order(self):
        from collections import deque
        cdef int i
        return deque(self.nodes[i] for i in self._topo_order[:self.n_nodes])
    
    @property
    def position(self):
        cdef int i
        return {self.nodes[i]: self._position[i] for i in range(self.n_nodes)}
    
    @property
    def current_hash(self):
        return self._hash
        
    cpdef int get_hash(self):
        return <int>self._hash
    
    cpdef bint toggle_edge(self, int u_idx, int v_idx) except -1:
        cdef bint exists
        cdef bint valid
        cdef int64 hash_val
        
        exists = self._adj_matrix[u_idx, v_idx]
        
        if exists:
            self._adj_matrix[u_idx, v_idx] = 0
            hash_val = self._zobrist.get((u_idx, v_idx), 0)
            self._hash ^= hash_val
            self._adj_dirty = True
            return True
        else:
            with nogil:
                valid = self._is_valid_add_c(u_idx, v_idx)
            
            if not valid:
                return False
            
            # Add the edge FIRST, then reorder if needed
            self._adj_matrix[u_idx, v_idx] = 1
            hash_val = self._zobrist.get((u_idx, v_idx), 0)
            self._hash ^= hash_val
            self._adj_dirty = True
            
            # Recompute topological order if needed
            # (when parent comes at or after child in current order)
            if self._position[u_idx] >= self._position[v_idx]:
                with nogil:
                    self._reorder_topo(u_idx, v_idx)
            
            return True
    
    cpdef bint is_valid_add(self, int u_idx, int v_idx) except -1:
        cdef bint result
        with nogil:
            result = self._is_valid_add_c(u_idx, v_idx)
        return result
    
    cdef bint _is_valid_add_c(self, int u_idx, int v_idx) nogil:
        if u_idx == v_idx:
            return False
        if v_idx < self.n_input:
            return False
        if u_idx >= self.n_input + self.n_hidden:
            return False
        if self._adj_matrix[u_idx, v_idx]:
            return False
            
        if self._position[u_idx] < self._position[v_idx]:
            return True
            
        return not self._reachable_c(v_idx, u_idx)
    
    cdef bint _reachable_c(self, int start_idx, int target_idx) nogil:
        cdef int* stack
        cdef int* visited
        cdef int stack_ptr, cur, i
        
        stack = <int*>malloc(self.n_nodes * sizeof(int))
        visited = <int*>malloc(self.n_nodes * sizeof(int))
        
        if not stack or not visited:
            if stack: free(stack)
            if visited: free(visited)
            return False
            
        memset(visited, 0, self.n_nodes * sizeof(int))
        
        stack_ptr = 0
        stack[stack_ptr] = start_idx
        stack_ptr += 1
        visited[start_idx] = 1
        
        while stack_ptr > 0:
            stack_ptr -= 1
            cur = stack[stack_ptr]
            
            for i in range(self.n_nodes):
                if self._adj_matrix[cur, i]:
                    if i == target_idx:
                        free(stack)
                        free(visited)
                        return True
                    if not visited[i]:
                        visited[i] = 1
                        stack[stack_ptr] = i
                        stack_ptr += 1
        
        free(stack)
        free(visited)
        return False
    
    cdef void _reorder_topo(self, int u_idx, int v_idx) nogil:
        """
        Recompute full topological order using Kahn's algorithm.
        
        The previous incremental approach had bugs that could create invalid orderings
        when multiple edges exist. This implementation is correct but slower O(V+E).
        
        Note: This assumes the edge u->v has already been added to _adj_matrix.
        """
        cdef int* in_degree
        cdef int* queue
        cdef int* new_order
        cdef int qhead, qtail, cur, i, ptr
        
        in_degree = <int*>malloc(self.n_nodes * sizeof(int))
        queue = <int*>malloc(self.n_nodes * sizeof(int))
        new_order = <int*>malloc(self.n_nodes * sizeof(int))
        
        if not in_degree or not queue or not new_order:
            if in_degree: free(in_degree)
            if queue: free(queue)
            if new_order: free(new_order)
            return
        
        # Initialize in_degree to zero
        memset(in_degree, 0, self.n_nodes * sizeof(int))
        
        # Compute in-degrees for all nodes
        for i in range(self.n_nodes):
            for cur in range(self.n_nodes):
                if self._adj_matrix[cur, i]:
                    in_degree[i] += 1
        
        # Initialize queue with nodes that have in-degree 0
        qhead = 0
        qtail = 0
        for i in range(self.n_nodes):
            if in_degree[i] == 0:
                queue[qtail] = i
                qtail += 1
        
        # Process nodes in topological order using Kahn's algorithm
        ptr = 0
        while qhead < qtail:
            cur = queue[qhead]
            qhead += 1
            new_order[ptr] = cur
            ptr += 1
            
            # Decrease in-degree of neighbors
            for i in range(self.n_nodes):
                if self._adj_matrix[cur, i]:
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue[qtail] = i
                        qtail += 1
        
        # If we didn't process all nodes, there's a cycle (shouldn't happen)
        # In that case, append remaining nodes to maintain all nodes in order
        if ptr < self.n_nodes:
            for i in range(self.n_nodes):
                if in_degree[i] > 0:
                    new_order[ptr] = i
                    ptr += 1
        
        # Update topo_order and position
        for i in range(self.n_nodes):
            self._topo_order[i] = new_order[i]
            self._position[new_order[i]] = i
        
        free(in_degree)
        free(queue)
        free(new_order)
    
    cpdef CythonNASGraph copy(self):
        cdef CythonNASGraph g
        cdef int n, i
        
        g = CythonNASGraph.__new__(CythonNASGraph)
        n = self.n_nodes
        
        g.n_input = self.n_input
        g.n_hidden = self.n_hidden
        g.n_output = self.n_output
        g.n_nodes = n
        g.nodes = self.nodes
        g._max_nodes = n
        g._zobrist = self._zobrist
        g._hash = self._hash
        
        g._adj_matrix = np.copy(self._adj_matrix)
        g._node_types = np.copy(self._node_types)
        g._topo_order = np.copy(self._topo_order)
        g._position = np.copy(self._position)
        
        # Don't copy cache - force rebuild on demand
        g._adj_cache = {}
        g._adj_dirty = True
        
        # Clear source cache to prevent memory buildup
        if len(self._adj_cache) > 100:
            self._adj_cache.clear()
            self._adj_dirty = True
        
        return g
    
    def get_num_edges(self):
        return int(np.sum(self._adj_matrix))
    
    def clear_cache(self):
        """Clear adjacency cache to free memory."""
        self._adj_cache.clear()
        self._adj_dirty = True
    
    def to_sparse_features(self):
        cdef int i, j
        edges = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if self._adj_matrix[i, j]:
                    edges.append([i, j])
        if not edges:
            return np.zeros((0, 2), dtype=np.int32), self.n_nodes
        return np.array(edges, dtype=np.int32), self.n_nodes

    cpdef object to_pyg_data(self):
        cdef int i
        edges, n = self.to_sparse_features()
        node_types = np.empty(n, dtype=np.int64)
        for i in range(n):
            node_types[i] = self._node_types[i]
        if edges.size == 0:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        else:
            edge_index = edges.T.astype(np.int64)
        return {'x': node_types, 'edge_index': edge_index}


def NASGraph(n_input, n_hidden, n_output):
    return CythonNASGraph(n_input, n_hidden, n_output)