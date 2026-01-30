# network_fast.pyx
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from libc.stdlib cimport malloc, free
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
            
            if self._position[u_idx] >= self._position[v_idx]:
                with nogil:
                    self._reorder_topo(u_idx, v_idx)
            
            self._adj_matrix[u_idx, v_idx] = 1
            hash_val = self._zobrist.get((u_idx, v_idx), 0)
            self._hash ^= hash_val
            self._adj_dirty = True
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
        cdef int* new_order
        cdef int* in_affected
        cdef int* queue
        cdef int qhead, qtail, cur, i, j, pos_u, ptr
        
        new_order = <int*>malloc(self.n_nodes * sizeof(int))
        in_affected = <int*>malloc(self.n_nodes * sizeof(int))
        queue = <int*>malloc(self.n_nodes * sizeof(int))
        
        if not new_order or not in_affected or not queue:
            if new_order: free(new_order)
            if in_affected: free(in_affected)
            if queue: free(queue)
            return
        
        memset(in_affected, 0, self.n_nodes * sizeof(int))
        
        qhead = 0
        qtail = 0
        pos_u = self._position[u_idx]
        
        queue[qtail] = v_idx
        qtail += 1
        in_affected[v_idx] = 1
        
        while qhead < qtail:
            cur = queue[qhead]
            qhead += 1
            
            for i in range(self.n_nodes):
                if self._adj_matrix[cur, i]:
                    if not in_affected[i]:
                        in_affected[i] = 1
                        queue[qtail] = i
                        qtail += 1
        
        free(queue)
        
        ptr = 0
        
        for i in range(self.n_nodes):
            if self._position[i] < pos_u and not in_affected[i]:
                new_order[ptr] = i
                ptr += 1
        
        for i in range(self.n_nodes):
            j = self._topo_order[i]
            if in_affected[j]:
                new_order[ptr] = j
                ptr += 1
        
        if not in_affected[u_idx]:
            new_order[ptr] = u_idx
            ptr += 1
        
        for i in range(self.n_nodes):
            if self._position[i] > pos_u and not in_affected[i]:
                new_order[ptr] = i
                ptr += 1
        
        for i in range(self.n_nodes):
            self._topo_order[i] = new_order[i]
            self._position[new_order[i]] = i
            
        free(new_order)
        free(in_affected)
    
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
        
        g._adj_cache = {}
        g._adj_dirty = True
        
        return g
    
    def get_num_edges(self):
        cdef int count = 0
        cdef int i, j
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                count += self._adj_matrix[i, j]
        return count
    
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