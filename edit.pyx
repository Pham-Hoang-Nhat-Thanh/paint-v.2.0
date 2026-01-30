# mcts_fast.pyx (Zero-allocation version)
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport sqrt, exp
import numpy as np
cimport numpy as cnp
from env.network_fast cimport CythonNASGraph, uint8, int64

cdef struct CNode:
    double visit_count
    double total_value  
    double prior
    int parent
    int* children       
    int num_children
    int capacity        
    int action_id       
    int virtual_loss     

cdef class CMCTSTree:
    cdef CNode* nodes
    cdef int capacity
    cdef public int size
    cdef public int root_idx
    cdef double c_puct
    cdef int virtual_loss_val
    cdef bint _initialized
    
    def __cinit__(self, int max_nodes=5000000):
        self.capacity = max_nodes
        self.nodes = <CNode*>malloc(max_nodes * sizeof(CNode))
        if not self.nodes:
            raise MemoryError(f"Failed to allocate {max_nodes} nodes")
        memset(<void*>self.nodes, 0, max_nodes * sizeof(CNode))
        self.size = 0
        self.root_idx = -1
        self.c_puct = 1.0
        self.virtual_loss_val = 3
        self._initialized = False
        
    def __dealloc__(self):
        if self.nodes:
            self.clear()
            free(self.nodes)
            
    cpdef void initialize(self, double c_puct=1.0, int virtual_loss=3) except *:
        self.clear()
        self.c_puct = c_puct
        self.virtual_loss_val = virtual_loss
        self.root_idx = self._create_node(-1, -1, 1.0)
        self._initialized = True
        
    cdef int _create_node(self, int parent, int action_id, double prior) noexcept nogil:
        if self.size >= self.capacity:
            return -1
        cdef int idx = self.size
        cdef CNode* node = &self.nodes[idx]
        node.visit_count = 0
        node.total_value = 0.0
        node.prior = prior
        node.parent = parent
        node.children = NULL
        node.num_children = 0
        node.capacity = 0
        node.action_id = action_id
        node.virtual_loss = 0
        self.size += 1
        return idx
        
    cpdef void clear(self) except *:
        cdef int i
        for i in range(self.size):
            if self.nodes[i].children != NULL:
                free(self.nodes[i].children)
        self.size = 0
        self.root_idx = -1
        self._initialized = False
        
    cdef int _select_and_apply_virtual_loss_nogil(self, int node_idx) noexcept nogil:
        cdef CNode* node = &self.nodes[node_idx]
        if node.num_children == 0:
            return -1
            
        cdef double best_score = -1e9
        cdef int best_idx = -1
        cdef double q, u, score
        cdef int i, child_idx
        cdef CNode* child
        cdef double parent_visits = node.visit_count
        cdef double vl_total, n_eff
        
        for i in range(node.num_children):
            child_idx = node.children[i]
            child = &self.nodes[child_idx]
            vl_total = child.visit_count + child.virtual_loss
            n_eff = vl_total if vl_total > 0 else 1.0
            if child.visit_count == 0:
                q = 0.0
            else:
                q = (child.total_value - child.virtual_loss * self.virtual_loss_val) / n_eff
            u = self.c_puct * child.prior * sqrt(parent_visits) / (1.0 + vl_total)
            score = q + u
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_idx == -1:
            return -1
            
        child_idx = node.children[best_idx]
        self.nodes[child_idx].virtual_loss += self.virtual_loss_val
        return child_idx
        
    cdef void _backup_nogil(self, int leaf_idx, double value) noexcept nogil:
        cdef int current = leaf_idx
        cdef CNode* node
        
        while current != -1:
            node = &self.nodes[current]
            node.visit_count += 1
            node.total_value += value
            if node.virtual_loss >= self.virtual_loss_val:
                node.virtual_loss -= self.virtual_loss_val
            else:
                node.virtual_loss = 0
            current = node.parent
            
    cdef int _expand_nogil(self, int node_idx, int* action_ids, double* priors, int n) noexcept nogil:
        cdef CNode* node = &self.nodes[node_idx]
        cdef int i, child_idx
        
        if node.num_children > 0 or n == 0:
            return 0
            
        node.children = <int*>malloc(n * sizeof(int))
        if not node.children:
            return -1
            
        node.capacity = n
        node.num_children = n
        
        for i in range(n):
            child_idx = self._create_node(node_idx, action_ids[i], priors[i])
            if child_idx == -1:
                return -1
            node.children[i] = child_idx
        return 0
            
    cpdef int get_node_visits(self, int node_idx) except -1:
        if node_idx < 0 or node_idx >= self.size:
            return -1
        return <int>self.nodes[node_idx].visit_count
        
    cpdef int get_node_num_children(self, int node_idx) except -1:
        if node_idx < 0 or node_idx >= self.size:
            return -1
        return self.nodes[node_idx].num_children
    
    cpdef double get_node_total_value(self, int node_idx) except? 0.0:
        if node_idx < 0 or node_idx >= self.size:
            raise IndexError("Invalid node index")
        return self.nodes[node_idx].total_value
        
    cpdef int get_node_action_id(self, int node_idx) except -1:
        if node_idx < 0 or node_idx >= self.size:
            return -1
        return self.nodes[node_idx].action_id
        
    cpdef int get_node_child(self, int node_idx, int child_rank) except -1:
        if node_idx < 0 or node_idx >= self.size:
            raise IndexError("Invalid node index")
        cdef CNode* node = &self.nodes[node_idx]
        if child_rank < 0 or child_rank >= node.num_children:
            raise IndexError("Child rank out of bounds")
        return node.children[child_rank]
        
    cpdef int get_node_virtual_loss(self, int node_idx) except -1:
        if node_idx < 0 or node_idx >= self.size:
            return -1
        return self.nodes[node_idx].virtual_loss
        
    cpdef double get_node_q(self, int node_idx) except? 0.0:
        if node_idx < 0 or node_idx >= self.size:
            return 0.0
        cdef CNode* node = &self.nodes[node_idx]
        cdef double n = node.visit_count + node.virtual_loss
        if n == 0:
            return 0.0
        return (node.total_value - node.virtual_loss) / n
        
    cpdef void get_visit_distribution(self, double[:] out_array, int node_idx=-1) except *:
        if node_idx < 0:
            node_idx = self.root_idx
        if node_idx < 0 or node_idx >= self.size:
            raise IndexError("Invalid node index")
        cdef CNode* node = &self.nodes[node_idx]
        cdef int i, child_idx, action_id
        cdef double total = 0.0
        cdef int out_size = out_array.shape[0]
        
        for i in range(out_size):
            out_array[i] = 0.0
            
        for i in range(node.num_children):
            child_idx = node.children[i]
            action_id = self.nodes[child_idx].action_id
            if 0 <= action_id < out_size:
                out_array[action_id] = self.nodes[child_idx].visit_count
                total += self.nodes[child_idx].visit_count
                
        if total > 0:
            for i in range(out_size):
                out_array[i] /= total
                
    cpdef int root_visits(self) except -1:
        if self.size == 0 or self.root_idx < 0:
            return 0
        return <int>self.nodes[self.root_idx].visit_count
        
    cpdef int root_num_children(self) except -1:
        if self.size == 0 or self.root_idx < 0:
            return 0
        return self.nodes[self.root_idx].num_children


cdef class MCTSEngine:
    cdef public CMCTSTree tree
    cdef int n_actions
    cdef double c_puct
    cdef int virtual_loss
    cdef double dirichlet_alpha
    cdef double dirichlet_epsilon
    
    cdef int* action_u
    cdef int* action_v
    cdef int max_depth
    cdef double[:] _root_noise_cache
    cdef bint _root_noise_computed
    
    def __cinit__(self, int max_nodes, double c_puct, int virtual_loss, 
                  int n_actions, double dirichlet_alpha=0.3, double dirichlet_epsilon=0.25):
        self.tree = CMCTSTree(max_nodes)
        self.tree.initialize(c_puct, virtual_loss)
        self.n_actions = n_actions
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_depth = 100
        self.action_u = NULL
        self.action_v = NULL
        self._root_noise_cache = np.zeros(n_actions, dtype=np.float64)
        self._root_noise_computed = False
    
    def __dealloc__(self):
        if self.action_u:
            free(self.action_u)
        if self.action_v:
            free(self.action_v)
    
    def set_action_space(self, actions):
        cdef int n = len(actions)
        if self.action_u:
            free(self.action_u)
            free(self.action_v)
        self.action_u = <int*>malloc(n * sizeof(int))
        self.action_v = <int*>malloc(n * sizeof(int))
        self.n_actions = n
        
        cdef int i, u, v
        for i, (u, v) in enumerate(actions):
            self.action_u[i] = u
            self.action_v[i] = v
    
    def clear(self):
        self.tree.clear()
        self.tree.initialize(self.c_puct, self.virtual_loss)
        self._root_noise_computed = False
        memset(<void*>&self._root_noise_cache[0], 0, self.n_actions * sizeof(double))
    
    @property
    def size(self):
        return self.tree.size
    
    @property
    def root_visits(self):
        return self.tree.root_visits()
    
    @property
    def root_num_children(self):
        return self.tree.root_num_children()

    cdef inline bint _is_valid_edge_c(self, uint8* adj, int n_nodes, int n_input, int n_hidden, int u, int v) noexcept nogil:
        if u == v:
            return False
        if v < n_input:
            return False
        if u >= n_input + n_hidden:
            return False
        if adj[u * n_nodes + v]:
            return False
        return True
    
    cdef void _softmax_nogil(self, double* x, int n) noexcept nogil:
        cdef int i
        cdef double max_val = x[0]
        cdef double sum_val = 0.0
        
        for i in range(1, n):
            if x[i] > max_val:
                max_val = x[i]
        
        for i in range(n):
            x[i] = exp(x[i] - max_val)
            sum_val += x[i]
        
        if sum_val > 0:
            for i in range(n):
                x[i] /= sum_val

    cdef int _select_leaf_nogil(self, uint8* base_adj, int n_nodes, int n_input, int n_hidden,
                                uint8* out_state, int* path_buffer, int max_depth) noexcept nogil:
        """
        Select leaf using pre-allocated path_buffer (size max_depth).
        Returns leaf index or -1 on error.
        """
        cdef int path_len = 0
        cdef int current = self.tree.root_idx
        cdef int child_idx, action_id, u, v, i
        
        memcpy(out_state, base_adj, n_nodes * n_nodes * sizeof(uint8))
        path_buffer[path_len] = current
        path_len += 1
        
        while True:
            if self.tree.nodes[current].num_children == 0:
                break
                
            child_idx = self.tree._select_and_apply_virtual_loss_nogil(current)
            if child_idx == -1:
                break
            
            current = child_idx
            
            if path_len >= max_depth:
                return -1
                
            path_buffer[path_len] = current
            path_len += 1
            
            action_id = self.tree.nodes[current].action_id
            u = self.action_u[action_id]
            v = self.action_v[action_id]
            if u >= 0 and u < n_nodes and v >= 0 and v < n_nodes:
                out_state[u * n_nodes + v] = 1
        
        return current

    cdef void _process_batch_nogil(self, int* leaf_indices, uint8* batch_states, double* values, 
                                   double* policies, int batch_count, int n_nodes, int n_input, 
                                   int n_hidden, int* valid_buffer, double* prior_buffer) noexcept nogil:
        """
        Process results using pre-allocated buffers (valid_buffer: batch_count*n_actions, prior_buffer: batch_count*n_actions).
        """
        cdef int i, j, leaf_idx, u, v, valid_count, err
        cdef uint8* state_ptr
        cdef double* policy_ptr
        cdef int* my_valid
        cdef double* my_prior
        
        for i in range(batch_count):
            leaf_idx = leaf_indices[i]
            state_ptr = batch_states + i * n_nodes * n_nodes
            policy_ptr = policies + i * self.n_actions
            
            # Use pre-allocated slice for this item
            my_valid = valid_buffer + i * self.n_actions
            my_prior = prior_buffer + i * self.n_actions
            
            # Find valid actions
            valid_count = 0
            for j in range(self.n_actions):
                u = self.action_u[j]
                v = self.action_v[j]
                if self._is_valid_edge_c(state_ptr, n_nodes, n_input, n_hidden, u, v):
                    my_valid[valid_count] = j
                    my_prior[valid_count] = policy_ptr[j]
                    valid_count += 1
            
            if valid_count > 0:
                self._softmax_nogil(my_prior, valid_count)
                err = self.tree._expand_nogil(leaf_idx, my_valid, my_prior, valid_count)
                # If err == -1, we still backup (terminal)
            
            self.tree._backup_nogil(leaf_idx, values[i])

    cpdef void search(self, CythonNASGraph state, object evaluator, int n_sims, int head_id, int batch_size=256):
        """
        Zero-allocation simulation loop.
        """
        cdef int n_nodes = state.n_nodes
        cdef int n_input = state.n_input
        cdef int n_hidden = state.n_hidden
        cdef int n_total = n_nodes * n_nodes
        
        cdef uint8* base_adj = &state._adj_matrix[0, 0]
        
        # Pre-allocate buffers (once per search, not per simulation)
        cdef uint8* batch_states = <uint8*>malloc(batch_size * n_total * sizeof(uint8))
        cdef int* leaf_indices = <int*>malloc(batch_size * sizeof(int))
        cdef double* values_buffer = <double*>malloc(batch_size * sizeof(double))
        cdef double* policies_buffer = <double*>malloc(batch_size * self.n_actions * sizeof(double))
        
        # Pre-allocate work buffers to avoid malloc in loops
        cdef int* path_buffer = <int*>malloc(self.max_depth * sizeof(int))
        cdef int* valid_buffer = <int*>malloc(batch_size * self.n_actions * sizeof(int))
        cdef double* prior_buffer = <double*>malloc(batch_size * self.n_actions * sizeof(double))
        
        if (not batch_states or not leaf_indices or not values_buffer or not policies_buffer or
            not path_buffer or not valid_buffer or not prior_buffer):
            raise MemoryError("Batch buffer allocation failed")
        
        cdef int sim = 0
        cdef int current_batch = 0
        cdef int i, leaf_idx
        cdef uint8* state_ptr
        cdef double* policy_ptr
        
        # Reusable numpy arrays
        cdef cnp.ndarray[cnp.uint8_t, ndim=3] states_array = np.zeros((batch_size, n_nodes, n_nodes), dtype=np.uint8)
        cdef list states_list
        cdef object policy_arr
        
        try:
            while sim < n_sims:
                current_batch = 0
                
                # === PHASE 1: COLLECT (nogil, zero allocation) ===
                with nogil:
                    for i in range(batch_size):
                        if sim >= n_sims:
                            break
                        
                        state_ptr = batch_states + current_batch * n_total
                        leaf_idx = self._select_leaf_nogil(
                            base_adj, n_nodes, n_input, n_hidden, state_ptr, path_buffer, self.max_depth)
                        
                        if leaf_idx == -1:
                            break
                                
                        leaf_indices[current_batch] = leaf_idx
                        current_batch += 1
                        sim += 1
                
                if current_batch == 0:
                    break
                
                # === PHASE 2: EVALUATE (gil) ===
                # Convert C buffers to Python objects for evaluator
                states_view = <cnp.uint8_t[:current_batch, :n_nodes, :n_nodes]>batch_states
                states_array[:current_batch] = np.asarray(states_view)
                
                states_list = []
                for i in range(current_batch):
                    states_list.append({
                        'adj': states_array[i],
                        'n_nodes': n_nodes,
                        'n_input': n_input,
                        'n_hidden': n_hidden,
                        'n_output': state.n_output
                    })
                
                policies, values = evaluator.evaluate(states_list, [head_id] * current_batch)
                
                # Copy results to C buffers
                for i in range(current_batch):
                    values_buffer[i] = <double>values[i]
                    policy_arr = policies[i]
                    policy_ptr = policies_buffer + i * self.n_actions
                    for j in range(self.n_actions):
                        policy_ptr[j] = <double>policy_arr[j]
                
                # === PHASE 3: EXPAND & BACKUP (nogil, zero allocation) ===
                with nogil:
                    self._process_batch_nogil(
                        leaf_indices, batch_states, values_buffer, policies_buffer,
                        current_batch, n_nodes, n_input, n_hidden, valid_buffer, prior_buffer)
                        
        finally:
            free(batch_states)
            free(leaf_indices)
            free(values_buffer)
            free(policies_buffer)
            free(path_buffer)
            free(valid_buffer)
            free(prior_buffer)

    def node_visits(self, idx): 
        return self.tree.get_node_visits(idx)
    def node_virtual_loss(self, idx): 
        return self.tree.get_node_virtual_loss(idx)
    def node_total_value(self, idx): 
        return self.tree.get_node_total_value(idx)
    def node_q(self, idx): 
        return self.tree.get_node_q(idx)
    def node_action_id(self, idx): 
        return self.tree.get_node_action_id(idx)
    def root_child_idx(self, i): 
        return self.tree.get_node_child(self.tree.root_idx, i)

    cpdef void get_visit_distribution(self, double[:] out_array) except *:
        self.tree.get_visit_distribution(out_array, -1)