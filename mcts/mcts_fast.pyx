# mcts_fast.pyx (Synchronized State Evolution - Segfault Fixed)
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from libc.stdlib cimport malloc, free, calloc, realloc
from libc.string cimport memcpy, memset
from libc.math cimport sqrt, exp
import numpy as np
cimport numpy as cnp
from env.network_fast cimport CythonNASGraph, uint8

cdef struct CHeadActionStats:
    double visit_count
    double total_value  
    double prior
    int virtual_loss

cdef struct CNode:
    int parent
    int* children           
    int num_children
    int capacity
    int* incoming_actions   
    int depth
    double visit_count      
    double total_value      
    CHeadActionStats** head_stats

cdef class CMCTSTree:
    cdef CNode* nodes
    cdef int capacity
    cdef int size
    cdef int root_idx
    cdef int n_heads
    cdef int* n_actions_per_head
    cdef double c_puct
    cdef int virtual_loss_val
    cdef bint _initialized
    
    def __cinit__(self, int max_nodes, int n_heads):
        self.capacity = max_nodes
        self.n_heads = n_heads
        self.nodes = NULL
        self.size = 0
        self.root_idx = -1
        self.n_actions_per_head = NULL
        self.c_puct = 1.0
        self.virtual_loss_val = 3
        self._initialized = False
        
        if max_nodes <= 0:
            raise MemoryError("max_nodes must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
            
        self.nodes = <CNode*>calloc(max_nodes, sizeof(CNode))
        if not self.nodes:
            raise MemoryError(f"Failed to allocate {max_nodes} nodes")
        
    def __dealloc__(self):
        cdef int i, h
        if self.nodes:
            for i in range(self.size):
                if self.nodes[i].children:
                    free(self.nodes[i].children)
                if self.nodes[i].incoming_actions:
                    free(self.nodes[i].incoming_actions)
                if self.nodes[i].head_stats:
                    for h in range(self.n_heads):
                        if self.nodes[i].head_stats[h]:
                            free(self.nodes[i].head_stats[h])
                    free(self.nodes[i].head_stats)
            free(self.nodes)
            self.nodes = NULL
        if self.n_actions_per_head:
            free(self.n_actions_per_head)
            self.n_actions_per_head = NULL
    
    cdef int _create_node(self, int parent, int* incoming_actions, int depth) noexcept nogil:
        if self.size >= self.capacity:
            return -1
            
        cdef int idx = self.size
        cdef CNode* node = &self.nodes[idx]
        
        node.parent = parent
        node.depth = depth
        node.children = NULL
        node.num_children = 0
        node.capacity = 0
        node.visit_count = 0
        node.total_value = 0.0
        
        if incoming_actions:
            node.incoming_actions = <int*>malloc(self.n_heads * sizeof(int))
            if not node.incoming_actions:
                return -1
            memcpy(node.incoming_actions, incoming_actions, self.n_heads * sizeof(int))
        else:
            node.incoming_actions = NULL
            
        node.head_stats = <CHeadActionStats**>calloc(self.n_heads, sizeof(CHeadActionStats*))
        if not node.head_stats:
            return -1
            
        cdef int h
        for h in range(self.n_heads):
            if self.n_actions_per_head and self.n_actions_per_head[h] > 0:
                node.head_stats[h] = <CHeadActionStats*>calloc(self.n_actions_per_head[h], sizeof(CHeadActionStats))
                if not node.head_stats[h]:
                    return -1
        
        self.size += 1
        return idx
        
    cpdef void initialize(self, int[:] n_actions_per_head, double c_puct=1.0, int virtual_loss=3) except *:
        if n_actions_per_head.shape[0] != self.n_heads:
            raise ValueError("n_actions_per_head length must match n_heads")
            
        self.clear()
        self.c_puct = c_puct
        self.virtual_loss_val = virtual_loss
        
        cdef int h
        self.n_actions_per_head = <int*>malloc(self.n_heads * sizeof(int))
        if not self.n_actions_per_head:
            raise MemoryError("Failed to allocate n_actions_per_head")
            
        for h in range(self.n_heads):
            self.n_actions_per_head[h] = n_actions_per_head[h]
        
        self.root_idx = self._create_node(-1, NULL, 0)
        if self.root_idx == -1:
            raise MemoryError("Failed to create root node")
        self._initialized = True
        
    cpdef void clear(self) except *:
        cdef int i, h
        for i in range(self.size):
            if self.nodes[i].children:
                free(self.nodes[i].children)
                self.nodes[i].children = NULL
            if self.nodes[i].incoming_actions:
                free(self.nodes[i].incoming_actions)
                self.nodes[i].incoming_actions = NULL
            if self.nodes[i].head_stats:
                for h in range(self.n_heads):
                    if self.nodes[i].head_stats[h]:
                        free(self.nodes[i].head_stats[h])
                        self.nodes[i].head_stats[h] = NULL
                free(self.nodes[i].head_stats)
                self.nodes[i].head_stats = NULL
        self.size = 0
        self.root_idx = -1
        self._initialized = False
        
    cdef int _select_best_action_for_head_nogil(self, int node_idx, int head_id) noexcept nogil:
        if node_idx < 0 or node_idx >= self.size:
            return -1
            
        cdef CNode* node = &self.nodes[node_idx]
        if not node.head_stats or not node.head_stats[head_id]:
            return -1
            
        cdef CHeadActionStats* stats = node.head_stats[head_id]
        cdef int n_actions = self.n_actions_per_head[head_id]
        
        cdef double best_score = -1e9
        cdef int best_action = 0
        cdef double q, u, n_total
        cdef int a
        
        cdef double parent_visits = node.visit_count
        if parent_visits < 1.0:
            parent_visits = 1.0
        
        for a in range(n_actions):
            n_total = stats[a].visit_count + stats[a].virtual_loss
            if n_total == 0.0:
                q = 0.0
            else:
                q = (stats[a].total_value - stats[a].virtual_loss * self.virtual_loss_val) / n_total
            
            u = self.c_puct * stats[a].prior * sqrt(parent_visits) / (1.0 + n_total)
            
            if q + u > best_score:
                best_score = q + u
                best_action = a
        
        stats[best_action].virtual_loss += self.virtual_loss_val
        return best_action
        
    cdef int _find_child_with_actions_nogil(self, int node_idx, int* actions) noexcept nogil:
        if node_idx < 0 or node_idx >= self.size or not actions:
            return -1
            
        cdef CNode* node = &self.nodes[node_idx]
        cdef int i, j, child_idx
        cdef bint match
        
        for i in range(node.num_children):
            child_idx = node.children[i]
            if child_idx < 0 or child_idx >= self.size:
                continue
            match = True
            for j in range(self.n_heads):
                if not self.nodes[child_idx].incoming_actions:
                    match = False
                    break
                if self.nodes[child_idx].incoming_actions[j] != actions[j]:
                    match = False
                    break
            if match:
                return child_idx
        return -1
        
    cdef int _add_child_nogil(self, int parent_idx, int* actions) noexcept nogil:
        if parent_idx < 0 or parent_idx >= self.size or not actions:
            return -1
            
        cdef CNode* parent = &self.nodes[parent_idx]
        cdef int new_cap
        cdef int* new_children
        
        if parent.num_children >= parent.capacity:
            if parent.capacity == 0:
                new_cap = 4
            else:
                new_cap = parent.capacity * 2
            new_children = <int*>realloc(parent.children, new_cap * sizeof(int))
            if not new_children:
                return -1
            parent.children = new_children
            parent.capacity = new_cap
        
        cdef int child_idx = self._create_node(parent_idx, actions, parent.depth + 1)
        if child_idx == -1:
            return -1
            
        parent.children[parent.num_children] = child_idx
        parent.num_children += 1
        return child_idx
        
    cdef void _backup_nogil(self, int leaf_idx, double value, int** path_actions, int path_len) noexcept nogil:
        """Backup value through the path, updating visit counts and values."""
        if leaf_idx < 0 or leaf_idx >= self.size:
            return
        
        # Update leaf node
        self.nodes[leaf_idx].visit_count += 1.0
        self.nodes[leaf_idx].total_value += value
        
        cdef int current = leaf_idx
        cdef CNode* node
        cdef int step, h, action
        
        # Walk up the path from leaf to root
        for step in range(path_len - 1, -1, -1):
            current = self.nodes[current].parent
            if current == -1:
                break
            
            node = &self.nodes[current]
            
            # Update node-level visit count (used in UCB parent_visits)
            node.visit_count += 1.0
            node.total_value += value
            
            # Update per-head action stats
            if node.head_stats and path_actions and path_actions[step]:
                for h in range(self.n_heads):
                    if not node.head_stats[h]:
                        continue
                    action = path_actions[step][h]
                    if action >= 0 and action < self.n_actions_per_head[h]:
                        node.head_stats[h][action].visit_count += 1
                        node.head_stats[h][action].total_value += value
                        if node.head_stats[h][action].virtual_loss >= self.virtual_loss_val:
                            node.head_stats[h][action].virtual_loss -= self.virtual_loss_val


cdef class MCTSEngine:
    cdef CMCTSTree tree
    cdef int n_heads
    cdef int** action_u
    cdef int** action_v
    cdef int* n_actions
    cdef double c_puct
    cdef int virtual_loss
    cdef double dirichlet_alpha
    cdef double dirichlet_epsilon
    cdef int max_depth
    cdef bint _initialized
    
    def __cinit__(self, int max_nodes, int n_heads, double c_puct=1.0, 
                  int virtual_loss=3, double dirichlet_alpha=0.3, 
                  double dirichlet_epsilon=0.25):
        if max_nodes <= 0 or n_heads <= 0:
            raise ValueError("max_nodes and n_heads must be positive")
            
        self.n_heads = n_heads
        self.tree = CMCTSTree(max_nodes, n_heads)
        self.c_puct = c_puct
        self.virtual_loss = virtual_loss
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_depth = 1000
        self.action_u = NULL
        self.action_v = NULL
        self.n_actions = NULL
        self._initialized = False
    
    def __dealloc__(self):
        cdef int h
        if self.action_u:
            for h in range(self.n_heads):
                if self.action_u[h]:
                    free(self.action_u[h])
            free(self.action_u)
        if self.action_v:
            for h in range(self.n_heads):
                if self.action_v[h]:
                    free(self.action_v[h])
            free(self.action_v)
        if self.n_actions:
            free(self.n_actions)
    
    def initialize_tree(self, list n_actions_per_head):
        if len(n_actions_per_head) != self.n_heads:
            raise ValueError("Length mismatch")
            
        cdef int h
        cdef int[:] arr = np.array(n_actions_per_head, dtype=np.int32)
        self.tree.initialize(arr, self.c_puct, self.virtual_loss)
        
        self.n_actions = <int*>malloc(self.n_heads * sizeof(int))
        if not self.n_actions:
            raise MemoryError("Failed to allocate n_actions")
            
        for h in range(self.n_heads):
            self.n_actions[h] = n_actions_per_head[h]
        self._initialized = True
    
    def set_head_action_spaces(self, list action_spaces):
        if len(action_spaces) != self.n_heads:
            raise ValueError("action_spaces length must match n_heads")
            
        cdef int h, i, u, v, n
        cdef list actions
        
        self.action_u = <int**>calloc(self.n_heads, sizeof(int*))
        self.action_v = <int**>calloc(self.n_heads, sizeof(int*))
        
        if not self.action_u or not self.action_v:
            raise MemoryError("Failed to allocate action space arrays")
        
        try:
            for h in range(self.n_heads):
                actions = action_spaces[h]
                n = len(actions)
                
                self.action_u[h] = <int*>malloc(n * sizeof(int))
                self.action_v[h] = <int*>malloc(n * sizeof(int))
                
                if not self.action_u[h] or not self.action_v[h]:
                    raise MemoryError(f"Failed to allocate action space for head {h}")
                
                for i, (u, v) in enumerate(actions):
                    self.action_u[h][i] = u
                    self.action_v[h][i] = v
        except Exception:
            self._free_action_spaces()
            raise
    
    cdef void _free_action_spaces(self) noexcept:
        cdef int h
        if self.action_u:
            for h in range(self.n_heads):
                if self.action_u[h]:
                    free(self.action_u[h])
                    self.action_u[h] = NULL
            free(self.action_u)
            self.action_u = NULL
        if self.action_v:
            for h in range(self.n_heads):
                if self.action_v[h]:
                    free(self.action_v[h])
                    self.action_v[h] = NULL
            free(self.action_v)
            self.action_v = NULL
    
    cdef inline bint _is_valid_edge_c(self, uint8* adj, int n, int u, int v, int n_input, int n_hidden) noexcept nogil:
        if u == v:
            return False
        if v < n_input:
            return False
        if u >= n_input + n_hidden:
            return False
        if adj[u*n + v]:
            return False
        return True
    
    cdef void _apply_actions_to_state_nogil(self, uint8* base_adj, uint8* out_adj, int n_nodes,
                                           int* selected_actions, int n_input, int n_hidden) noexcept nogil:
        if not base_adj or not out_adj or not selected_actions:
            return
        memcpy(out_adj, base_adj, n_nodes * n_nodes * sizeof(uint8))
        cdef int h, a, u, v
        for h in range(self.n_heads):
            a = selected_actions[h]
            if a >= 0 and a < self.n_actions[h]:
                u = self.action_u[h][a]
                v = self.action_v[h][a]
                if u >= 0 and u < n_nodes and v >= 0 and v < n_nodes:
                    if self._is_valid_edge_c(out_adj, n_nodes, u, v, n_input, n_hidden):
                        out_adj[u * n_nodes + v] = 1
    
    cdef void _softmax_nogil(self, double* x, int n) noexcept nogil:
        if not x or n <= 0:
            return
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
    
    cdef void _compute_valid_mask_and_mask_policy_nogil(self, int head_id, uint8* adj, 
                                                         int n_nodes, int n_input, int n_hidden,
                                                         double* policy, double* valid_mask) noexcept nogil:
        """Compute validity mask and mask invalid actions in policy (set to -1e9)."""
        cdef int a, u, v, n_act
        n_act = self.n_actions[head_id]
        for a in range(n_act):
            u = self.action_u[head_id][a]
            v = self.action_v[head_id][a]
            if self._is_valid_edge_c(adj, n_nodes, u, v, n_input, n_hidden):
                valid_mask[a] = 1.0
            else:
                valid_mask[a] = 0.0
                policy[a] = -1e9
    
    cdef void _apply_softmax_to_policy_nogil(self, double* policy, int n) noexcept nogil:
        """Apply softmax to policy array in-place."""
        cdef int i
        cdef double max_val, sum_val
        if n <= 0:
            return
        max_val = policy[0]
        for i in range(1, n):
            if policy[i] > max_val:
                max_val = policy[i]
        sum_val = 0.0
        for i in range(n):
            policy[i] = exp(policy[i] - max_val)
            sum_val += policy[i]
        if sum_val > 1e-8:
            for i in range(n):
                policy[i] /= sum_val

    cdef void _reconstruct_state_along_path_nogil(self, uint8* base_adj, uint8* out_adj, 
                                                    int n_nodes, int n_input, int n_hidden,
                                                    int** path_actions, int path_len) noexcept nogil:
        """Reconstruct state by applying all actions along the path from root."""
        memcpy(out_adj, base_adj, n_nodes * n_nodes * sizeof(uint8))
        cdef int step, h, a, u, v
        for step in range(path_len):
            if not path_actions[step]:
                continue
            for h in range(self.n_heads):
                a = path_actions[step][h]
                if a >= 0 and a < self.n_actions[h]:
                    u = self.action_u[h][a]
                    v = self.action_v[h][a]
                    if u >= 0 and u < n_nodes and v >= 0 and v < n_nodes:
                        if self._is_valid_edge_c(out_adj, n_nodes, u, v, n_input, n_hidden):
                            out_adj[u * n_nodes + v] = 1

    cpdef void search(self, CythonNASGraph state, object evaluator, int n_sims) except *:
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize_tree() first.")
        if not state:
            raise ValueError("state is None")
        if not evaluator:
            raise ValueError("evaluator is None")
        if n_sims <= 0:
            return
            
        cdef int n_nodes = state.n_nodes
        cdef int n_input = state.n_input
        cdef int n_hidden = state.n_hidden
        
        cdef uint8* base_adj = &state._adj_matrix[0, 0]
        cdef uint8* temp_adj = <uint8*>malloc(n_nodes * n_nodes * sizeof(uint8))
        if not temp_adj:
            raise MemoryError("Failed to allocate temp_adj")
        
        cdef int* selected_actions = <int*>malloc(self.n_heads * sizeof(int))
        cdef int** path_actions = <int**>malloc(self.max_depth * sizeof(int*))
        cdef int* path_nodes = <int*>malloc(self.max_depth * sizeof(int))
        
        if not selected_actions or not path_actions or not path_nodes:
            free(temp_adj)
            raise MemoryError("Failed to allocate path buffers")
        
        cdef int sim, step, h, current_node, child_node, i, a, n_act
        cdef bint is_new_node
        cdef double value
        cdef CNode* leaf_ptr
        cdef CNode* root_node
        cdef bint root_needs_priors
        cdef double prior_sum
        cdef int n_valid, noise_idx
        
        # Pre-allocate policy buffers ONCE (reused across all expansions)
        cdef int max_actions = 0
        for h in range(self.n_heads):
            if self.n_actions[h] > max_actions:
                max_actions = self.n_actions[h]
        
        cdef double* policy_buf = <double*>malloc(max_actions * sizeof(double))
        cdef double* valid_buf = <double*>malloc(max_actions * sizeof(double))
        if not policy_buf or not valid_buf:
            if policy_buf: free(policy_buf)
            if valid_buf: free(valid_buf)
            free(temp_adj)
            free(selected_actions)
            free(path_actions)
            free(path_nodes)
            raise MemoryError("Failed to allocate policy buffers")
        
        for i in range(self.max_depth):
            path_actions[i] = <int*>malloc(self.n_heads * sizeof(int))
            if not path_actions[i]:
                # Cleanup
                for j in range(i):
                    free(path_actions[j])
                free(path_actions)
                free(selected_actions)
                free(path_nodes)
                free(temp_adj)
                raise MemoryError("Failed to allocate path_actions[i]")
        
        # Initialize root priors if not done yet (first simulation)
        root_node = &self.tree.nodes[self.tree.root_idx]
        root_needs_priors = True
        if root_node.head_stats and root_node.head_stats[0]:
            # Check if priors are already set (sum > 0)
            prior_sum = 0.0
            for a in range(self.n_actions[0]):
                prior_sum += root_node.head_stats[0][a].prior
            if prior_sum > 0.0:
                root_needs_priors = False
        
        if root_needs_priors:
            # Evaluate root state to get priors for all heads (ONE network call)
            adj_arr = np.asarray(<uint8[:n_nodes*n_nodes]>base_adj).reshape(n_nodes, n_nodes).copy()
            features = {
                'adj': adj_arr,
                'n_nodes': n_nodes,
                'n_input': n_input,
                'n_hidden': n_hidden,
                'n_output': state.n_output
            }
            
            # Single evaluation call - network returns all head policies for this state
            policies, values = evaluator.evaluate([features], [0])
            
            # Set priors for each head with Dirichlet noise at root
            for h in range(self.n_heads):
                n_act = self.n_actions[h]
                
                # policies[h] is already a numpy array from predict_batch
                policy_np = policies[h].ravel()
                
                if len(policy_np) >= n_act:
                    for a in range(n_act):
                        policy_buf[a] = policy_np[a]
                else:
                    for a in range(n_act):
                        policy_buf[a] = 1.0 / n_act
                
                # Compute valid mask and mask policy in pure Cython
                self._compute_valid_mask_and_mask_policy_nogil(h, base_adj, n_nodes, n_input, n_hidden, policy_buf, valid_buf)
                
                # Apply softmax in Cython
                self._apply_softmax_to_policy_nogil(policy_buf, n_act)
                
                # Add Dirichlet noise at root (only to valid actions)
                n_valid = 0
                for a in range(n_act):
                    if valid_buf[a] > 0:
                        n_valid += 1
                
                if n_valid > 0:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * n_valid)
                    noise_idx = 0
                    for a in range(n_act):
                        if valid_buf[a] > 0:
                            policy_buf[a] = (1.0 - self.dirichlet_epsilon) * policy_buf[a] + self.dirichlet_epsilon * noise[noise_idx]
                            noise_idx += 1
                
                # Copy priors to tree
                for a in range(n_act):
                    root_node.head_stats[h][a].prior = policy_buf[a]
        
        try:
            for sim in range(n_sims):
                current_node = self.tree.root_idx
                step = 0
                is_new_node = False
                
                # Selection: traverse tree until we find a node to expand
                while step < self.max_depth:
                    for h in range(self.n_heads):
                        selected_actions[h] = self.tree._select_best_action_for_head_nogil(current_node, h)
                        if selected_actions[h] == -1:
                            selected_actions[h] = 0  # Fallback
                        path_actions[step][h] = selected_actions[h]
                    
                    child_node = self.tree._find_child_with_actions_nogil(current_node, selected_actions)
                    
                    if child_node == -1:
                        # Expansion: create new child node
                        child_node = self.tree._add_child_nogil(current_node, selected_actions)
                        if child_node == -1:
                            break
                        is_new_node = True
                        path_nodes[step] = child_node
                        step += 1
                        break
                    else:
                        # Continue traversing existing tree
                        path_nodes[step] = child_node
                        current_node = child_node
                        step += 1
                
                if step == 0:
                    continue  # Failed to expand
                
                # The leaf is always the last node in path_nodes
                leaf_node = path_nodes[step - 1]
                
                # Only evaluate and set priors for NEW nodes
                if is_new_node:
                    # Reconstruct state at leaf by applying ALL actions along the path
                    self._reconstruct_state_along_path_nogil(
                        base_adj, temp_adj, n_nodes, n_input, n_hidden,
                        path_actions, step
                    )
                    
                    # Single network call - returns policies for all heads + value
                    adj_arr = np.asarray(<uint8[:n_nodes*n_nodes]>temp_adj).reshape(n_nodes, n_nodes).copy()
                    features = {
                        'adj': adj_arr,
                        'n_nodes': n_nodes,
                        'n_input': n_input,
                        'n_hidden': n_hidden,
                        'n_output': state.n_output
                    }
                    
                    # Single evaluation call - network returns all head policies for this state
                    policies, values = evaluator.evaluate([features], [0])
                    value = float(values[0])
                    
                    # Set priors on the newly expanded node (no Dirichlet noise for non-root)
                    leaf_ptr = &self.tree.nodes[leaf_node]
                    for h in range(self.n_heads):
                        n_act = self.n_actions[h]
                        
                        # policies[h] is already a numpy array from predict_batch
                        policy_np = policies[h].ravel()
                        
                        # Copy to buffer (policies are already numpy float arrays)
                        if len(policy_np) >= n_act:
                            for a in range(n_act):
                                policy_buf[a] = policy_np[a]
                        else:
                            # Uniform fallback
                            for a in range(n_act):
                                policy_buf[a] = 1.0 / n_act
                        
                        # Mask invalid actions and compute validity
                        self._compute_valid_mask_and_mask_policy_nogil(h, temp_adj, n_nodes, n_input, n_hidden, policy_buf, valid_buf)
                        
                        # Apply softmax
                        self._apply_softmax_to_policy_nogil(policy_buf, n_act)
                        
                        # Copy priors to tree
                        for a in range(n_act):
                            leaf_ptr.head_stats[h][a].prior = policy_buf[a]
                else:
                    # For existing nodes, use the stored value (or a default)
                    # In standard MCTS, we should not reach here often - 
                    # this only happens if we traverse to a terminal state
                    value = self.tree.nodes[leaf_node].total_value / max(1.0, self.tree.nodes[leaf_node].visit_count)
                
                # Backup value through the path
                self.tree._backup_nogil(leaf_node, value, path_actions, step)
                    
        finally:
            free(temp_adj)
            free(selected_actions)
            free(path_nodes)
            free(policy_buf)
            free(valid_buf)
            for i in range(self.max_depth):
                if path_actions[i]:
                    free(path_actions[i])
            free(path_actions)
    
    def get_visit_distribution_for_head(self, int head_id, double[:] out_array):
        if head_id < 0 or head_id >= self.n_heads:
            raise ValueError("Invalid head_id")
        if not self._initialized:
            raise RuntimeError("Not initialized")
            
        cdef int root = self.tree.root_idx
        if root < 0 or root >= self.tree.size:
            return
            
        if not self.tree.nodes[root].head_stats or not self.tree.nodes[root].head_stats[head_id]:
            return
            
        cdef CHeadActionStats* stats = self.tree.nodes[root].head_stats[head_id]
        cdef int a
        cdef double total = 0.0
        
        for a in range(self.n_actions[head_id]):
            out_array[a] = stats[a].visit_count
            total += stats[a].visit_count
        
        if total > 0:
            for a in range(self.n_actions[head_id]):
                out_array[a] /= total

        # Proxy-compatible methods for Python wrapper access

    def get_node_parent(self, int node_idx):
        """Get parent node index."""
        if node_idx < 0 or node_idx >= self.tree.size:
            return -1
        return self.tree.nodes[node_idx].parent
    
    def get_head_action_visits(self, int node_idx, int head_id, int action_idx):
        """Get visit count for specific head/action at node."""
        if node_idx < 0 or node_idx >= self.tree.size:
            return 0
        cdef CNode* node = &self.tree.nodes[node_idx]
        if not node.head_stats or not node.head_stats[head_id]:
            return 0
        if action_idx < 0 or action_idx >= self.n_actions[head_id]:
            return 0
        return int(node.head_stats[head_id][action_idx].visit_count)
    
    def get_head_action_virtual_loss(self, int node_idx, int head_id, int action_idx):
        """Get virtual loss for specific head/action at node."""
        if node_idx < 0 or node_idx >= self.tree.size:
            return 0
        cdef CNode* node = &self.tree.nodes[node_idx]
        if not node.head_stats or not node.head_stats[head_id]:
            return 0
        if action_idx < 0 or action_idx >= self.n_actions[head_id]:
            return 0
        return node.head_stats[head_id][action_idx].virtual_loss
    
    def get_head_action_total_value(self, int node_idx, int head_id, int action_idx):
        """Get total value for specific head/action at node."""
        if node_idx < 0 or node_idx >= self.tree.size:
            return 0.0
        cdef CNode* node = &self.tree.nodes[node_idx]
        if not node.head_stats or not node.head_stats[head_id]:
            return 0.0
        if action_idx < 0 or action_idx >= self.n_actions[head_id]:
            return 0.0
        return node.head_stats[head_id][action_idx].total_value
    
    @property
    def root_num_children(self):
        if not self._initialized or self.tree.root_idx < 0:
            return 0
        return self.tree.nodes[self.tree.root_idx].num_children
    
    def root_child_idx(self, int i):
        """Get the i-th child index of root."""
        cdef int root = self.tree.root_idx
        if root < 0 or i < 0 or i >= self.tree.nodes[root].num_children:
            return -1
        return self.tree.nodes[root].children[i]
    
    def get_node_child(self, int node_idx, int child_rank):
        """Get child index by rank."""
        if node_idx < 0 or node_idx >= self.tree.size:
            return -1
        cdef CNode* node = &self.tree.nodes[node_idx]
        if child_rank < 0 or child_rank >= node.num_children:
            return -1
        return node.children[child_rank]
    
    def node_action_id(self, int node_idx):
        """Get incoming action for head 0 (or -1 if root)."""
        if node_idx < 0 or node_idx >= self.tree.size:
            return -1
        if self.tree.nodes[node_idx].incoming_actions:
            return self.tree.nodes[node_idx].incoming_actions[0]  # Return first head's action
        return -1
    
    def get_head_action_at_node(self, int node_idx, int head_id):
        """Get the action head_id took to reach node_idx."""
        if node_idx < 0 or node_idx >= self.tree.size:
            return -1
        if not self.tree.nodes[node_idx].incoming_actions:
            return -1
        if head_id < 0 or head_id >= self.n_heads:
            return -1
        return self.tree.nodes[node_idx].incoming_actions[head_id]
    
    def node_visits(self, int node_idx):
        if node_idx < 0 or node_idx >= self.tree.size:
            return 0
        return int(self.tree.nodes[node_idx].visit_count)
    
    def node_virtual_loss(self, int node_idx):
        # Virtual loss is per-head per-action, not per-node in this design
        # Return 0 for node-level queries
        return 0
    
    def node_total_value(self, int node_idx):
        if node_idx < 0 or node_idx >= self.tree.size:
            return 0.0
        return self.tree.nodes[node_idx].total_value
    
    def node_q(self, int node_idx):
        if node_idx < 0 or node_idx >= self.tree.size:
            return 0.0
        cdef CNode* node = &self.tree.nodes[node_idx]
        if node.visit_count == 0:
            return 0.0
        return node.total_value / node.visit_count
    
    def root_num_children_for_head(self, int head_id):
        """Get number of children at root (same for all heads in synchronized tree)."""
        return self.root_num_children
    
    @property
    def root_visits(self):
        if not self._initialized or self.tree.root_idx < 0:
            return 0
        return int(self.tree.nodes[self.tree.root_idx].visit_count)
    
    @property
    def size(self):
        return self.tree.size
    
    @property
    def root_idx(self):
        return self.tree.root_idx