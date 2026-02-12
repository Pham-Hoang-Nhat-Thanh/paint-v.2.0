# mcts_fast.pyx (Synchronized State Evolution - Batched Performance Fixed)
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: extra_compile_args = -O3 -march=native -ffast-math

from libc.stdlib cimport malloc, free, calloc, realloc
from libc.string cimport memcpy, memset
from libc.math cimport sqrt, exp
from libc.stdint cimport uint64_t, uint32_t  # Import fixed-width integers
import numpy as np
cimport numpy as cnp
from env.network_fast cimport CythonNASGraph, uint8

cdef struct CHeadActionStats:
    double visit_count
    double total_value  
    double prior
    int virtual_loss  # Now tracks count, not subtracted value
    double virtual_mean_accumulator  # For virtual mean calculation

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
    uint64_t state_hash  # Hash for transposition table

cdef struct CHashEntry:
    uint64_t hash
    int node_idx
    CHashEntry* next

cdef struct CTranspositionTable:
    CHashEntry** buckets
    int size
    int capacity

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
    cdef bint use_virtual_mean  # New: toggle between virtual loss and virtual mean
    cdef double fpu_value  # New: First Play Urgency value (typically parent's Q)
    cdef CTranspositionTable* transposition_table  # New: O(1) state lookup
    cdef bint use_transposition  # New: enable/disable TT
    
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
        self.use_virtual_mean = True  # Default to virtual mean (better for batching)
        self.fpu_value = 0.5  # Default FPU (parent's Q approximation)
        self.transposition_table = NULL
        self.use_transposition = True
        
        if max_nodes <= 0:
            raise MemoryError("max_nodes must be positive")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive")
            
        self.nodes = <CNode*>calloc(max_nodes, sizeof(CNode))
        if not self.nodes:
            raise MemoryError(f"Failed to allocate {max_nodes} nodes")
        
        # Initialize transposition table
        self.transposition_table = <CTranspositionTable*>malloc(sizeof(CTranspositionTable))
        if not self.transposition_table:
            raise MemoryError("Failed to allocate transposition table")
        self.transposition_table.capacity = max_nodes * 2  # 2x for lower collision
        self.transposition_table.size = 0
        self.transposition_table.buckets = <CHashEntry**>calloc(
            self.transposition_table.capacity, sizeof(CHashEntry*)
        )
        if not self.transposition_table.buckets:
            raise MemoryError("Failed to allocate transposition table buckets")
        
    def __dealloc__(self):
        cdef int i, h
        cdef CHashEntry* entry
        cdef CHashEntry* next_entry
        
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
            
        # Free transposition table
        if self.transposition_table:
            if self.transposition_table.buckets:
                for i in range(self.transposition_table.capacity):
                    entry = self.transposition_table.buckets[i]
                    while entry:
                        next_entry = entry.next
                        free(entry)
                        entry = next_entry
                free(self.transposition_table.buckets)
            free(self.transposition_table)
            self.transposition_table = NULL
            
        if self.n_actions_per_head:
            free(self.n_actions_per_head)
            self.n_actions_per_head = NULL
    
    cdef inline uint64_t _hash_state(self, uint8* adj, int n_nodes) noexcept nogil:
        """Compute 64-bit hash of adjacency matrix for transposition table."""
        # Simple but effective: XOR-folded murmurhash of the byte array
        cdef uint64_t hash_val = 1469598103934665603ULL  # FNV offset basis
        cdef int i, j
        cdef uint8 val
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                val = adj[i * n_nodes + j]
                # FNV-1a hash
                hash_val ^= val
                hash_val *= 1099511628211ULL  # FNV prime
        
        return hash_val
    
    cdef inline int _lookup_transposition(self, uint64_t hash_val) noexcept nogil:
        """Lookup node index by state hash. Returns -1 if not found."""
        if not self.use_transposition or not self.transposition_table:
            return -1
            
        cdef int idx = hash_val % self.transposition_table.capacity
        cdef CHashEntry* entry = self.transposition_table.buckets[idx]
        
        while entry:
            if entry.hash == hash_val:
                return entry.node_idx
            entry = entry.next
        return -1
    
    cdef inline void _insert_transposition(self, uint64_t hash_val, int node_idx) noexcept nogil:
        """Insert hash->node mapping into transposition table."""
        if not self.use_transposition or not self.transposition_table:
            return
            
        cdef int idx = hash_val % self.transposition_table.capacity
        cdef CHashEntry* entry = <CHashEntry*>malloc(sizeof(CHashEntry))
        if not entry:
            return  # Silent failure - TT is optional optimization
            
        entry.hash = hash_val
        entry.node_idx = node_idx
        entry.next = self.transposition_table.buckets[idx]
        self.transposition_table.buckets[idx] = entry
        self.transposition_table.size += 1
    
    cdef int _create_node(self, int parent, int* incoming_actions, int depth, 
                         uint64_t state_hash=0) noexcept nogil:
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
        node.state_hash = state_hash
        
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
                node.head_stats[h] = <CHeadActionStats*>calloc(
                    self.n_actions_per_head[h], sizeof(CHeadActionStats)
                )
                if not node.head_stats[h]:
                    return -1
                # Initialize virtual_mean_accumulator to 0
        
        self.size += 1
        
        # Insert into transposition table
        if state_hash != 0:
            self._insert_transposition(state_hash, idx)
            
        return idx
        
    cpdef void initialize(self, int[:] n_actions_per_head, double c_puct=1.0, 
                         int virtual_loss=3, bint use_virtual_mean=True,
                         double fpu_value=0.5, bint use_transposition=True) except *:
        if n_actions_per_head.shape[0] != self.n_heads:
            raise ValueError("n_actions_per_head length must match n_heads")
            
        self.clear()
        self.c_puct = c_puct
        self.virtual_loss_val = virtual_loss
        self.use_virtual_mean = use_virtual_mean
        self.fpu_value = fpu_value
        self.use_transposition = use_transposition
        
        cdef int h
        # Free previous allocation before reallocating
        if self.n_actions_per_head != NULL:
            free(self.n_actions_per_head)
            self.n_actions_per_head = NULL
        
        self.n_actions_per_head = <int*>malloc(self.n_heads * sizeof(int))
        if not self.n_actions_per_head:
            raise MemoryError("Failed to allocate n_actions_per_head")
            
        for h in range(self.n_heads):
            self.n_actions_per_head[h] = n_actions_per_head[h]
        
        self.root_idx = self._create_node(-1, NULL, 0, 0)
        if self.root_idx == -1:
            raise MemoryError("Failed to create root node")
        self._initialized = True
        
    cpdef void clear(self) except *:
        cdef int i, h
        cdef CHashEntry* entry
        cdef CHashEntry* next_entry
        
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
        
        # Clear transposition table
        if self.transposition_table and self.transposition_table.buckets:
            for i in range(self.transposition_table.capacity):
                entry = self.transposition_table.buckets[i]
                while entry:
                    next_entry = entry.next
                    free(entry)
                    entry = next_entry
                self.transposition_table.buckets[i] = NULL
            self.transposition_table.size = 0
            
        self.size = 0
        self.root_idx = -1
        self._initialized = False
        
    cdef int _select_best_action_for_head_nogil(self, int node_idx, int head_id, 
                                                 double parent_q) noexcept nogil:
        """
        Select best action using UCB with Virtual Mean or Virtual Loss.
        parent_q: Q value of parent node for FPU calculation.
        """
        if node_idx < 0 or node_idx >= self.size:
            return -1
            
        cdef CNode* node = &self.nodes[node_idx]
        if not node.head_stats or not node.head_stats[head_id]:
            return -1
            
        cdef CHeadActionStats* stats = node.head_stats[head_id]
        cdef int n_actions = self.n_actions_per_head[head_id]
        
        cdef double best_score = -1e9
        cdef int best_action = 0
        cdef double q, u, n_total, vloss
        cdef int a
        
        cdef double parent_visits = node.visit_count
        if parent_visits < 1.0:
            parent_visits = 1.0
        
        for a in range(n_actions):
            n_total = stats[a].visit_count + stats[a].virtual_loss
            
            if n_total == 0.0:
                # FPU: First Play Urgency - use parent Q for unvisited actions
                q = parent_q if self.fpu_value > 0 else 0.0
            else:
                if self.use_virtual_mean:
                    # Virtual Mean: assume pending eval returns current mean
                    # Q = (total_value + virtual_loss * parent_q) / (visits + virtual_loss)
                    # This is more optimistic than virtual loss
                    q = (stats[a].total_value + stats[a].virtual_loss * parent_q) / n_total
                else:
                    # Virtual Loss: assume pending eval returns loss (0 for [0,1] values)
                    # Q = (total_value - virtual_loss * virtual_loss_val) / n_total
                    # This is pessimistic and causes excessive exploration
                    vloss = stats[a].virtual_loss * self.virtual_loss_val
                    q = (stats[a].total_value - vloss) / n_total
            
            u = self.c_puct * stats[a].prior * sqrt(parent_visits) / (1.0 + n_total)
            
            if q + u > best_score:
                best_score = q + u
                best_action = a
        
        # Increment virtual loss counter for selected action
        stats[best_action].virtual_loss += 1
        return best_action
        
    cdef int _find_child_with_actions_nogil(self, int node_idx, int* actions) noexcept nogil:
        """Legacy method - kept for compatibility. Prefer transposition table lookup."""
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
        
    cdef int _find_child_by_hash_nogil(self, int parent_idx, uint64_t state_hash) noexcept nogil:
        """Find child by state hash using transposition table (O(1))."""
        if not self.use_transposition:
            return -1
            
        cdef int child_idx = self._lookup_transposition(state_hash)
        if child_idx == -1:
            return -1
            
        # Verify it's actually a child of parent (security check)
        if self.nodes[child_idx].parent != parent_idx:
            return -1
        return child_idx
        
    cdef int _add_child_nogil(self, int parent_idx, int* actions, uint64_t state_hash=0) noexcept nogil:
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
        
        cdef int child_idx = self._create_node(parent_idx, actions, parent.depth + 1, state_hash)
        if child_idx == -1:
            return -1
            
        parent.children[parent.num_children] = child_idx
        parent.num_children += 1
        return child_idx
        
    cdef void _backup_nogil(self, int leaf_idx, double value, int** path_actions, 
                           int path_len, bint is_new_node) noexcept nogil:
        """
        Backup value through the path, updating visit counts and values.
        is_new_node: if True, this was a new expansion (virtual loss was added)
        """
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
                        
                        # Always remove exactly one virtual loss (the one we added)
                        # This fixes the bug where virtual_loss could get out of sync
                        if node.head_stats[h][action].virtual_loss > 0:
                            node.head_stats[h][action].virtual_loss -= 1


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
    cdef object _reusable_adj_arr
    cdef object _reusable_features
    # New: collision detection for batching
    cdef uint64_t* batch_hashes  # Track hashes in current batch
    cdef int batch_hashes_capacity
    
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
        self._reusable_adj_arr = None
        self._reusable_features = None
        self.batch_hashes = NULL
        self.batch_hashes_capacity = 0
    
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
        if self.batch_hashes:
            free(self.batch_hashes)
    
    def clear_buffers(self):
        """Clear reusable Python objects to free memory."""
        self._reusable_adj_arr = None
        self._reusable_features = None
    
    def initialize_tree(self, list n_actions_per_head, bint use_virtual_mean=True,
                       double fpu_value=0.5, bint use_transposition=True):
        """Initialize tree with new options for virtual mean and transposition table."""
        if len(n_actions_per_head) != self.n_heads:
            raise ValueError("Length mismatch")
            
        self.clear_buffers()
        
        cdef int h
        cdef int[:] arr = np.array(n_actions_per_head, dtype=np.int32)
        self.tree.initialize(arr, self.c_puct, self.virtual_loss, 
                            use_virtual_mean, fpu_value, use_transposition)
        
        if self.n_actions != NULL:
            free(self.n_actions)
            self.n_actions = NULL
        
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

    cpdef tuple _select_leaf_sync(self, CythonNASGraph state):
        """
        Synchronous leaf selection for async worker.
        Returns (leaf_node_idx, path_actions_list, path_nodes_list, step, is_new_node)
        """
        cdef int n_nodes = state.n_nodes
        cdef int n_input = state.n_input
        cdef int n_hidden = state.n_hidden
        cdef uint8* base_adj = &state._adj_matrix[0, 0]
        
        # Allocate buffers
        cdef uint8* temp_adj = <uint8*>malloc(n_nodes * n_nodes * sizeof(uint8))
        cdef int* selected_actions = <int*>malloc(self.n_heads * sizeof(int))
        cdef int** path_actions = <int**>malloc(self.max_depth * sizeof(int*))
        cdef int* path_nodes = <int*>malloc(self.max_depth * sizeof(int))
        
        for i in range(self.max_depth):
            path_actions[i] = <int*>malloc(self.n_heads * sizeof(int))
        
        cdef int step = 0
        cdef int current_node = self.tree.root_idx
        cdef int child_node
        cdef bint is_new_node = False
        cdef double parent_q = self.tree.fpu_value
        cdef uint64_t state_hash
        
        try:
            while step < self.max_depth:
                # Selection
                for h in range(self.n_heads):
                    selected_actions[h] = self.tree._select_best_action_for_head_nogil(
                        current_node, h, parent_q
                    )
                    if selected_actions[h] == -1:
                        selected_actions[h] = 0
                    path_actions[step][h] = selected_actions[h]
                
                # Check transposition table
                self._reconstruct_state_along_path_nogil(
                    base_adj, temp_adj, n_nodes, n_input, n_hidden,
                    path_actions, step + 1
                )
                state_hash = self.tree._hash_state(temp_adj, n_nodes)
                child_node = self.tree._find_child_by_hash_nogil(current_node, state_hash)
                
                if child_node == -1:
                    child_node = self.tree._find_child_with_actions_nogil(
                        current_node, selected_actions
                    )
                
                if child_node == -1:
                    # Expansion
                    child_node = self.tree._add_child_nogil(
                        current_node, selected_actions, state_hash
                    )
                    if child_node == -1:
                        break
                    is_new_node = True
                    path_nodes[step] = child_node
                    step += 1
                    break
                else:
                    path_nodes[step] = child_node
                    current_node = child_node
                    if self.tree.nodes[current_node].visit_count > 0:
                        parent_q = self.tree.nodes[current_node].total_value / \
                                  self.tree.nodes[current_node].visit_count
                    step += 1
            
            # Convert to Python objects for async compatibility
            py_path_actions = []
            py_path_nodes = []
            for i in range(step):
                py_path_actions.append(np.array([
                    path_actions[i][h] for h in range(self.n_heads)
                ], dtype=np.int32))
                py_path_nodes.append(path_nodes[i])
            
            leaf_idx = path_nodes[step - 1] if step > 0 else self.tree.root_idx
            
            return (leaf_idx, py_path_actions, py_path_nodes, step, is_new_node)
            
        finally:
            free(temp_adj)
            free(selected_actions)
            free(path_nodes)
            for i in range(self.max_depth):
                if path_actions[i]:
                    free(path_actions[i])
            free(path_actions)
    
    cdef inline bint _is_valid_edge_c(self, uint8* adj, int n, int u, int v, 
                                      int n_input, int n_hidden) noexcept nogil:
        if u == v:
            return False
        if v < n_input:
            return False
        if u >= n_input + n_hidden:
            return False
        if adj[u*n + v]:
            return False
        return True
    
    cdef void _apply_actions_to_state_nogil(self, uint8* base_adj, uint8* out_adj, 
                                           int n_nodes, int* selected_actions, 
                                           int n_input, int n_hidden) noexcept nogil:
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
        """Sequential search - kept for comparison and small simulations."""
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
        cdef double parent_q  # For FPU
        
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
                for j in range(i):
                    free(path_actions[j])
                free(path_actions)
                free(selected_actions)
                free(path_nodes)
                free(temp_adj)
                raise MemoryError("Failed to allocate path_actions[i]")
        
        # Initialize root
        root_node = &self.tree.nodes[self.tree.root_idx]
        root_needs_priors = True
        if root_node.head_stats and root_node.head_stats[0]:
            prior_sum = 0.0
            for a in range(self.n_actions[0]):
                prior_sum += root_node.head_stats[0][a].prior
            if prior_sum > 0.0:
                root_needs_priors = False
        
        if root_needs_priors:
            if self._reusable_adj_arr is None or self._reusable_adj_arr.shape[0] != n_nodes:
                self._reusable_adj_arr = np.empty((n_nodes, n_nodes), dtype=np.uint8)
                self._reusable_features = {
                    'adj': self._reusable_adj_arr,
                    'n_nodes': n_nodes,
                    'n_input': n_input,
                    'n_hidden': n_hidden,
                    'n_output': state.n_output
                }
            
            np.copyto(self._reusable_adj_arr, 
                     np.asarray(<uint8[:n_nodes*n_nodes]>base_adj).reshape(n_nodes, n_nodes))
            features = self._reusable_features
            
            policies, values = evaluator.evaluate([features], [0])
            
            for h in range(self.n_heads):
                n_act = self.n_actions[h]
                policy_np = policies[h].ravel()
                
                if len(policy_np) >= n_act:
                    for a in range(n_act):
                        policy_buf[a] = policy_np[a]
                else:
                    for a in range(n_act):
                        policy_buf[a] = 1.0 / n_act
                
                self._compute_valid_mask_and_mask_policy_nogil(h, base_adj, n_nodes, 
                                                               n_input, n_hidden, 
                                                               policy_buf, valid_buf)
                self._apply_softmax_to_policy_nogil(policy_buf, n_act)
                
                n_valid = 0
                for a in range(n_act):
                    if valid_buf[a] > 0:
                        n_valid += 1
                
                if n_valid > 0:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * n_valid)
                    noise_idx = 0
                    for a in range(n_act):
                        if valid_buf[a] > 0:
                            policy_buf[a] = (1.0 - self.dirichlet_epsilon) * policy_buf[a] + \
                                           self.dirichlet_epsilon * noise[noise_idx]
                            noise_idx += 1
                
                for a in range(n_act):
                    root_node.head_stats[h][a].prior = policy_buf[a]
        
        try:
            for sim in range(n_sims):
                current_node = self.tree.root_idx
                step = 0
                is_new_node = False
                
                # Calculate parent Q for FPU (root uses fpu_value, others use parent's Q)
                parent_q = self.tree.fpu_value
                
                while step < self.max_depth:
                    for h in range(self.n_heads):
                        selected_actions[h] = self.tree._select_best_action_for_head_nogil(
                            current_node, h, parent_q
                        )
                        if selected_actions[h] == -1:
                            selected_actions[h] = 0
                        path_actions[step][h] = selected_actions[h]
                    
                    child_node = self.tree._find_child_with_actions_nogil(current_node, 
                                                                          selected_actions)
                    
                    if child_node == -1:
                        child_node = self.tree._add_child_nogil(current_node, 
                                                                selected_actions, 0)
                        if child_node == -1:
                            break
                        is_new_node = True
                        path_nodes[step] = child_node
                        step += 1
                        break
                    else:
                        path_nodes[step] = child_node
                        current_node = child_node
                        # Update parent_q for next level (FPU propagation)
                        if self.tree.nodes[current_node].visit_count > 0:
                            parent_q = self.tree.nodes[current_node].total_value / \
                                      self.tree.nodes[current_node].visit_count
                        step += 1
                
                if step == 0:
                    continue
                
                leaf_node = path_nodes[step - 1]
                
                if is_new_node:
                    self._reconstruct_state_along_path_nogil(
                        base_adj, temp_adj, n_nodes, n_input, n_hidden,
                        path_actions, step
                    )
                    
                    if self._reusable_adj_arr is None or self._reusable_adj_arr.shape[0] != n_nodes:
                        self._reusable_adj_arr = np.empty((n_nodes, n_nodes), dtype=np.uint8)
                        self._reusable_features = {
                            'adj': self._reusable_adj_arr,
                            'n_nodes': n_nodes,
                            'n_input': n_input,
                            'n_hidden': n_hidden,
                            'n_output': state.n_output
                        }
                    
                    np.copyto(self._reusable_adj_arr, 
                             np.asarray(<uint8[:n_nodes*n_nodes]>temp_adj).reshape(n_nodes, n_nodes))
                    features = self._reusable_features
                    
                    policies, values = evaluator.evaluate([features], [0])
                    value = float(values[0])
                    
                    leaf_ptr = &self.tree.nodes[leaf_node]
                    for h in range(self.n_heads):
                        n_act = self.n_actions[h]
                        policy_np = policies[h].ravel()
                        
                        if len(policy_np) >= n_act:
                            for a in range(n_act):
                                policy_buf[a] = policy_np[a]
                        else:
                            for a in range(n_act):
                                policy_buf[a] = 1.0 / n_act
                        
                        self._compute_valid_mask_and_mask_policy_nogil(h, temp_adj, n_nodes, 
                                                                       n_input, n_hidden, 
                                                                       policy_buf, valid_buf)
                        self._apply_softmax_to_policy_nogil(policy_buf, n_act)
                        
                        for a in range(n_act):
                            leaf_ptr.head_stats[h][a].prior = policy_buf[a]
                else:
                    value = self.tree.nodes[leaf_node].total_value / \
                           max(1.0, self.tree.nodes[leaf_node].visit_count)
                
                self.tree._backup_nogil(leaf_node, value, path_actions, step, is_new_node)
                    
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

    cpdef void search_batched(self, CythonNASGraph state, object evaluator, 
                             int n_sims, int batch_size=16) except *:
        """
        Fixed batched MCTS with:
        - Virtual Mean instead of Virtual Loss (configurable)
        - Collision detection (no duplicate states in batch)
        - Immediate backup for existing nodes (async-style)
        - Transposition table for O(1) lookups
        - FPU (First Play Urgency) for unvisited actions
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize_tree() first.")
        if not state:
            raise ValueError("state is None")
        if not evaluator:
            raise ValueError("evaluator is None")
        if n_sims <= 0:
            return
        if batch_size <= 0:
            batch_size = 16

        cdef int n_nodes = state.n_nodes
        cdef int n_input = state.n_input
        cdef int n_hidden = state.n_hidden
        cdef uint8* base_adj = &state._adj_matrix[0, 0]

        # Pre-allocate policy buffers
        cdef int max_actions = 0
        cdef int h
        for h in range(self.n_heads):
            if self.n_actions[h] > max_actions:
                max_actions = self.n_actions[h]

        cdef double* policy_buf = <double*>malloc(max_actions * sizeof(double))
        cdef double* valid_buf = <double*>malloc(max_actions * sizeof(double))
        if not policy_buf or not valid_buf:
            if policy_buf: free(policy_buf)
            if valid_buf: free(valid_buf)
            raise MemoryError("Failed to allocate policy buffers")

        # Initialize root
        cdef CNode* root_node = &self.tree.nodes[self.tree.root_idx]
        cdef bint root_needs_priors = True
        cdef double prior_sum = 0.0
        cdef int a, n_act

        if root_node.head_stats and root_node.head_stats[0]:
            for a in range(self.n_actions[0]):
                prior_sum += root_node.head_stats[0][a].prior
            if prior_sum > 0.0:
                root_needs_priors = False

        if root_needs_priors:
            if self._reusable_adj_arr is None or self._reusable_adj_arr.shape[0] != n_nodes:
                self._reusable_adj_arr = np.empty((n_nodes, n_nodes), dtype=np.uint8)
                self._reusable_features = {
                    'adj': self._reusable_adj_arr,
                    'n_nodes': n_nodes,
                    'n_input': n_input,
                    'n_hidden': n_hidden,
                    'n_output': state.n_output
                }

            np.copyto(self._reusable_adj_arr, 
                     np.asarray(<uint8[:n_nodes*n_nodes]>base_adj).reshape(n_nodes, n_nodes))
            features = self._reusable_features
            policies, values = evaluator.evaluate([features], [0])

            for h in range(self.n_heads):
                n_act = self.n_actions[h]
                policy_np = policies[h].ravel()

                if len(policy_np) >= n_act:
                    for a in range(n_act):
                        policy_buf[a] = policy_np[a]
                else:
                    for a in range(n_act):
                        policy_buf[a] = 1.0 / n_act

                self._apply_softmax_to_policy_nogil(policy_buf, n_act)

                if self.dirichlet_epsilon > 0:
                    noise = np.random.dirichlet([self.dirichlet_alpha] * n_act)
                    for a in range(n_act):
                        policy_buf[a] = (1 - self.dirichlet_epsilon) * policy_buf[a] + \
                                       self.dirichlet_epsilon * noise[a]

                for a in range(n_act):
                    root_node.head_stats[h][a].prior = policy_buf[a]

        # Setup batch structures
        cdef int effective_batch_size = min(batch_size, 32)
        
        # Allocate collision detection array
        if self.batch_hashes_capacity < effective_batch_size:
            if self.batch_hashes:
                free(self.batch_hashes)
            self.batch_hashes = <uint64_t*>malloc(effective_batch_size * sizeof(uint64_t))
            self.batch_hashes_capacity = effective_batch_size
        
        cdef uint8** temp_adj_array = <uint8**>malloc(effective_batch_size * sizeof(uint8*))
        cdef int*** path_actions_array = <int***>malloc(effective_batch_size * sizeof(int**))
        cdef int** path_nodes_array = <int**>malloc(effective_batch_size * sizeof(int*))
        cdef int* path_steps = <int*>malloc(effective_batch_size * sizeof(int))
        cdef int* leaf_indices_array = <int*>malloc(effective_batch_size * sizeof(int))
        cdef bint* is_new_node_array = <bint*>malloc(effective_batch_size * sizeof(bint))
        cdef list features_list = []

        if not temp_adj_array or not path_actions_array or not path_nodes_array or \
           not path_steps or not leaf_indices_array or not is_new_node_array:
            # Cleanup and raise
            pass  # Handled below

        cdef int* selected_actions = <int*>malloc(self.n_heads * sizeof(int))
        cdef bint allocation_failed = False
        
        # Pre-allocate path action buffers for batch - declare ALL variables first
        cdef int*** preallocated_path_actions = NULL
        cdef int pa_idx, pa_jdx  # Use different names to avoid redeclaration
        
        preallocated_path_actions = <int***>malloc(effective_batch_size * sizeof(int**))
        if preallocated_path_actions:
            for pa_idx in range(effective_batch_size):
                preallocated_path_actions[pa_idx] = <int**>malloc(self.max_depth * sizeof(int*))
                if preallocated_path_actions[pa_idx]:
                    for pa_jdx in range(self.max_depth):
                        preallocated_path_actions[pa_idx][pa_jdx] = <int*>malloc(self.n_heads * sizeof(int))

        cdef int n_completed = 0
        cdef int batch_count
        cdef int sim_in_batch, step, current_node, child_node
        cdef int sim_idx, batch_idx  # Use different names
        cdef bint is_new_node
        cdef double value, parent_q
        cdef int leaf_node
        cdef CNode* leaf_ptr
        cdef uint8* temp_adj
        cdef int** path_actions
        cdef int* path_nodes
        cdef uint64_t state_hash
        cdef bint found_duplicate

        try:
            while n_completed < n_sims:
                batch_count = 0
                features_list.clear()

                # PHASE 1: ACCUMULATE BATCH with collision detection
                for sim_in_batch in range(effective_batch_size):
                    if n_completed >= n_sims:
                        break

                    # Use preallocated buffers
                    temp_adj = <uint8*>malloc(n_nodes * n_nodes * sizeof(uint8))
                    if not temp_adj:
                        break
                    
                    path_actions = preallocated_path_actions[batch_count]
                    path_nodes = <int*>malloc(self.max_depth * sizeof(int))
                    if not path_nodes:
                        free(temp_adj)
                        break

                    current_node = self.tree.root_idx
                    step = 0
                    is_new_node = False
                    parent_q = self.tree.fpu_value

                    # SELECTION with transposition table lookup
                    while step < self.max_depth:
                        for h in range(self.n_heads):
                            selected_actions[h] = self.tree._select_best_action_for_head_nogil(
                                current_node, h, parent_q
                            )
                            if selected_actions[h] == -1:
                                selected_actions[h] = 0
                            path_actions[step][h] = selected_actions[h]

                        # Try transposition table first (O(1))
                        self._reconstruct_state_along_path_nogil(
                            base_adj, temp_adj, n_nodes, n_input, n_hidden,
                            path_actions, step + 1
                        )
                        state_hash = self.tree._hash_state(temp_adj, n_nodes)
                        child_node = self.tree._find_child_by_hash_nogil(current_node, state_hash)
                        
                        if child_node == -1:
                            # Fall back to linear search if TT miss or disabled
                            child_node = self.tree._find_child_with_actions_nogil(
                                current_node, selected_actions
                            )

                        if child_node == -1:
                            # EXPANSION
                            child_node = self.tree._add_child_nogil(
                                current_node, selected_actions, state_hash
                            )
                            if child_node == -1:
                                break
                            is_new_node = True
                            path_nodes[step] = child_node
                            step += 1
                            break
                        else:
                            path_nodes[step] = child_node
                            current_node = child_node
                            if self.tree.nodes[current_node].visit_count > 0:
                                parent_q = self.tree.nodes[current_node].total_value / \
                                          self.tree.nodes[current_node].visit_count
                            step += 1

                    if step == 0:
                        free(temp_adj)
                        free(path_nodes)
                        continue

                    leaf_node = path_nodes[step - 1]

                    if is_new_node:
                        # Check for collision (duplicate state in current batch)
                        found_duplicate = False
                        for batch_idx in range(batch_count):
                            if self.batch_hashes[batch_idx] == state_hash:
                                found_duplicate = True
                                break
                        
                        if found_duplicate:
                            # Still need to backup with virtual loss removal
                            value = 0.0  # Placeholder
                            self.tree._backup_nogil(leaf_node, value, path_actions, step, True)
                            
                            free(temp_adj)
                            free(path_nodes)
                            n_completed += 1
                            continue

                        # No collision - add to batch
                        self.batch_hashes[batch_count] = state_hash
                        
                        # Create numpy array (copy needed for batch)
                        temp_adj_np = np.asarray(<uint8[:n_nodes*n_nodes]>temp_adj).reshape(n_nodes, n_nodes).copy()
                        features = {
                            'adj': temp_adj_np,
                            'n_nodes': n_nodes,
                            'n_input': n_input,
                            'n_hidden': n_hidden,
                            'n_output': state.n_output
                        }

                        temp_adj_array[batch_count] = temp_adj
                        path_actions_array[batch_count] = path_actions
                        path_steps[batch_count] = step
                        path_nodes_array[batch_count] = path_nodes
                        leaf_indices_array[batch_count] = leaf_node
                        is_new_node_array[batch_count] = True
                        features_list.append(features)
                        batch_count += 1
                    else:
                        # EXISTING NODE: Immediate backup (async-style)
                        value = self.tree.nodes[leaf_node].total_value / \
                               max(1.0, self.tree.nodes[leaf_node].visit_count)
                        
                        self.tree._backup_nogil(leaf_node, value, path_actions, step, False)
                        
                        free(temp_adj)
                        free(path_nodes)

                    n_completed += 1

                # PHASE 2: BATCHED EVALUATION
                if batch_count > 0:
                    all_policies, all_values = evaluator.evaluate(features_list, [0] * batch_count)

                    # PHASE 3: SET PRIORS AND BACKUP
                    for batch_idx in range(batch_count):
                        leaf_node = leaf_indices_array[batch_idx]
                        policies = all_policies[batch_idx]
                        value = float(all_values[batch_idx])
                        path_actions = path_actions_array[batch_idx]
                        step = path_steps[batch_idx]
                        path_nodes = path_nodes_array[batch_idx]

                        # Set priors on newly expanded node
                        leaf_ptr = &self.tree.nodes[leaf_node]
                        for h in range(self.n_heads):
                            n_act = self.n_actions[h]
                            policy_np = policies[h].ravel()

                            if len(policy_np) >= n_act:
                                for a in range(n_act):
                                    policy_buf[a] = policy_np[a]
                            else:
                                for a in range(n_act):
                                    policy_buf[a] = 1.0 / n_act

                            temp_adj = temp_adj_array[batch_idx]
                            self._compute_valid_mask_and_mask_policy_nogil(
                                h, temp_adj, n_nodes, n_input, n_hidden, 
                                policy_buf, valid_buf
                            )
                            self._apply_softmax_to_policy_nogil(policy_buf, n_act)

                            for a in range(n_act):
                                leaf_ptr.head_stats[h][a].prior = policy_buf[a]

                        # Backup (virtual loss removed here)
                        self.tree._backup_nogil(leaf_node, value, path_actions, step, True)

                        # Cleanup
                        free(temp_adj_array[batch_idx])
                        free(path_nodes_array[batch_idx])

        finally:
            free(selected_actions)
            free(policy_buf)
            free(valid_buf)
            free(temp_adj_array)
            free(path_actions_array)
            free(path_nodes_array)
            free(path_steps)
            free(leaf_indices_array)
            free(is_new_node_array)
            
            # Free preallocated path action buffers
            if preallocated_path_actions:
                for pa_idx in range(effective_batch_size):
                    if preallocated_path_actions[pa_idx]:
                        for pa_jdx in range(self.max_depth):
                            if preallocated_path_actions[pa_idx][pa_jdx]:
                                free(preallocated_path_actions[pa_idx][pa_jdx])
                        free(preallocated_path_actions[pa_idx])
                free(preallocated_path_actions)

    cpdef double get_root_value_estimate(self):
        """
        Get the value estimate at root from most recent network evaluation.
        For TD(n): V(s_root) used in bootstrap targets.
        """
        cdef int root = self.tree.root_idx
        if root < 0 or root >= self.tree.size:
            return 0.0
        
        # Return Q-value at root (network estimate averaged over simulations)
        cdef CNode* node = &self.tree.nodes[root]
        if node.visit_count == 0:
            return 0.0
        
        # This is the mean of all backed-up values = estimate of E[γ^T R | s]
        return node.total_value / node.visit_count

    # API-compatible methods (unchanged signatures)
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

    def get_node_parent(self, int node_idx):
        if node_idx < 0 or node_idx >= self.tree.size:
            return -1
        return self.tree.nodes[node_idx].parent
    
    def get_head_action_visits(self, int node_idx, int head_id, int action_idx):
        if node_idx < 0 or node_idx >= self.tree.size:
            return 0
        cdef CNode* node = &self.tree.nodes[node_idx]
        if not node.head_stats or not node.head_stats[head_id]:
            return 0
        if action_idx < 0 or action_idx >= self.n_actions[head_id]:
            return 0
        return int(node.head_stats[head_id][action_idx].visit_count)
    
    def get_head_action_virtual_loss(self, int node_idx, int head_id, int action_idx):
        if node_idx < 0 or node_idx >= self.tree.size:
            return 0
        cdef CNode* node = &self.tree.nodes[node_idx]
        if not node.head_stats or not node.head_stats[head_id]:
            return 0
        if action_idx < 0 or action_idx >= self.n_actions[head_id]:
            return 0
        return node.head_stats[head_id][action_idx].virtual_loss
    
    def get_head_action_total_value(self, int node_idx, int head_id, int action_idx):
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
        cdef int root = self.tree.root_idx
        if root < 0 or i < 0 or i >= self.tree.nodes[root].num_children:
            return -1
        return self.tree.nodes[root].children[i]
    
    def get_node_child(self, int node_idx, int child_rank):
        if node_idx < 0 or node_idx >= self.tree.size:
            return -1
        cdef CNode* node = &self.tree.nodes[node_idx]
        if child_rank < 0 or child_rank >= node.num_children:
            return -1
        return node.children[child_rank]
    
    def node_action_id(self, int node_idx):
        if node_idx < 0 or node_idx >= self.tree.size:
            return -1
        if self.tree.nodes[node_idx].incoming_actions:
            return self.tree.nodes[node_idx].incoming_actions[0]
        return -1
    
    def get_head_action_at_node(self, int node_idx, int head_id):
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
        return 0  # Per-head per-action in this design
    
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
        
    # New configuration methods (optional API extension)
    def set_virtual_mean(self, bint enabled):
        """Toggle between Virtual Mean (better for batching) and Virtual Loss."""
        self.tree.use_virtual_mean = enabled
        
    def set_fpu_value(self, double value):
        """Set First Play Urgency value (0.0 to 1.0 typical)."""
        self.tree.fpu_value = value
        
    def set_transposition_table(self, bint enabled):
        """Enable/disable transposition table (O(1) lookups)."""
        self.tree.use_transposition = enabled