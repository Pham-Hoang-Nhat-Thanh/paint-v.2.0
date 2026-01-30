# network_fast.pxd
# Must be in SAME folder as network_fast.pyx

ctypedef unsigned char uint8
ctypedef long long int64

cdef class CythonNASGraph:
    # C-level scalars - no 'readonly' or 'public'!
    cdef readonly int n_input, n_hidden, n_output, n_nodes
    cdef readonly int _max_nodes
    cdef readonly int64 _hash
    cdef readonly bint _adj_dirty
    
    # Python objects
    cdef readonly object nodes
    cdef readonly object _zobrist
    cdef readonly object _adj_cache
    
    # Memoryviews
    cdef uint8[:, ::1] _adj_matrix
    cdef uint8[::1] _node_types
    cdef int[::1] _topo_order
    cdef int[::1] _position
    
    # Public API methods
    cpdef str _get_type(self, int idx)
    cpdef int get_hash(self)
    cpdef bint toggle_edge(self, int u_idx, int v_idx) except -1
    cpdef bint is_valid_add(self, int u_idx, int v_idx) except -1
    cpdef CythonNASGraph copy(self)
    cpdef object to_pyg_data(self)
    
    # Internal nogil methods MUST be declared here
    cdef bint _is_valid_add_c(self, int u_idx, int v_idx) nogil
    cdef bint _reachable_c(self, int start_idx, int target_idx) nogil
    cdef void _reorder_topo(self, int u_idx, int v_idx) nogil