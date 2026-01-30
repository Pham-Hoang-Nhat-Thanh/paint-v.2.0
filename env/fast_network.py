"""
Production-Ready Fast NASGraph
Version 1.0 - Cython Accelerated

Drop-in replacement for env.network.NASGraph with 10-50x speedup on:
- toggle_edge()
- is_valid_add()
- copy()
- get_hash()

Usage:
    from env.fast_network import NASGraph, Node
    
    # Works exactly like the original, but faster
    graph = NASGraph(n_input=3, n_hidden=5, n_output=2)
    graph.toggle_edge(0, 3)  # 20-50x faster if Cython compiled
"""

import warnings
from typing import Optional

__all__ = ['NASGraph', 'Node', 'CythonNASGraph', 'HAS_CYTHON', 'get_backend_info']

# Attempt to import Cython extension
try:
    from .network_fast import NASGraph as _FastNASGraph
    from .network_fast import CythonNASGraph
    # Node class is shared (immutable), import from original location
    from .network import Node
    
    HAS_CYTHON = True
    __version__ = "1.0-cython"
    
    # Re-export the factory function directly for speed
    NASGraph = _FastNASGraph
    
except ImportError as e:
    # Fallback to pure Python implementation
    HAS_CYTHON = False
    __version__ = "1.0-python"
    CythonNASGraph = None  # type: ignore
    
    warnings.warn(
        f"Cython network extension not compiled ({e}). "
        f"Using Python NASGraph (slower). "
        f"To compile: python setup.py build_ext --inplace",
        RuntimeWarning,
        stacklevel=2
    )
    
    # Fallback imports
    from .network import NASGraph, Node


def get_backend_info() -> dict:
    """Return information about which backend is active."""
    return {
        'has_cython': HAS_CYTHON,
        'backend': 'cython' if HAS_CYTHON else 'python',
        'version': __version__,
        'node_class': Node.__name__
    }