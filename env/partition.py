import random
import math
from typing import List, Set, Tuple

def build_node_subsets(
    n_input: int = 784,
    n_hidden: int = 128,
    n_output: int = 10,
    target_coverage: int = 2,  # Each node appears in this many heads (redundancy)
    subset_size: int = 64,    # Larger heads = more edge possibilities
    min_valid_edges_per_head: int = 100,  # Ensure head has sufficient valid edges
    seed: int = 42,
) -> List[List[int]]:
    """
    NEAT-style unrestrictive partitioning for multi-head MCTS.
    
    Key ideas:
    - Large overlapping heads (not small disjoint subsets)
    - Each node appears in multiple heads (target_coverage)
    - Heads automatically sized to ensure many valid edges exist
    - Number of heads determined by coverage requirements
    
    Valid edges: input→hidden, hidden→hidden, hidden→output
    (no input→input, input→output directly, output→*)
    """
    random.seed(seed)
    
    n_nodes = n_input + n_hidden + n_output
    input_nodes = set(range(n_input))
    hidden_nodes = set(range(n_input, n_input + n_hidden))
    output_nodes = set(range(n_input + n_hidden, n_nodes))
    
    # Each node needs to appear in target_coverage heads
    # We'll build heads greedily to satisfy this
    
    node_coverage_count = {n: 0 for n in range(n_nodes)}
    heads = []
    max_attempts = 10000
    
    def count_valid_edges(head_nodes: Set[int]) -> int:
        """Count valid edges possible within this head."""
        count = 0
        for u in head_nodes:
            for v in head_nodes:
                if u == v: continue
                if v in input_nodes: continue      # Can't edge TO input
                if u in output_nodes: continue     # Can't edge FROM output
                count += 1
        return count
    
    def get_underserved_nodes() -> List[int]:
        """Nodes that haven't met target coverage yet."""
        return [n for n, c in node_coverage_count.items() if c < target_coverage]
    
    for attempt in range(max_attempts):
        underserved = get_underserved_nodes()
        if not underserved:
            break  # All nodes have target coverage
        
        # Build a head starting from underserved nodes
        candidate = set()
        
        # Seed with random underserved nodes
        seed_nodes = random.sample(underserved, min(10, len(underserved)))
        candidate.update(seed_nodes)
        
        # Add nodes to reach target size and ensure validity
        # Must have: hidden nodes (for connectivity), inputs (sources), outputs (targets)
        
        # Ensure we have hidden nodes (critical for valid edges)
        if not (candidate & hidden_nodes):
            candidate.add(random.choice(list(hidden_nodes)))
        
        # Fill to target size, preferring nodes that need coverage
        available = list(set(range(n_nodes)) - candidate)
        random.shuffle(available)
        
        while len(candidate) < subset_size and available:
            # Prefer nodes with low coverage count
            available.sort(key=lambda n: node_coverage_count[n])
            node = available.pop(0)
            candidate.add(node)
        
        # Validate
        if len(candidate) < 10:  # Too small
            continue
            
        valid_edges = count_valid_edges(candidate)
        if valid_edges < min_valid_edges_per_head:
            continue
        
        # Accept this head
        heads.append(sorted(candidate))
        for n in candidate:
            node_coverage_count[n] += 1
    
    # Final verification
    print(f"Generated {len(heads)} heads")
    print(f"Node coverage: {min(node_coverage_count.values())}-{max(node_coverage_count.values())}")
    
    # Check all nodes covered
    uncovered = [n for n, c in node_coverage_count.items() if c == 0]
    if uncovered:
        raise ValueError(f"Nodes not covered: {uncovered}")
    
    # Print edge statistics
    total_valid_edges = set()
    for i, head in enumerate(heads):
        head_set = set(head)
        edges = count_valid_edges(head_set)
        # Sample some edges this head can explore
        print(f"Head {i}: {len(head)} nodes, ~{edges} valid edges")
        
        # Track unique (u,v) pairs across all heads
        for u in head_set:
            for v in head_set:
                if u != v and v not in input_nodes and u not in output_nodes:
                    total_valid_edges.add((u, v))
    
    print(f"Total unique valid edges across all heads: {len(total_valid_edges)}")
    print(f"Possible valid edges in full graph: {n_input*n_hidden + n_hidden*(n_hidden-1) + n_hidden*n_output}")
    
    return heads