import random

def build_node_subsets(
    n_input=784,
    n_hidden=128,
    n_output=10,
    subset_size=32,
    extra_overlap_subsets=0,
    seed=42,
):
    random.seed(seed)

    n_nodes = n_input + n_hidden + n_output
    all_nodes = list(range(n_nodes))

    # -------------------------
    # Layer A: full coverage
    # -------------------------
    random.shuffle(all_nodes)

    subsets = []
    for i in range(0, n_nodes, subset_size):
        subset = all_nodes[i:i + subset_size]

        # pad last subset if needed (overlap allowed)
        if len(subset) < subset_size:
            subset += random.sample(all_nodes, subset_size - len(subset))

        subsets.append(subset)

    # -------------------------
    # Layer B: optional overlaps
    # -------------------------
    for _ in range(extra_overlap_subsets):
        subset = random.sample(all_nodes, subset_size)
        subsets.append(subset)

    # -------------------------
    # Sanity checks
    # -------------------------
    covered = set()
    for s in subsets:
        assert len(s) == subset_size
        covered.update(s)

    assert len(covered) == n_nodes, "Coverage violated"

    return subsets
