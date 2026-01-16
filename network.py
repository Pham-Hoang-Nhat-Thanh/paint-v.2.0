class Node:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Node({self.name})"
    
    def __eq__(self, value):
        return isinstance(value, Node) and self.name == value.name
    
    def __hash__(self):
        return hash(self.name)


class GraphNetwork:
    def __init__(self):
        # direct adjacency only; `children`/`parents` removed
        self.adjacency = {}
        # Topological order (list) and position map for incremental maintenance
        self.topo_order = []
        self.position = {}

    def add_node(self, node):
        if node not in self.adjacency:
            self.adjacency[node] = set()
            # place new nodes at the end of the topological order
            self.position[node] = len(self.topo_order)
            self.topo_order.append(node)

    def is_valid_edge(self, parent, child):
        """
        Check if adding an edge from parent to child would create a cycle.
        
        :param parent: parent node
        :param child: child node
        """
        self.add_node(parent)
        self.add_node(child)

        if parent == child:
            return False
        if child in self.adjacency[parent]:
            return False # Already exists

        # quick check using topological order: if parent comes before child,
        # adding the edge cannot create a cycle. Otherwise check reachability
        # from child -> parent.
        if self.position[parent] < self.position[child]:
            return True

        # parent is after child in current order; adding edge may create a cycle
        return not self._reachable(child, parent)

    def add_edge(self, parent, child):
        """
        Add a directed edge from parent to child if it doesn't create a cycle.
        
        :param parent: parent node
        :param child: child node
        """
        # Ensure nodes exist
        self.add_node(parent)
        self.add_node(child)

        if parent == child:
            print("Adding self-edge is not allowed")
            return False

        if child in self.adjacency[parent]:
            return False

        # Fast check using topological positions
        if self.position[parent] < self.position[child]:
            # safe to add, preserves topo order
            self.adjacency[parent].add(child)
        else:
            # parent is after child: check if child reaches parent (would form cycle)
            if self._reachable(child, parent):
                print("Adding edge from {} to {} would create a cycle.".format(parent, child))
                return False

            # No cycle — we must move the affected nodes (those reachable from
            # child that are positioned <= parent) to just after parent.
            parent_pos = self.position[parent]
            # compute reachable set constrained by position
            affected = [n for n in self._reachable_nodes(child) if self.position.get(n, float('inf')) <= parent_pos]
            if affected:
                # preserve current relative order
                affected_set = set(affected)
                new_order = [n for n in self.topo_order if n not in affected_set]
                # find parent index in new_order
                insert_idx = new_order.index(parent) + 1
                # insert affected nodes after parent
                for i, n in enumerate(affected):
                    new_order.insert(insert_idx + i, n)
                # update topo_order and positions
                self.topo_order = new_order
                for idx, n in enumerate(self.topo_order):
                    self.position[n] = idx

            # finally add the edge
            self.adjacency[parent].add(child)

        return True

    def _reachable(self, start, target):
        """Return True if target reachable from start via `adjacency` (DFS)."""
        if start == target:
            return True
        visited = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for ch in self.adjacency.get(cur, ()):  # direct edges
                if ch == target:
                    return True
                if ch not in visited:
                    stack.append(ch)
        return False

    def _reachable_nodes(self, start):
        """Return list of nodes reachable from `start` in topological order."""
        visited = set()
        stack = [start]
        result = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            result.append(cur)
            for ch in self.adjacency.get(cur, ()):  # direct edges
                if ch not in visited:
                    stack.append(ch)
        # return nodes in the order they appear in topo_order
        result_in_order = [n for n in self.topo_order if n in visited]
        return result_in_order

    def add_edges(self, edges):
        """
        Add multiple edges to the graph.
        
        :param edges: list of (parent, child) tuples
        """
        for parent, child in edges:
            if not self.add_edge(parent, child):
                raise ValueError("Cannot add edge from {} to {} as it creates a cycle.".format(parent, child))
            
    def remove_edge(self, parent, child):
        """
        Remove a directed edge from parent to child.
        
        :param parent: parent node
        :param child: child node
        """
        if parent in self.adjacency and child in self.adjacency[parent]:
            self.adjacency[parent].remove(child)
            return True
        return False
    
    def remove_edges(self, edges):
        """
        Remove multiple edges from the graph.
        
        :param edges: list of (parent, child) tuples
        """
        for parent, child in edges:
            if not self.remove_edge(parent, child):
                raise ValueError("Edge from {} to {} does not exist.".format(parent, child))
