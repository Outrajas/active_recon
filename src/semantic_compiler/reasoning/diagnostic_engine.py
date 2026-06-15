# src/semantic_compiler/reasoning/diagnostic_engine.py
import networkx as nx

class DiagnosticEngine:
    """Stage 7 core: basic structural reasoning with UNVERIFIED_ADJACENCY and TAINTED flags."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.unverified_edges = []   # list of (u, v)

    def execute(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Scans for nodes that are poorly connected to the rest of the graph,
        adding UNVERIFIED_ADJACENCY edges and marking downstream nodes TAINTED.
        """
        # Simple heuristic: if a node has no incoming edges and no outgoing edges,
        # connect it to the most central node with an UNVERIFIED_ADJACENCY edge.
        for node in list(graph.nodes()):
            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0:
                # Find central node (highest degree)
                central_node = max(graph.nodes(), key=lambda n: graph.degree(n))
                if node != central_node and not graph.has_edge(node, central_node):
                    graph.add_edge(node, central_node, label="UNVERIFIED_ADJACENCY", tainted=True)
                    self.unverified_edges.append((node, central_node))

        # Tag downstream nodes of unverified edges as TAINTED
        for u, v in self.unverified_edges:
            # Mark target and all its descendants
            self._propagate_tainted(graph, v)

        return graph

    def _propagate_tainted(self, graph, node, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)
        if 'tainted' not in graph.nodes[node]:
            graph.nodes[node]['tainted'] = True
        for succ in graph.successors(node):
            self._propagate_tainted(graph, succ, visited)
