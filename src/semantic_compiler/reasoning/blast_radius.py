# src/semantic_compiler/reasoning/blast_radius.py

def compute_blast_radius(graph):
    """Return the fraction of nodes flagged as TAINTED."""
    total = graph.number_of_nodes()
    if total == 0:
        return 0.0
    tainted = sum(1 for _, d in graph.nodes(data=True) if d.get('tainted'))
    return tainted / total
