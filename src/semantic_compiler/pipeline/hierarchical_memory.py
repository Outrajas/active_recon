# src/semantic_compiler/pipeline/hierarchical_memory.py
import networkx as nx

class HierarchicalMemoryStage:
    """Stage 4: Merge local trees into a document-level AST (merged DiGraph)."""

    def run(self, local_graphs: dict) -> nx.DiGraph:
        """
        local_graphs: {chunk_id: nx.DiGraph}
        Returns: a single nx.DiGraph combining all chunks, with edge attribute 'chunks' for provenance.
        """
        merged = nx.DiGraph()
        for chunk_id, g in local_graphs.items():
            # Add nodes
            for node, data in g.nodes(data=True):
                if node not in merged:
                    merged.add_node(node, **data)
            # Add edges with chunk provenance
            for u, v, data in g.edges(data=True):
                if merged.has_edge(u, v):
                    # Edge already exists; append chunk_id to list of chunks
                    if 'chunks' in merged[u][v]:
                        merged[u][v]['chunks'].append(chunk_id)
                    else:
                        merged[u][v]['chunks'] = [chunk_id]
                else:
                    merged.add_edge(u, v, label=data.get('label', ''), chunks=[chunk_id])
        return merged
