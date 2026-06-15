# src/semantic_compiler/pipeline/reconstruction.py
import networkx as nx

class ReconstructionStage:
    """Reassemble original text from source chunks in page+y order."""
    def run(self, graph: nx.DiGraph) -> str:
        chunks = []
        for n, d in graph.nodes(data=True):
            if d.get('type') == 'source_chunk':
                page = d.get('page', 1)
                y = d.get('y', 0)
                text = d.get('text', '')
                chunks.append((page, y, n, text))
        chunks.sort(key=lambda x: (x[0], x[1], x[2]))   # page, then y, then node ID for tie-breaking
        reconstructed = '\n\n'.join(text for _, _, _, text in chunks)
        return reconstructed
