# src/semantic_compiler/pipeline/consolidation.py
import networkx as nx, torch
from sentence_transformers import SentenceTransformer, util

class ConsolidationStage:
    def __init__(self, threshold=0.75, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name, device=device)

    def run(self, merged_graph: nx.DiGraph) -> nx.DiGraph:
        nodes = list(merged_graph.nodes())
        if len(nodes) < 2:
            return merged_graph
        # Separate source chunks – never merge them
        chunk_nodes = [n for n, d in merged_graph.nodes(data=True) if d.get('type') == 'source_chunk']
        other_nodes = [n for n in nodes if n not in chunk_nodes]
        # Use the actual text of the node, fallback to node ID
        texts = [merged_graph.nodes[n].get("text", str(n)) for n in other_nodes]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        sim = util.cos_sim(embeddings, embeddings)
        G = merged_graph.copy()
        merged_set = set()
        for i in range(len(other_nodes)):
            if other_nodes[i] in merged_set:
                continue
            for j in range(i+1, len(other_nodes)):
                if other_nodes[j] in merged_set:
                    continue
                if sim[i][j] > self.threshold:
                    self._merge_nodes(G, other_nodes[j], other_nodes[i])
                    merged_set.add(other_nodes[j])
        return G

    def _merge_nodes(self, G, old, new):
        if old == new or old not in G:
            return
        for u, v, data in list(G.edges(data=True)):
            if u == old:
                G.add_edge(new, v, **data)
                G.remove_edge(u, v)
            elif v == old:
                G.add_edge(u, new, **data)
                G.remove_edge(u, v)
        if old in G:
            G.remove_node(old)
