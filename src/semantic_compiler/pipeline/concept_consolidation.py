# src/semantic_compiler/pipeline/concept_consolidation.py
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

class ConceptConsolidationStage:
    """Merges similar ENTITY nodes into canonical CONCEPT nodes."""
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.8, device="cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = threshold

    def run(self, graph: nx.DiGraph) -> nx.DiGraph:
        entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'ENTITY']
        if len(entity_nodes) < 2:
            return graph

        texts = [graph.nodes[n].get('text', n) for n in entity_nodes]
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        sim = util.cos_sim(embeddings, embeddings)

        merged = set()
        concept_map = {}
        concept_texts = {}

        for i in range(len(entity_nodes)):
            if entity_nodes[i] in merged:
                continue
            cluster = [entity_nodes[i]]
            for j in range(i+1, len(entity_nodes)):
                if entity_nodes[j] in merged:
                    continue
                if sim[i][j] > self.threshold:
                    cluster.append(entity_nodes[j])
                    merged.add(entity_nodes[j])
            merged.add(entity_nodes[i])
            best_node = max(cluster, key=lambda n: len(graph.nodes[n].get('text', '')))
            concept_id = f"CONCEPT_{best_node}"
            concept_texts[concept_id] = graph.nodes[best_node].get('text', best_node)
            for node in cluster:
                concept_map[node] = concept_id

        # Build new graph
        G = nx.DiGraph()
        # Copy all non-entity nodes
        for n, d in graph.nodes(data=True):
            if d.get('type') != 'ENTITY':
                G.add_node(n, **d)
        # Add concept nodes
        for cid, text in concept_texts.items():
            G.add_node(cid, type="CONCEPT", text=text)

        # Rebuild edges: replace entity nodes with concept nodes
        for u, v, d in graph.edges(data=True):
            u_new = concept_map.get(u, u)
            v_new = concept_map.get(v, v)
            if u_new != u or v_new != v:
                if not G.has_edge(u_new, v_new):
                    G.add_edge(u_new, v_new, **d)
                else:
                    existing = G[u_new][v_new]
                    if 'chunks' in d and 'chunks' in existing:
                        existing['chunks'] = list(set(existing['chunks'] + d['chunks']))
            else:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, **d)

        # Remove all old entity nodes that were mapped to concepts (keep only if unmapped)
        for old_entity in concept_map:
            if old_entity in G:
                G.remove_node(old_entity)

        return G
