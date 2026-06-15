# src/semantic_compiler/pipeline/local_trees.py
import networkx as nx
from semantic_compiler.llm.canonicalize import is_valid_entity, canonical_name

class LocalTreesStage:
    def run(self, extraction_results: dict, chunk_texts: dict = {}, chunk_attrs: dict = {}) -> dict:
        local_graphs = {}
        for chunk_id, objects in extraction_results.items():
            G = nx.DiGraph()
            chunk_node = f"source_{chunk_id}"
            original_text = chunk_texts.get(chunk_id, "")
            attrs = chunk_attrs.get(chunk_id, {})
            G.add_node(chunk_node,
                       type="source_chunk",
                       chunk_id=chunk_id,
                       text=original_text,
                       **attrs)

            for obj in objects:
                sem_role = obj.get("semantic_role", "UNKNOWN")
                doc_role = obj.get("document_role", "MAIN_CONTENT")
                text = obj.get("text", "")
                confidence = obj.get("confidence", 1.0)
                entities = obj.get("entities", [])
                triples = obj.get("triples", [])

                # Semantic object node (with role)
                sem_node_id = f"{sem_role}_{chunk_id}_{len(G.nodes)}"
                G.add_node(sem_node_id,
                           semantic_role=sem_role,
                           document_role=doc_role,
                           text=text,
                           confidence=confidence,
                           source_chunk=chunk_id)
                G.add_edge(chunk_node, sem_node_id, label="EXTRACTED_FROM", confidence=confidence)

                # Only main content gets entities and triples
                if doc_role != "MAIN_CONTENT":
                    continue

                for ent_name in entities:
                    canonical = canonical_name(ent_name)
                    if not is_valid_entity(canonical):
                        continue
                    ent_node = f"entity_{canonical}"
                    if ent_node not in G:
                        G.add_node(ent_node, type="ENTITY", text=canonical)
                    G.add_edge(sem_node_id, ent_node, label="MENTIONS", confidence=confidence)

                for subj, pred, objt in triples:
                    subj_canon = canonical_name(subj.lower())
                    objt_canon = canonical_name(objt.lower())
                    if not is_valid_entity(subj_canon) or not is_valid_entity(objt_canon):
                        continue
                    subj_node = f"entity_{subj_canon}"
                    objt_node = f"entity_{objt_canon}"
                    if subj_node not in G:
                        G.add_node(subj_node, type="ENTITY", text=subj_canon)
                    if objt_node not in G:
                        G.add_node(objt_node, type="ENTITY", text=objt_canon)
                    G.add_edge(subj_node, objt_node, label=pred.upper(), confidence=confidence)

            local_graphs[chunk_id] = G
        return local_graphs
