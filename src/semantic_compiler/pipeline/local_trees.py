# src/semantic_compiler/pipeline/local_trees.py
import networkx as nx
from semantic_compiler.llm.canonicalize import is_valid_entity, canonical_name
from semantic_compiler.llm.fact_metric_extractor import FactMetricExtractor
from semantic_compiler.llm.concept_validator import ConceptValidator

class LocalTreesStage:
    def __init__(self):
        self.fact_metric_extractor = FactMetricExtractor()
        self.validator = ConceptValidator()

    def run(self, extraction_results: dict, chunk_texts: dict = {}, chunk_attrs: dict = {}) -> dict:
        extraction_results = self.fact_metric_extractor.enrich(extraction_results)
        local_graphs = {}
        for chunk_id, objects in extraction_results.items():
            G = nx.DiGraph()
            chunk_node = f"source_{chunk_id}"
            original_text = chunk_texts.get(chunk_id, "")
            attrs = dict(chunk_attrs.get(chunk_id, {}))
            attrs.setdefault('section', 'Unknown')
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
                concepts = obj.get("concepts", [])
                facts = obj.get("facts", [])
                metrics = obj.get("metrics", [])

                sem_node_id = f"{sem_role}_{chunk_id}_{len(G.nodes)}"
                G.add_node(sem_node_id,
                           semantic_role=sem_role,
                           document_role=doc_role,
                           text=text,
                           confidence=confidence,
                           source_chunk=chunk_id,
                           section=attrs['section'])
                G.add_edge(chunk_node, sem_node_id, label="EXTRACTED_FROM", confidence=confidence)

                if doc_role != "MAIN_CONTENT":
                    continue

                # Add concept nodes
                for concept_name in concepts:
                    if not self.validator._is_valid(concept_name):
                        continue
                    canonical = canonical_name(concept_name)
                    if not is_valid_entity(canonical):
                        continue
                    ent_node = f"entity_{canonical}"
                    if ent_node not in G:
                        G.add_node(ent_node, type="ENTITY", text=canonical)
                    G.add_edge(sem_node_id, ent_node, label="MENTIONS", confidence=confidence)

                # --- DIAGNOSTIC: raw facts ---
                print(f"[RAW] Chunk {chunk_id}: {len(facts)} facts: {facts}")

                # Process relation-based facts
                for fact in facts:
                    # --- DIAGNOSTIC: processing fact ---
                    print(f"[PROCESS] Fact: {fact}")

                    subj = fact['subject']
                    objt = fact['object']
                    rel = fact['relation']
                    evidence = fact.get('evidence', text)

                    # Validate with reason logging
                    subj_valid, subj_reason = self.validator._is_valid_with_reason(subj)
                    objt_valid, objt_reason = self.validator._is_valid_with_reason(objt)
                    if not subj_valid:
                        print(f"[REJECTED] subject '{subj}' invalid: {subj_reason}")
                    if not objt_valid:
                        print(f"[REJECTED] object '{objt}' invalid: {objt_reason}")
                    if not subj_valid or not objt_valid:
                        print(f"[DROPPED] Fact discarded: {fact}")
                        continue

                    subj_canon = canonical_name(subj)
                    objt_canon = canonical_name(objt)
                    if not is_valid_entity(subj_canon):
                        print(f"[REJECTED] subj_canon '{subj_canon}' not a valid entity")
                        continue
                    if not is_valid_entity(objt_canon):
                        print(f"[REJECTED] objt_canon '{objt_canon}' not a valid entity")
                        continue

                    fact_id = f"FACT_{chunk_id}_{len(G.nodes)}"
                    G.add_node(fact_id, type="FACT",
                               subject=subj_canon, relation=rel, object=objt_canon,
                               confidence=fact.get('confidence', 0.85), source_chunk=chunk_id, section=attrs['section'],
                               evidence=evidence)
                    if is_valid_entity(subj_canon):
                        subj_node = f"entity_{subj_canon}"
                        if subj_node not in G: G.add_node(subj_node, type="ENTITY", text=subj_canon)
                        G.add_edge(fact_id, subj_node, label="HAS_SUBJECT", confidence=fact['confidence'])
                    if is_valid_entity(objt_canon):
                        objt_node = f"entity_{objt_canon}"
                        if objt_node not in G: G.add_node(objt_node, type="ENTITY", text=objt_canon)
                        G.add_edge(fact_id, objt_node, label="HAS_OBJECT", confidence=fact['confidence'])
                    G.add_edge(chunk_node, fact_id, label="CONTAINS_FACT", confidence=fact['confidence'])

                # Metrics as before
                for metric in metrics:
                    metric_id = f"METRIC_{chunk_id}_{len(G.nodes)}"
                    G.add_node(metric_id, type="METRIC",
                               metric=metric['metric'],
                               value=metric['value'],
                               dataset=metric['dataset'],
                               confidence=metric['confidence'],
                               source_chunk=chunk_id,
                               section=attrs['section'])
                    G.add_edge(chunk_node, metric_id, label="CONTAINS_METRIC", confidence=metric['confidence'])
                    metric_entity = metric['metric']
                    if self.validator._is_valid(metric_entity):
                        ent_node = f"entity_{metric_entity}"
                        if ent_node not in G: G.add_node(ent_node, type="ENTITY", text=metric_entity)
                        G.add_edge(metric_id, ent_node, label="MEASURES", confidence=metric['confidence'])

            # --- DIAGNOSTIC: final counts ---
            total_extracted = sum(len(obj.get('facts', [])) for obj in objects)
            fact_nodes = sum(1 for _, d in G.nodes(data=True) if d.get('type') == 'FACT')
            print(f"[DIAG] Chunk {chunk_id}: facts_extracted={total_extracted}, fact_nodes_created={fact_nodes}")

            local_graphs[chunk_id] = G
        return local_graphs
