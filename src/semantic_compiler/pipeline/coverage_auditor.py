# src/semantic_compiler/pipeline/coverage_auditor.py
import networkx as nx
from collections import defaultdict

class CoverageAuditor:
    def audit(self, graph: nx.DiGraph) -> dict:
        source_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'source_chunk']
        total_chunks = len(source_nodes)
        if total_chunks == 0:
            return {"error": "No source chunks found"}

        covered_chunks = set()
        total_text_len = 0
        covered_text_len = 0
        chunk_details = {}

        for node in source_nodes:
            node_data = graph.nodes[node]
            chunk_id = node_data.get('chunk_id', node)
            text = node_data.get('text', '')
            text_len = len(text)
            total_text_len += text_len
            sem_objects = []
            for _, v, d in graph.out_edges(node, data=True):
                if d.get('label') == 'EXTRACTED_FROM':
                    sem_objects.append(v)
            if sem_objects:
                covered_chunks.add(chunk_id)
                covered = sum(len(graph.nodes[o].get('text', '')) for o in sem_objects)
                covered_text_len += covered
            else:
                covered = 0
            chunk_details[chunk_id] = {
                'text_len': text_len,
                'covered': len(sem_objects) > 0,
                'semantic_objects_count': len(sem_objects),
                'covered_chars': covered
            }

        # Collect all semantic objects (non-source, non-entity)
        sem_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') != 'source_chunk' and d.get('type') != 'ENTITY']
        role_counts = defaultdict(int)
        role_examples = defaultdict(list)
        for n in sem_nodes:
            role = graph.nodes[n].get('semantic_role', 'UNKNOWN')
            role_counts[role] += 1
            if len(role_examples[role]) < 20:
                role_examples[role].append(graph.nodes[n].get('text', '')[:200])

        return {
            "total_chunks": total_chunks,
            "covered_chunks": len(covered_chunks),
            "coverage_ratio": len(covered_chunks) / total_chunks if total_chunks else 0,
            "total_text_chars": total_text_len,
            "covered_text_chars_approx": covered_text_len,
            "text_coverage_ratio": covered_text_len / total_text_len if total_text_len else 0,
            "uncovered_chunks": [c for c in chunk_details if not chunk_details[c]['covered']],
            "role_distribution": dict(role_counts),
            "role_examples": {role: examples for role, examples in role_examples.items()}
        }
