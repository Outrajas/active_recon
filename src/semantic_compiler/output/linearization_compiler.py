# src/semantic_compiler/output/linearization_compiler.py
import networkx as nx
from collections import defaultdict

class LinearizationCompiler:
    def compile(self, graph: nx.DiGraph) -> str:
        lines = []
        lines.append("=== Semantic Compiler Entity-Centric Output ===")
        lines.append(f"Total nodes: {graph.number_of_nodes()}")
        lines.append(f"Total edges: {graph.number_of_edges()}")
        lines.append("")

        # Gather both ENTITY and CONCEPT nodes
        display_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') in ('ENTITY', 'CONCEPT')]
        if not display_nodes:
            lines.append("No entities/concepts found.")
            return "\n".join(lines)

        for ent in display_nodes:
            ent_text = graph.nodes[ent].get('text', ent)
            ent_type = graph.nodes[ent].get('type', 'ENTITY')
            lines.append(f"## {ent_type}: {ent_text}")

            related_objects = set()
            for u, v, d in graph.edges(data=True):
                if v == ent and d.get('label') == 'MENTIONS':
                    related_objects.add(u)

            by_role = defaultdict(list)
            for obj_id in related_objects:
                node_data = graph.nodes[obj_id]
                if node_data.get('type') == 'FACT':
                    continue
                role = node_data.get('semantic_role', 'UNKNOWN')
                text = node_data.get('text', '')
                doc_role = node_data.get('document_role', '')
                chunk = node_data.get('source_chunk', '')
                confidence = node_data.get('confidence', 1.0)
                by_role[role].append((text, doc_role, chunk, confidence))

            for role, items in sorted(by_role.items()):
                lines.append(f"  {role}:")
                for text, doc_role, chunk, conf in items:
                    doc_str = f" [{doc_role}]" if doc_role not in ("", "MAIN_CONTENT") else ""
                    lines.append(f"    - {text} (chunk {chunk}, conf={conf:.2f}){doc_str}")

            outgoing = []
            for u, v, d in graph.edges(data=True):
                if u == ent and d.get('label') not in ('MENTIONS', 'EXTRACTED_FROM', 'HAS_SUBJECT', 'HAS_OBJECT'):
                    outgoing.append((v, d['label']))
            if outgoing:
                lines.append(f"  RELATIONS:")
                for target, rel in outgoing:
                    target_text = graph.nodes[target].get('text', target)
                    lines.append(f"    -> {target_text} : {rel}")

            lines.append("")

        unverified = [(u, v) for u, v, d in graph.edges(data=True) if d.get("label") == "UNVERIFIED_ADJACENCY"]
        if unverified:
            lines.append("## UNVERIFIED ADJACENCIES")
            for u, v in unverified:
                lines.append(f"  {u} -- UNVERIFIED --> {v}")

        lines.append("=== End of Entity-Centric Output ===")
        return "\n".join(lines)
