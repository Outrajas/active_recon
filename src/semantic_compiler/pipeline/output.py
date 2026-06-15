# src/semantic_compiler/pipeline/output.py
import networkx as nx
from collections import defaultdict

class OutputStage:
    def run_pseudo_language(self, graph: nx.DiGraph) -> str:
        lines = []
        lines.append("=== Semantic Compiler Fact‑Centric Output ===")
        lines.append(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        lines.append("")

        section_map = defaultdict(lambda: {"facts": [], "metrics": [], "entities": set()})
        for n, d in graph.nodes(data=True):
            if d.get('type') == 'FACT':
                section = d.get('section', 'Unknown')
                section_map[section]['facts'].append(d)
                for _, v, ed in graph.out_edges(n, data=True):
                    if ed.get('label') in ('HAS_SUBJECT', 'HAS_OBJECT'):
                        ent_text = graph.nodes[v].get('text', v)
                        section_map[section]['entities'].add(ent_text)
            elif d.get('type') == 'METRIC':
                section = d.get('section', 'Unknown')
                section_map[section]['metrics'].append(d)

        sorted_sections = sorted(section_map.keys(), key=lambda s: (s != 'Unknown', s))

        for section in sorted_sections:
            data = section_map[section]
            lines.append(f"## Section: {section}")
            if data['entities']:
                lines.append(f"  Entities: {', '.join(sorted(data['entities'])[:10])}")
            if data['facts']:
                lines.append("  Facts:")
                for fact in data['facts'][:15]:
                    lines.append(f"    ({fact['subject']}) --[{fact['relation']}]--> ({fact['object']})")
            if data['metrics']:
                lines.append("  Metrics:")
                for metric in data['metrics']:
                    ds = f" on {metric['dataset']}" if metric['dataset'] != 'unknown' else ""
                    lines.append(f"    {metric['value']:.1f} {metric['metric']}{ds}")
            lines.append("")

        entity_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'ENTITY']
        if entity_nodes:
            lines.append("## Top Entities:")
            for ent in sorted(entity_nodes, key=lambda n: graph.degree(n), reverse=True)[:20]:
                lines.append(f"  {graph.nodes[ent].get('text', ent)}")
        lines.append("")
        return "\n".join(lines)

    def run_latent_injection(self, graph: nx.DiGraph) -> 'torch.Tensor':
        import torch
        return torch.randn(1, 768)
