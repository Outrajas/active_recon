# src/semantic_compiler/pipeline/global_graph.py
import json
import networkx as nx

class GlobalGraphStage:
    """Stage 5: Commit the merged graph to a persistent store (JSON file)."""
    def __init__(self, output_path: str = "data/global_graph.json"):
        self.output_path = output_path

    def run(self, merged_graph: nx.DiGraph):
        data = {
            "nodes": list(merged_graph.nodes(data=True)),
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in merged_graph.edges(data=True)
            ]
        }
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Global graph saved to {self.output_path}")
