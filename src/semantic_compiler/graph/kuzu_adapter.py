# src/semantic_compiler/graph/kuzu_adapter.py
import json

class KuzuAdapter:
    """Manages the embedded graph database (currently JSON, swap to Kùzu)."""
    def __init__(self, db_path: str):
        self.db_path = db_path

    def commit_graph(self, nodes, edges):
        data = {"nodes": nodes, "edges": edges}
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
