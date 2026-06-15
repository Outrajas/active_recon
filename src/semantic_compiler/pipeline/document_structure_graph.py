# src/semantic_compiler/pipeline/document_structure_graph.py
import networkx as nx

class DocumentStructureGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.insertion_order = []   # record order of added leaf blocks

    def build(self, layout_blocks: list[dict]) -> nx.DiGraph:
        self.graph.add_node("DOCUMENT", type="DOCUMENT", text="")
        current_section = "DOCUMENT"
        prev_node = None

        for block in layout_blocks:
            role = block['document_role']
            chunk_id = block['source_chunk']
            text = block['text']
            node_type = self._map_role_to_type(role)
            self.graph.add_node(chunk_id,
                                type=node_type,
                                text=text,
                                page=block['page'],
                                y=block['y'],
                                document_role=role)
            self.insertion_order.append(chunk_id)   # record insertion order

            if role == "SECTION_HEADER" or role == "TITLE":
                section_id = f"SECTION_{chunk_id}"
                self.graph.add_node(section_id, type="SECTION", text=text)
                self.graph.add_edge("DOCUMENT", section_id, rel="CONTAINS")
                self.graph.add_edge(section_id, chunk_id, rel="CONTAINS")
                current_section = section_id
                prev_node = chunk_id
                continue

            self.graph.add_edge(current_section, chunk_id, rel="CONTAINS")
            if prev_node is not None:
                self.graph.add_edge(prev_node, chunk_id, rel="PRECEDES")
            prev_node = chunk_id

        print(f"[DSG] Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges", file=__import__('sys').stderr)
        return self.graph

    def _map_role_to_type(self, role: str) -> str:
        mapping = {
            "TITLE": "TITLE", "ABSTRACT": "ABSTRACT", "PARAGRAPH": "PARAGRAPH",
            "SECTION_HEADER": "SECTION_HEADER", "CAPTION": "CAPTION", "TABLE": "TABLE",
            "EQUATION": "EQUATION", "REFERENCE": "REFERENCE", "FOOTNOTE": "FOOTNOTE",
            "AUTHOR_BLOCK": "AUTHOR_BLOCK", "PAGE_NUMBER": "PAGE_NUMBER",
        }
        return mapping.get(role, "UNKNOWN")

    def get_ordered_text(self) -> str:
        """Return text in insertion order (page reading order)."""
        ordered = []
        for node_id in self.insertion_order:
            if node_id in self.graph:
                ordered.append(self.graph.nodes[node_id].get('text', ''))
        return '\n\n'.join(ordered)
