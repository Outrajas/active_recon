# src/semantic_compiler/sir/semantic_layer.py
from .layer import Layer

class SemanticLayer(Layer):
    """Layer 1: Explicit triples, entities, events, causal links."""
    def __init__(self):
        super().__init__(id="layer1", data={})
        self.triples = []  # (subject, predicate, object)
