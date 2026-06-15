# src/semantic_compiler/sir/explanation_layer.py
from .layer import Layer

class ExplanationLayer(Layer):
    """Layer 4: Reasoning trace mapping Layer 3 → Layer 1."""
    def __init__(self):
        super().__init__(id="layer4", data={})
        self.traces = []  # ABDUCED_FROM, INFERRED_VIA
