# src/semantic_compiler/sir/interpretive_layer.py
from .layer import Layer

class InterpretiveLayer(Layer):
    """Layer 3: Candidate psychological/symbolic interpretations."""
    def __init__(self):
        super().__init__(id="layer3", data={})
        self.candidates = []  # Each with probability > 0.85
