# src/semantic_compiler/sir/pragmatic_layer.py
from .layer import Layer
import torch

class PragmaticLayer(Layer):
    """Layer 2: Continuous pragmatic vectors (VSA)."""
    def __init__(self):
        super().__init__(id="layer2", data={})
        self.pragmatic_vectors: Dict[str, torch.Tensor] = {}
