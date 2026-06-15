import torch

class VSAStore:
    """Container for Vector-Symbolic Architecture tensors (Layer 2)."""
    def __init__(self):
        self.embeddings = {}  # chunk_id -> tensor

    def store(self, chunk_id: str, tensor: torch.Tensor):
        self.embeddings[chunk_id] = tensor

    def retrieve(self, chunk_id: str) -> torch.Tensor:
        return self.embeddings.get(chunk_id)
