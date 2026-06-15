# src/semantic_compiler/llm/extractor.py
from sentence_transformers import SentenceTransformer
from .semantic_objects import SemanticObjectExtractor
from typing import List, Dict

class ExtractorLLM:
    """Extracts typed semantic objects from chunks, plus VSA embeddings."""
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        self.obj_extractor = SemanticObjectExtractor()
        self.pragmatic_encoder = SentenceTransformer(model_name, device=device)

    def extract(self, text: str, chunk_id: str) -> List[Dict]:
        """Return list of semantic objects and compute pragmatic vector."""
        objects = self.obj_extractor.extract(text, chunk_id)
        return objects

    def compute_pragmatic_vector(self, text: str):
        return self.pragmatic_encoder.encode(text, convert_to_tensor=True)
