# src/semantic_compiler/llm/evidence.py
from typing import List, Dict

class EvidenceChain:
    """Stores provenance for a single fact."""
    def __init__(self, fact_id: str):
        self.fact_id = fact_id
        self.mentions: List[Dict] = []  # each mention: {'chunk_id': str, 'sentence': str, 'offsets': [start, end]}

    def add_mention(self, chunk_id: str, sentence: str, offsets: List[int]):
        self.mentions.append({
            "chunk_id": chunk_id,
            "sentence": sentence,
            "offsets": offsets
        })

    def to_dict(self):
        return {
            "fact_id": self.fact_id,
            "mentions": self.mentions
        }
