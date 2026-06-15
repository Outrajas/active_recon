# src/semantic_compiler/pipeline/extraction.py
from semantic_compiler.llm.extractor import ExtractorLLM
from semantic_compiler.utils.vsa import VSAStore

class ExtractionStage:
    def __init__(self, llm: ExtractorLLM, vsa: VSAStore):
        self.llm = llm
        self.vsa = vsa

    def run(self, chunks: dict) -> dict:
        """chunks: dict {chunk_id: text} -> results dict {chunk_id: list of objects}"""
        results = {}
        for chunk_id, chunk_text in chunks.items():
            objects = self.llm.extract(chunk_text, f"chunk_{chunk_id}")
            results[chunk_id] = objects
            # Store VSA embedding
            prag_vec = self.llm.compute_pragmatic_vector(chunk_text)
            self.vsa.store(f"chunk_{chunk_id}", prag_vec)
        return results
