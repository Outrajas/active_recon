# src/semantic_compiler/llm/numerical_extractor.py
import re
from .concept_extractor import ConceptExtractor

class NumericalRelationExtractor:
    def __init__(self):
        self.concept_extractor = ConceptExtractor()
        self.patterns = [
            (r'\b([A-Za-z_]+)\s*=\s*(\d+)\s*', 'HAS_ATTRIBUTE'),
            (r'(\d+)\s*(GPUs|gpus|GPU|gpu)\b', 'USES_GPU_COUNT'),
            (r'(\d+)\s*(warmup steps|warmup_steps|training steps|epochs|batch size|dimension|layers)\b', 'HAS_ATTRIBUTE'),
            (r'\b(d_model|embedding dimension|hidden size|model dimension)\s*(?:of\s*)?(\d+)\b', 'HAS_DIMENSION'),
        ]

    def extract(self, sentence: str, chunk_id: str = "", page: int = 0, section: str = "Unknown") -> list:
        facts = []
        concepts = self.concept_extractor.extract_concepts(sentence)
        for pattern, relation in self.patterns:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                groups = match.groups()
                if len(groups) == 2:
                    subject, value = groups[0].strip(), groups[1].strip()
                    # Try to map subject to a concept
                    subj_concept = None
                    for c in concepts:
                        if c.lower() in subject.lower() or subject.lower() in c.lower():
                            subj_concept = c
                            break
                    if not subj_concept:
                        subj_concept = subject
                    facts.append({
                        "subject": subj_concept,
                        "relation": relation,
                        "object": value,
                        "confidence": 0.9,
                        "evidence": sentence,
                        "chunk_id": chunk_id,
                        "page": page,
                        "section": section,
                        "start_offset": match.start(),
                        "end_offset": match.end(),
                        "fact_type": "NUMERICAL"
                    })
        return facts
