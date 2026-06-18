# src/semantic_compiler/llm/definition_extractor.py
import re
from .concept_extractor import ConceptExtractor

class DefinitionExtractor:
    def __init__(self):
        self.concept_extractor = ConceptExtractor()
        self.patterns = [
            (r'(\b[A-Za-z][\w\s-]+?)\s+(is|are)\s+defined\s+as\s+([\w\s-]+?)\.', 'IS_DEFINED_AS'),
            (r'(\b[A-Za-z][\w\s-]+?)\s+refers\s+to\s+([\w\s-]+?)\.', 'IS_DEFINED_AS'),
            (r'(\b[A-Za-z][\w\s-]+?)\s+is\s+an?\s+([\w\s-]+?)\s+that\s+', 'IS_DEFINED_AS'),
            (r'(\b[A-Za-z][\w\s-]+?)\s+is\s+called\s+([\w\s-]+?)\.', 'IS_DEFINED_AS'),
        ]

    def extract(self, sentence: str, chunk_id: str = "", page: int = 0, section: str = "Unknown") -> list:
        facts = []
        concepts = self.concept_extractor.extract_concepts(sentence)
        for pattern, relation in self.patterns:
            for match in re.finditer(pattern, sentence, re.IGNORECASE | re.DOTALL):
                groups = match.groups()
                if len(groups) >= 2:
                    subj, obj = groups[0].strip(), groups[-1].strip()
                    subj_concept = None
                    for c in concepts:
                        if c.lower() in subj.lower() or subj.lower() in c.lower():
                            subj_concept = c
                            break
                    if not subj_concept:
                        subj_concept = subj
                    obj_concept = None
                    for c in concepts:
                        if c.lower() in obj.lower() or obj.lower() in c.lower():
                            obj_concept = c
                            break
                    if not obj_concept:
                        obj_concept = obj
                    start = sentence.lower().find(subj.lower()) if subj else 0
                    end = sentence.lower().rfind(obj.lower()) + len(obj) if obj else 0
                    facts.append({
                        "subject": subj_concept,
                        "relation": relation,
                        "object": obj_concept,
                        "confidence": 0.85,
                        "evidence": sentence,
                        "chunk_id": chunk_id,
                        "page": page,
                        "section": section,
                        "start_offset": start,
                        "end_offset": end,
                        "fact_type": "DEFINITION"
                    })
        return facts
