# src/semantic_compiler/llm/method_extractor.py
import spacy
from .concept_extractor import ConceptExtractor

class MethodRelationExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.concept_extractor = ConceptExtractor()
        self.verb_map = {
            "use": "USES",
            "utilize": "USES",
            "employ": "USES",
            "leverage": "USES",
            "apply": "APPLIES",
            "train": "TRAINS_WITH",
            "optimize": "OPTIMIZES_WITH",
        }

    def extract(self, sentence: str, chunk_id: str = "", page: int = 0, section: str = "Unknown") -> list:
        doc = self.nlp(sentence)
        facts = []
        concepts = self.concept_extractor.extract_concepts(sentence)
        def find_concept(text: str):
            text_lower = text.lower().strip()
            for c in concepts:
                if c.lower() == text_lower:
                    return c
            for c in concepts:
                if text_lower in c.lower() or c.lower() in text_lower:
                    return c
            return text_lower

        for token in doc:
            if token.pos_ != "VERB" or token.lemma_ not in self.verb_map:
                continue
            subj_tok = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj_tok = child
                    break
            if not subj_tok:
                continue
            obj_tok = None
            for child in token.children:
                if child.dep_ in ("dobj", "attr", "pobj"):
                    obj_tok = child
                    break
            if not obj_tok:
                continue
            subj_span = subj_tok.text
            for child in subj_tok.children:
                if child.dep_ in ("compound", "amod"):
                    subj_span = child.text + " " + subj_span
            obj_span = obj_tok.text
            for child in obj_tok.children:
                if child.dep_ in ("compound", "amod"):
                    obj_span = child.text + " " + obj_span
            subj_concept = find_concept(subj_span)
            obj_concept = find_concept(obj_span)
            if not subj_concept or not obj_concept or subj_concept == obj_concept:
                continue
            start = sentence.lower().find(subj_span.lower()) if subj_span else 0
            end = sentence.lower().rfind(obj_span.lower()) + len(obj_span) if obj_span else 0
            facts.append({
                "subject": subj_concept,
                "relation": self.verb_map[token.lemma_],
                "object": obj_concept,
                "confidence": 0.85,
                "evidence": sentence,
                "chunk_id": chunk_id,
                "page": page,
                "section": section,
                "start_offset": start,
                "end_offset": end,
                "fact_type": "METHOD"
            })
        return facts
