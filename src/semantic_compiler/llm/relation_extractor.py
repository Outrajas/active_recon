# src/semantic_compiler/llm/relation_extractor.py
import re
import spacy
from .concept_extractor import ConceptExtractor

RELATION_KEYWORDS = [
    (r'\b(uses|utilizes|employs|relies on|leverages|is based on|adopts)\b', 'USES'),
    (r'\b(replaces|substitutes|removes|eliminates)\b', 'REPLACES'),
    (r'\b(enables|allows|permits)\b', 'ENABLES'),
    (r'\b(improves|enhances|boosts|increases)\b', 'IMPROVES'),
    (r'\b(outperforms|beats|surpasses|exceeds|performs better than)\b', 'OUTPERFORMS'),
    (r'\b(achieves|obtains|reaches|yields|scores)\b', 'ACHIEVES'),
    (r'\b(consists of|contains|comprises|includes|is composed of)\b', 'CONTAINS'),
    (r'\b(causes|leads to|results in|contributes to)\b', 'CAUSES'),
    (r'\b(follows|adopts)\b', 'FOLLOWS'),
    (r'\b(depends on|relies on)\b', 'DEPENDS_ON'),
]

class RelationExtractor:
    def __init__(self):
        self.concept_extractor = ConceptExtractor()
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, sentence: str) -> list:
        """
        Finds concepts, detects relation keywords, and uses dependency parse
        to assign subject/object direction.
        """
        concepts = self.concept_extractor.extract_concepts(sentence)
        if len(concepts) < 2:
            return []

        # Parse the sentence once
        doc = self.nlp(sentence)
        facts = []

        # For each relation keyword pattern, find matches
        for pattern, rel_label in RELATION_KEYWORDS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                keyword_start = match.start()
                keyword_end = match.end()

                # Find the concept that is closest to the keyword as subject (usually before)
                # and the concept that is closest after as object (simplified heuristic)
                subject_concept = None
                object_concept = None
                best_dist = float('inf')

                for token in doc:
                    if token.dep_ == "ROOT" and token.pos_ == "VERB":
                        # Get subject (nsubj)
                        subj_tok = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
                        if subj_tok:
                            subj_text = subj_tok.text.lower()
                            # Find if any concept contains this subject token
                            for concept in concepts:
                                if subj_text in concept.split():
                                    subject_concept = concept
                                    break
                        # Get object (dobj, attr, pobj)
                        obj_tok = next((c for c in token.children if c.dep_ in ("dobj", "attr", "pobj")), None)
                        if obj_tok:
                            obj_text = obj_tok.text.lower()
                            for concept in concepts:
                                if obj_text in concept.split():
                                    object_concept = concept
                                    break

                # Fallback: use positional heuristic (concept before keyword vs after)
                if not subject_concept or not object_concept:
                    for concept in concepts:
                        # Find position of concept in sentence
                        pos = sentence.lower().find(concept)
                        if pos == -1:
                            continue
                        if pos < keyword_start:
                            if not subject_concept:
                                subject_concept = concept
                        elif pos > keyword_end:
                            if not object_concept:
                                object_concept = concept

                if subject_concept and object_concept and subject_concept != object_concept:
                    facts.append({
                        'subject': subject_concept,
                        'relation': rel_label,
                        'object': object_concept,
                        'confidence': 0.8,
                        'evidence': sentence
                    })
                    break  # only one fact per keyword match

        return facts
