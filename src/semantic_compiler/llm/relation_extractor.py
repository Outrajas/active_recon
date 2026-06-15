# src/semantic_compiler/llm/relation_extractor.py
import re
import spacy
from .concept_extractor import ConceptExtractor

# Mini-ontology: map verb lemmas to canonical relation label
VERB_TO_RELATION = {
    "use": "USES",
    "utilize": "USES",
    "employ": "USES",
    "rely": "USES",          # relies on
    "leverage": "USES",
    "base": "USES",          # is based on
    "replace": "REPLACES",
    "substitute": "REPLACES",
    "remove": "REPLACES",
    "eliminate": "REPLACES",
    "enable": "ENABLES",
    "allow": "ENABLES",
    "permit": "ENABLES",
    "improve": "IMPROVES",
    "enhance": "IMPROVES",
    "boost": "IMPROVES",
    "increase": "IMPROVES",
    "outperform": "OUTPERFORMS",
    "beat": "OUTPERFORMS",
    "surpass": "OUTPERFORMS",
    "exceed": "OUTPERFORMS",
    "achieve": "ACHIEVES",
    "obtain": "ACHIEVES",
    "reach": "ACHIEVES",
    "yield": "ACHIEVES",
    "score": "ACHIEVES",
    "consist": "CONTAINS",
    "contain": "CONTAINS",
    "comprise": "CONTAINS",
    "include": "CONTAINS",
    "cause": "CAUSES",
    "lead": "CAUSES",        # leads to
    "result": "CAUSES",      # results in
    "contribute": "CAUSES",
    "follow": "FOLLOWS",
    "adopt": "FOLLOWS",
    "depend": "DEPENDS_ON",
}

BANNED_SUBJECTS = {
    "figure", "table", "section", "appendix", "paper", "work",
    "authors", "we", "our model", "our approach", "our method",
    "this model", "this approach", "this method", "this architecture",
    "the model", "the approach", "the method", "the architecture"
}

class RelationExtractor:
    def __init__(self):
        self.concept_extractor = ConceptExtractor()
        self.nlp = spacy.load("en_core_web_sm")

    def extract(self, sentence: str) -> list:
        """
        Uses dependency parsing to find subject-verb-object relationships,
        then filters by known concepts and relation ontology.
        """
        doc = self.nlp(sentence)
        concepts = self.concept_extractor.extract_concepts(sentence)
        if len(concepts) < 2:
            return []

        facts = []
        for token in doc:
            # Focus on verbs that are the root or have both subject and object children
            if token.pos_ != "VERB":
                continue
            subj_tok = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            obj_tok = next((c for c in token.children if c.dep_ in ("dobj", "attr", "pobj")), None)
            if not subj_tok or not obj_tok:
                continue

            # Get the text of subject and object, and find matching concept (allow substring)
            subj_text = subj_tok.text.lower().strip()
            obj_text = obj_tok.text.lower().strip()
            subject_concept = None
            object_concept = None
            for conc in concepts:
                if subj_text in conc.split():
                    subject_concept = conc
                if obj_text in conc.split():
                    object_concept = conc
            # Fallback: try matching the whole token phrase against concept names
            if not subject_concept:
                for conc in concepts:
                    if conc.split()[0] == subj_text:
                        subject_concept = conc
                        break
            if not object_concept:
                for conc in concepts:
                    if conc.split()[-1] == obj_text:
                        object_concept = conc
                        break

            # Skip if subject or object not found or if they are the same
            if not subject_concept or not object_concept or subject_concept == object_concept:
                continue

            # Filter banned subjects
            if subject_concept.lower() in BANNED_SUBJECTS or object_concept.lower() in BANNED_SUBJECTS:
                continue

            # Map verb to canonical relation
            verb_lemma = token.lemma_.lower()
            rel_label = VERB_TO_RELATION.get(verb_lemma, token.text.upper())

            # Ensure we haven't already emitted this fact for the sentence
            fact = {
                'subject': subject_concept,
                'relation': rel_label,
                'object': object_concept,
                'confidence': 0.85,
                'evidence': sentence
            }
            # Simple deduplication: check if identical fact already in list
            if fact not in facts:
                facts.append(fact)

        return facts
