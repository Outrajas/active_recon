# src/semantic_compiler/llm/semantic_objects.py
import spacy, re
from typing import List, Dict, Tuple
from .document_role import DocumentRoleClassifier
from .canonicalize import is_valid_entity, canonical_name
from .concept_extractor import ConceptExtractor
from .relation_extractor import RelationExtractor

class SemanticObjectExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.role_classifier = DocumentRoleClassifier()
        self.concept_extractor = ConceptExtractor()
        self.relation_extractor = RelationExtractor()

    def extract(self, text: str, chunk_id: str) -> List[Dict]:
        doc = self.nlp(text)
        objects = []
        seen_texts = set()

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text or len(sent_text) < 10:
                continue
            if sent_text in seen_texts:
                continue
            seen_texts.add(sent_text)

            doc_role = self.role_classifier.classify(sent_text)

            if doc_role == "MAIN_CONTENT":
                sem_role = self._classify_semantic_role(sent)
                concepts = self.concept_extractor.extract_concepts(sent_text)
                # Use relation extractor to get facts from concepts
                facts = self.relation_extractor.extract(sent_text)
                triples = []  # no longer needed
            else:
                sem_role = "NONE"
                concepts = []
                facts = []
                triples = []

            objects.append({
                "text": sent_text,
                "semantic_role": sem_role,
                "document_role": doc_role,
                "confidence": 0.9 if sem_role not in ("CLAIM", "NONE") else 0.85,
                "source_chunk": chunk_id,
                "source_offsets": [sent.start_char, sent.end_char],
                "triples": triples,
                "concepts": concepts,
                "facts": facts
            })

        prag_objects = self._detect_pragmatic(text, chunk_id)
        for po in prag_objects:
            po["document_role"] = "MAIN_CONTENT"
            po["concepts"] = []
            po["triples"] = []
            po["facts"] = []
            po.setdefault("semantic_role", "NONE")
        objects.extend(prag_objects)

        return objects

    def _classify_semantic_role(self, sent) -> str:
        text = sent.text
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\bwe\s+(propose|introduce|present|describe|train|implement|evaluate|compare|use)\b']): return "METHOD"
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\b(bleu|accuracy|perplexity|f1|score|outperforms|achieves?\s+\d+\.\d+)\b']): return "RESULT"
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\b(is\s+defined\s+as|is\s+called|refers\s+to|denotes)\b']): return "DEFINITION"
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\b(compared\s+to|in\s+contrast|while|whereas|outperforms)\b']): return "COMPARISON"
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\b(limitation|drawback|however|fails?\s+to|cannot)\b']): return "LIMITATION"
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\b(we\s+assume|assuming|suppose)\b']): return "ASSUMPTION"
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\b(we\s+observe|we\s+note|interestingly|we\s+find\s+that)\b']): return "OBSERVATION"
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\b(we\s+conclude|in\s+summary|overall|thus|therefore)\b']): return "CONCLUSION"
        if any(re.search(p, text, re.IGNORECASE) for p in [r'\b(to\s+address|to\s+overcome|we\s+aim|motivated\s+by)\b']): return "MOTIVATION"
        return "CLAIM"

    def _detect_pragmatic(self, text: str, chunk_id: str) -> List[Dict]:
        res = []
        for m in re.finditer(r'(\b(?:like\s+(?:a\s+)?\w+|as\s+if)\b.*?[\.!,])', text, re.IGNORECASE):
            res.append({"semantic_role": "NONE", "text": m.group(1), "confidence": 0.8, "source_chunk": chunk_id, "source_offsets": [m.start(), m.end()]})
        if any(w in text.lower() for w in ['drown', 'prison']) and not res:
            res.append({"semantic_role": "NONE", "text": "Text contains strong metaphorical imagery", "confidence": 0.7, "source_chunk": chunk_id, "source_offsets": [0, len(text)]})
        for m in re.finditer(r'\b(might|perhaps|possibly|may|could|suggest)\b', text, re.IGNORECASE):
            res.append({"semantic_role": "HEDGING", "text": m.group(0), "confidence": 0.85, "source_chunk": chunk_id, "source_offsets": [m.start(), m.end()]})
        for m in re.finditer(r'\b(clearly|obviously|undoubtedly|certainly)\b', text, re.IGNORECASE):
            res.append({"semantic_role": "CERTAINTY", "text": m.group(0), "confidence": 0.85, "source_chunk": chunk_id, "source_offsets": [m.start(), m.end()]})
        return res
