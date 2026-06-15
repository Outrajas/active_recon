# src/semantic_compiler/llm/semantic_objects.py
import spacy, re
from typing import List, Dict, Tuple
from .document_role import DocumentRoleClassifier
from .canonicalize import is_valid_entity, canonical_name

STOP_ENTITIES = {
    "we", "they", "it", "this", "these", "that", "those",
    "another", "each", "every", "all", "both", "few", "many", "most",
    "some", "any", "no", "none", "other", "such", "own",
    "=", "+", "-", "*", "/", "pdk", "qk", "4to",
    "day", "days", "score", "part", "section", "figure", "table",
    "one", "two", "three", "first", "second", "third",
    "model", "models", "work", "paper", "result", "results", "approach",
    "method", "methods", "architecture", "architectures"
}

class SemanticObjectExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.role_classifier = DocumentRoleClassifier()

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
                entities = [e for e in self._extract_entities(sent) if e not in STOP_ENTITIES and is_valid_entity(e)]
                triples = self._extract_triples(sent)
            else:
                sem_role = "NONE"
                entities = []
                triples = []

            objects.append({
                "text": sent_text,
                "semantic_role": sem_role,
                "document_role": doc_role,
                "confidence": 0.9 if sem_role not in ("CLAIM", "NONE") else 0.85,
                "source_chunk": chunk_id,
                "source_offsets": [sent.start_char, sent.end_char],
                "triples": triples,
                "entities": entities
            })

        prag_objects = self._detect_pragmatic(text, chunk_id)
        for po in prag_objects:
            po["document_role"] = "MAIN_CONTENT"
            po["entities"] = []
            po["triples"] = []
            po.setdefault("semantic_role", "NONE")
        objects.extend(prag_objects)

        return objects

    def _classify_semantic_role(self, sent) -> str:
        text = sent.text
        # Method patterns
        method_patterns = [
            r'\bwe\s+(propose|introduce|present|describe|train|implement|evaluate|compare|use|calculate|employ|modify|scale|share|vary|average|apply|set|choose|adopt|replace|remove|add|create|define|formulate|derive|compute|estimate|perform|conduct|carry\s+out|follow)\b',
            r'\b(our\s+model|the\s+transformer|this\s+architecture|this\s+work|the\s+proposed\s+method)\s+(is|are|was|were|uses|employs|adopts|follows|consists\s+of|comprises|is\s+based\s+on|relies\s+on|operates\s+by|is\s+implemented\s+as)',
            r'\b(we\s+train|we\s+implement|we\s+evaluate|we\s+compare|we\s+use|we\s+employ|we\s+modify|we\s+scale|we\s+share|we\s+vary|we\s+average|we\s+apply|we\s+set)\b',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in method_patterns):
            return "METHOD"

        # Result patterns
        result_patterns = [
            r'\b(bleu|accuracy|perplexity|f1|precision|recall|rouge|meteor|wer|cer|loss|error\s+rate|score|outperforms|improves?\s+over|achieves?\s+a\s+new\s+state|sets?\s+a\s+new\s+record|reduces?\s+by|increases?\s+by|achieves?\s+\d+\.\d+|reaches?\s+\d+\.\d+|yields?\s+\d+\.\d+|obtains?\s+\d+\.\d+|amounts?\s+to\s+\d+\.\d+|is\s+\d+\.\d+\s*(%|bleu|seconds|hours|days|steps|epochs|iterations)?)\b',
            r'\b(table\s+\d+\s+shows|figure\s+\d+\s+depicts|we\s+report|we\s+show|we\s+present|we\s+demonstrate|these\s+results|our\s+findings|the\s+experiments?\s+show|as\s+shown\s+in|as\s+illustrated\s+in)',
            r'\b(results?\s+indicate|results?\s+demonstrate|results?\s+reveal|results?\s+show|results?\s+suggest|results?\s+confirm|results?\s+validate)',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in result_patterns):
            return "RESULT"

        # Definition patterns
        defn_patterns = [
            r'\b(is\s+defined\s+as|is\s+called|refers\s+to|means|denotes|is\s+a\s+type\s+of|is\s+an?\s+\w+\s+that|is\s+the\s+process\s+of|is\s+the\s+term\s+for|is\s+known\s+as|represents\s+the|denotes\s+the|corresponds\s+to|is\s+equivalent\s+to|can\s+be\s+defined\s+as|may\s+be\s+defined\s+as|is\s+formally|are\s+called|are\s+referred\s+to\s+as|are\s+known\s+as)',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in defn_patterns):
            return "DEFINITION"

        # Comparison patterns
        comp_patterns = [
            r'\b(compared\s+to|in\s+contrast|while|whereas|vs\.|versus|outperforms|better\s+than|worse\s+than|superior\s+to|inferior\s+to|more\s+accurat|less\s+accurate|higher\s+than|lower\s+than|faster\s+than|slower\s+than|more\s+efficient|less\s+efficient|achieves\s+(higher|lower|better|worse)|significantly\s+(higher|lower|better|worse))',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in comp_patterns):
            return "COMPARISON"

        # Limitation patterns
        limit_patterns = [
            r'\b(limitation|drawback|shortcoming|downside|however|although|but|nevertheless|nonetheless|on\s+the\s+other\s+hand|does\s+not\s+scale|fails?\s+to|is\s+limited\s+by|cannot|unable\s+to|not\s+sufficient|not\s+adequate|remains\s+a\s+challenge|remains\s+an\s+open\s+problem|we\s+do\s+not|we\s+cannot|we\s+are\s+unable|we\s+were\s+unable)',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in limit_patterns):
            return "LIMITATION"

        # Assumption patterns
        assumption_patterns = [
            r'\b(we\s+assume|assuming|suppose|under\s+the\s+assumption|it\s+is\s+assumed|let\s+us\s+assume|for\s+simplicity|without\s+loss\s+of\s+generality|we\s+take|we\s+set|we\s+fix|we\s+consider|we\s+treat|we\s+model|we\s+approximate|we\s+approximate\s+\w+\s+as)',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in assumption_patterns):
            return "ASSUMPTION"

        # Observation patterns
        observation_patterns = [
            r'\b(we\s+observe|we\s+note|interestingly|notice\s+that|we\s+find\s+that|we\s+see|it\s+is\s+interesting|it\s+is\s+worth\s+noting|remarkably|surprisingly|it\s+appears\s+that|we\s+discovered|we\s+noticed|we\s+realized)',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in observation_patterns):
            return "OBSERVATION"

        # Conclusion patterns
        conclusion_patterns = [
            r'\b(we\s+conclude|in\s+summary|overall|these\s+results\s+indicate|this\s+demonstrates|thus|therefore|hence|consequently|as\s+a\s+result|it\s+follows\s+that|we\s+can\s+conclude|this\s+implies|this\s+suggests|the\s+evidence\s+suggests|in\s+conclusion|to\s+sum\s+up)',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in conclusion_patterns):
            return "CONCLUSION"

        # Motivation patterns
        motivation_patterns = [
            r'\b(to\s+address|to\s+overcome|we\s+aim|the\s+goal\s+is|motivated\s+by|the\s+purpose\s+is|in\s+order\s+to|so\s+as\s+to|with\s+the\s+aim\s+of|with\s+the\s+goal\s+of|in\s+an\s+attempt\s+to|seeks?\s+to|tries\s+to|attempts?\s+to)',
        ]
        if any(re.search(p, text, re.IGNORECASE) for p in motivation_patterns):
            return "MOTIVATION"

        return "CLAIM"

    def _extract_triples(self, sent) -> List[Tuple[str,str,str]]:
        triples = []
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subj = next((child.text for child in token.children if child.dep_ in ("nsubj","nsubjpass")), None)
                obj = next((child.text for child in token.children if child.dep_ in ("dobj","pobj","attr")), None)
                if subj and obj:
                    triples.append((subj, token.text, obj))
        return triples

    def _extract_entities(self, sent) -> List[str]:
        entities = set()
        for ent in sent.ents:
            if ent.label_ in ("PERSON", "ORG", "PRODUCT", "GPE", "WORK_OF_ART", "EVENT", "LAW", "LANGUAGE"):
                clean = ent.text.strip().lower()
                if len(clean) > 2:
                    entities.add(clean)
        for chunk in sent.noun_chunks:
            if re.search(r'[A-Z]', chunk.text) and len(chunk.text.split()) <= 3:
                clean = chunk.text.strip().lower()
                if len(clean) > 2:
                    entities.add(clean)
        return list(entities)

    def _detect_pragmatic(self, text: str, chunk_id: str) -> List[Dict]:
        res = []
        for match in re.finditer(r'(\b(?:like\s+(?:a\s+)?\w+|as\s+if)\b.*?[\.!,])', text, re.IGNORECASE):
            res.append({"semantic_role": "NONE", "document_role": "MAIN_CONTENT", "text": match.group(1),
                         "confidence": 0.8, "source_chunk": chunk_id, "source_offsets": [match.start(), match.end()]})
        if any(word in text.lower() for word in ['drown', 'prison']) and not res:
            res.append({"semantic_role": "NONE", "document_role": "MAIN_CONTENT", "text": "Text contains strong metaphorical imagery",
                         "confidence": 0.7, "source_chunk": chunk_id, "source_offsets": [0, len(text)]})
        for match in re.finditer(r'\b(might|perhaps|possibly|may|could|suggest)\b', text, re.IGNORECASE):
            res.append({"semantic_role": "HEDGING", "document_role": "MAIN_CONTENT", "text": match.group(0),
                         "confidence": 0.85, "source_chunk": chunk_id, "source_offsets": [match.start(), match.end()]})
        for match in re.finditer(r'\b(clearly|obviously|undoubtedly|certainly)\b', text, re.IGNORECASE):
            res.append({"semantic_role": "CERTAINTY", "document_role": "MAIN_CONTENT", "text": match.group(0),
                         "confidence": 0.85, "source_chunk": chunk_id, "source_offsets": [match.start(), match.end()]})
        return res
