# src/semantic_compiler/llm/structural_extractor.py
import re
import spacy
from .concept_extractor import ConceptExtractor

class StructuralRelationExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.concept_extractor = ConceptExtractor()
        self.verb_map = {
            "contain": "CONTAINS",
            "consist": "CONTAINS",
            "comprise": "CONTAINS",
            "include": "CONTAINS",
            "have": "CONTAINS",
            "compose": "CONTAINS",
            "build": "CONTAINS",
            "form": "CONTAINS",
        }
        self.pronouns = {"it", "this", "that", "these", "those", "which", "who", "whom", "whose"}

    def extract(self, sentence: str, chunk_id: str = "", page: int = 0, section: str = "Unknown") -> list:
        doc = self.nlp(sentence)
        facts = []
        concepts = self.concept_extractor.extract_concepts(sentence)

        def find_concept(text: str):
            text_lower = text.lower().strip()
            for prefix in ["the ", "a ", "an "]:
                if text_lower.startswith(prefix):
                    text_lower = text_lower[len(prefix):]
            # Exact match
            for c in concepts:
                if c.lower() == text_lower:
                    return c
            # Avoid partial matches that collapse phrases
            if text_lower == "the transformer":
                return "transformer"
            return text_lower

        def get_noun_chunk(token):
            # Find the noun chunk containing this token
            for chunk in doc.noun_chunks:
                if chunk.start <= token.i < chunk.end:
                    return chunk.text
            # Fallback to span
            return doc[token.left_edge.i : token.right_edge.i + 1].text

        def split_coordinated(phrase):
            # Split on " and " and ", " to get individual conjuncts
            parts = []
            # First split on " and " (case insensitive)
            for part in re.split(r'\s+and\s+', phrase, flags=re.IGNORECASE):
                # Then split on commas
                for subpart in re.split(r'\s*,\s*', part):
                    subpart = subpart.strip()
                    if subpart:
                        parts.append(subpart)
            return parts

        for token in doc:
            if token.pos_ != "VERB" or token.lemma_ not in self.verb_map:
                continue

            # Negation (verb-level)
            has_neg = any(child.dep_ == "neg" for child in token.children)
            relation = self.verb_map[token.lemma_]
            if has_neg:
                relation = "DOES_NOT_" + relation

            # Subject
            subj_tok = None
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj_tok = child
                    break
            if not subj_tok:
                continue
            if subj_tok.text.lower() in self.pronouns:
                continue

            subj_span = get_noun_chunk(subj_tok)
            subj_concept = find_concept(subj_span)

            # Objects (direct or via preposition)
            obj_tokens = []
            for child in token.children:
                if child.dep_ in ("dobj", "attr"):
                    obj_tokens.append(child)
            for child in token.children:
                if child.dep_ == "prep" and child.lemma_ in ("of", "with"):
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj":
                            obj_tokens.append(grandchild)
            if not obj_tokens:
                continue

            # Expand with conjuncts but avoid over-expansion; we'll use split_coordinated later
            expanded_objs = []
            for obj_tok in obj_tokens:
                queue = [obj_tok]
                seen = set()
                while queue:
                    cur = queue.pop()
                    if cur.i in seen:
                        continue
                    seen.add(cur.i)
                    expanded_objs.append(cur)
                    for conj in cur.conjuncts:
                        if conj.i not in seen:
                            queue.append(conj)

            unique_objs = []
            seen_indices = set()
            for tok in expanded_objs:
                if tok.i not in seen_indices:
                    seen_indices.add(tok.i)
                    unique_objs.append(tok)

            for obj_tok in unique_objs:
                obj_span = get_noun_chunk(obj_tok)
                obj_span = obj_span.strip().rstrip(',').rstrip(';')
                if not obj_span:
                    continue

                # --- Split coordinated objects ---
                # We'll split on " and " and "," to get individual entities
                # But we need to be careful: if the span contains "stack of X", we don't want to split that.
                # So we only split if the span doesn't contain "of" or "with" (prepositional phrases)
                # Heuristic: if "of" or "with" appears, treat as a single phrase.
                if re.search(r'\b(of|with)\b', obj_span, re.IGNORECASE):
                    # Single phrase, do not split
                    split_parts = [obj_span]
                else:
                    split_parts = split_coordinated(obj_span)

                for part in split_parts:
                    part = part.strip()
                    if not part:
                        continue
                    obj_concept = find_concept(part)
                    if subj_concept == obj_concept:
                        continue

                    # Noun‑level negation
                    if any(tok.text.lower() == "no" for tok in obj_tok.subtree):
                        if not has_neg:
                            relation = "DOES_NOT_" + self.verb_map[token.lemma_]

                    start = sentence.lower().find(subj_span.lower()) if subj_span else 0
                    end = sentence.lower().rfind(part.lower()) + len(part) if part else 0
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
                        "fact_type": "STRUCTURAL"
                    })

        # Deduplicate
        seen = set()
        unique_facts = []
        for f in facts:
            key = (f['subject'], f['relation'], f['object'])
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)
        return unique_facts
