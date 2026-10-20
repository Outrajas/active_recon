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

        def get_noun_phrase(token):
            """
            Return the full noun phrase containing the token, with prepositional modifiers.
            If the token is not inside a noun chunk, return its subtree span.
            """
            # Find the noun chunk that contains this token
            for chunk in doc.noun_chunks:
                if chunk.start <= token.i < chunk.end:
                    # Get the chunk text
                    chunk_text = chunk.text
                    # Check if the chunk's head has a prepositional modifier (of/with)
                    head = chunk.root
                    # If head is a noun, check children for prep
                    if head.pos_ in ("NOUN", "PROPN"):
                        for child in head.children:
                            if child.dep_ == "prep" and child.lemma_ in ("of", "with"):
                                # Include the prepositional phrase by extending the span
                                expanded = doc[chunk.start : child.right_edge.i + 1].text
                                return expanded
                    return chunk_text
            # Fallback: use the token's subtree span
            return doc[token.left_edge.i : token.right_edge.i + 1].text

        def split_coordinated(phrase):
            """Split a phrase on ' and ' and commas to get separate conjuncts."""
            # Split on ' and ' (case‑insensitive) and on commas, but preserve phrases
            parts = []
            # Use re.split with capturing groups to keep the separators? Not needed.
            # First split on ' and '
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

            subj_span = get_noun_phrase(subj_tok)
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

            # Expand with conjuncts
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

            # Deduplicate object tokens
            unique_objs = []
            seen_indices = set()
            for tok in expanded_objs:
                if tok.i not in seen_indices:
                    seen_indices.add(tok.i)
                    unique_objs.append(tok)

            for obj_tok in unique_objs:
                obj_span = get_noun_phrase(obj_tok)
                obj_span = obj_span.strip().rstrip(',').rstrip(';')
                if not obj_span:
                    continue

                # Decide whether to split coordinated objects.
                # If the span contains "of" or "with", treat as a single phrase.
                if re.search(r'\b(of|with)\b', obj_span, re.IGNORECASE):
                    split_parts = [obj_span]
                else:
                    # Split on " and " and commas to get individual conjuncts
                    split_parts = split_coordinated(obj_span)

                for part in split_parts:
                    part = part.strip()
                    if not part:
                        continue
                    obj_concept = find_concept(part)
                    if subj_concept == obj_concept:
                        continue

                    # Noun‑level negation (if "no" appears in the subtree)
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

        # Deduplicate exact triples
        seen = set()
        unique_facts = []
        for f in facts:
            key = (f['subject'], f['relation'], f['object'])
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)
        return unique_facts
