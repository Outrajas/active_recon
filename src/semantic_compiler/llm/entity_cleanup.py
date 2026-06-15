# src/semantic_compiler/llm/entity_cleanup.py
import re
from collections import Counter, defaultdict

class EntityCleanup:
    def __init__(self):
        self.noise_words = {
            "goal", "way", "section", "chapter", "figure", "table", "work",
            "foundation", "problem", "approach", "method", "result", "results",
            "architecture", "model", "models", "paper", "task", "tasks",
            "our model", "this architecture", "the proposed model", "our approach",
            "this method", "the model", "this work", "recent work"
        }
        self.coref_phrases = {
            "our model", "this architecture", "the proposed model", "our approach",
            "this method", "the model", "this work", "the transformer"
        }

    def clean(self, extraction_results: dict) -> dict:
        chunk_entity_counts = defaultdict(Counter)
        for chunk_id, objects in extraction_results.items():
            for obj in objects:
                if obj.get('document_role') == 'MAIN_CONTENT':
                    for ent in obj.get('entities', []):
                        chunk_entity_counts[chunk_id][ent] += 1

        chunk_salient = {}
        for chunk_id, counter in chunk_entity_counts.items():
            proper = {e: c for e, c in counter.items() if any(ch.isupper() for ch in e) and len(e) > 2}
            if proper:
                chunk_salient[chunk_id] = max(proper, key=proper.get)
            elif counter:
                chunk_salient[chunk_id] = max(counter, key=counter.get)
            else:
                chunk_salient[chunk_id] = None

        for chunk_id, objects in extraction_results.items():
            salient = chunk_salient.get(chunk_id)
            for obj in objects:
                if obj.get('document_role') != 'MAIN_CONTENT':
                    continue
                new_entities = []
                for ent in obj.get('entities', []):
                    if self._is_noise(ent):
                        continue
                    if ent.lower() in self.coref_phrases and salient:
                        new_entities.append(salient)
                    else:
                        new_entities.append(ent)
                obj['entities'] = new_entities
        return extraction_results

    def _is_noise(self, ent: str) -> bool:
        ent_lower = ent.lower().strip()
        if ent_lower in self.noise_words:
            return True
        if len(ent) < 3 or len(ent) > 50:
            return True
        if re.match(r'^[\d\W]+$', ent):
            return True
        # ** NEW: filter formula fragments **
        # If the entity is mostly math symbols or has very few letters
        letters = sum(1 for ch in ent if ch.isalpha())
        if letters < len(ent) * 0.5 and len(ent) > 1:   # less than half letters
            return True
        # Very short and containing numbers or isolated symbols
        if len(ent) <= 4 and re.search(r'[\d\\^_{}()\[\]]', ent):
            return True
        # Generic patterns that leaked
        if re.search(r'\b(section|chapter|figure|table|appendix)\b', ent_lower):
            return True
        if not any(c.isupper() for c in ent) and len(ent.split()) == 1 and ent_lower not in {
            'transformer', 'attention', 'encoder', 'decoder', 'bleu', 'adam', 'wmt', 'gpu', 'nmt'
        }:
            return True
        return False
