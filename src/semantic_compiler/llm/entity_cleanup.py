# src/semantic_compiler/llm/entity_cleanup.py
import re
from collections import Counter, defaultdict

class EntityCleanup:
    """Filters noisy entities and resolves simple coreferences within each chunk."""
    def __init__(self):
        # Words that make an entity likely to be noise
        self.noise_words = {
            "goal", "way", "section", "chapter", "figure", "table", "work",
            "foundation", "problem", "approach", "method", "result", "results",
            "architecture", "model", "models", "paper", "task", "tasks",
            "our model", "this architecture", "the proposed model", "our approach",
            "this method", "the model", "this work", "recent work"
        }
        # Coreference phrases that should be replaced by the chunk's salient entity
        self.coref_phrases = {
            "our model", "this architecture", "the proposed model", "our approach",
            "this method", "the model", "this work", "the transformer"
        }

    def clean(self, extraction_results: dict) -> dict:
        # First pass: count entity frequencies per chunk
        chunk_entity_counts = defaultdict(Counter)
        for chunk_id, objects in extraction_results.items():
            for obj in objects:
                if obj.get('document_role') == 'MAIN_CONTENT':
                    for ent in obj.get('entities', []):
                        chunk_entity_counts[chunk_id][ent] += 1

        # Determine the most salient entity for each chunk
        chunk_salient = {}
        for chunk_id, counter in chunk_entity_counts.items():
            # Prefer entities that look like proper nouns (contain uppercase)
            proper = {e: c for e, c in counter.items() if any(ch.isupper() for ch in e) and len(e) > 2}
            if proper:
                chunk_salient[chunk_id] = max(proper, key=proper.get)
            elif counter:
                chunk_salient[chunk_id] = max(counter, key=counter.get)
            else:
                chunk_salient[chunk_id] = None

        # Second pass: filter entities and resolve coreferences
        for chunk_id, objects in extraction_results.items():
            salient = chunk_salient.get(chunk_id)
            for obj in objects:
                if obj.get('document_role') != 'MAIN_CONTENT':
                    continue
                new_entities = []
                for ent in obj.get('entities', []):
                    # Filter out noise
                    if self._is_noise(ent):
                        continue
                    # Resolve coreference
                    if ent.lower() in self.coref_phrases and salient:
                        new_entities.append(salient)
                    else:
                        new_entities.append(ent)
                obj['entities'] = new_entities
        return extraction_results

    def _is_noise(self, ent: str) -> bool:
        ent_lower = ent.lower().strip()
        # Absolute stopwords
        if ent_lower in self.noise_words:
            return True
        # Too short or too long
        if len(ent) < 3 or len(ent) > 50:
            return True
        # Only numbers / punctuation
        if re.match(r'^[\d\W]+$', ent):
            return True
        # Contains generic section markers
        if re.search(r'\b(section|chapter|figure|table|appendix)\b', ent_lower):
            return True
        # If it's a single common English word (no proper nouns)
        if not any(c.isupper() for c in ent) and len(ent.split()) == 1 and ent_lower not in {'transformer', 'attention', 'encoder', 'decoder', 'bleu', 'adam'}:
            # Very likely generic (like "goal", "foundation") – we keep only known key terms
            return True
        return False
