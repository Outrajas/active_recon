# src/semantic_compiler/llm/concept_validator.py
import re

class ConceptValidator:
    """Filters extraction results to keep only entities that look like real concepts."""
    def __init__(self):
        # Whitelist of known good terms (lowercase)
        self.known_good = {
            'transformer', 'attention', 'encoder', 'decoder',
            'self-attention', 'multi-head attention', 'multi-head attention',
            'positional encoding', 'residual connection', 'layer normalization',
            'feed-forward', 'bleu', 'wmt', 'adam', 'bytenet', 'convs2s',
            'gpu', 'nmt', 'rnn', 'lstm', 'cnn', 'softmax',
            'dropout', 'training', 'inference', 'beam search',
        }

    def validate(self, extraction_results: dict) -> dict:
        for chunk_id, objects in extraction_results.items():
            for obj in objects:
                if obj.get('document_role') != 'MAIN_CONTENT':
                    continue
                new_entities = []
                for ent in obj.get('entities', []):
                    if self._is_valid(ent):
                        new_entities.append(ent)
                obj['entities'] = new_entities
        return extraction_results

    def _is_valid(self, ent: str) -> bool:
        if not ent or len(ent) < 3:
            return False
        ent_lower = ent.strip().lower()
        if ent_lower in self.known_good:
            return True
        # Heuristic: must be mostly alphabetic and contain at least one letter
        if not any(ch.isalpha() for ch in ent):
            return False
        # Must not be purely numeric or symbolic
        if re.match(r'^[\d\W_]+$', ent):
            return False
        # Must not be a single word that looks like a variable (e.g., "x", "n", "k")
        if len(ent) <= 2:
            return False
        # If it contains a lot of math symbols, reject
        math_chars = sum(1 for ch in ent if ch in r'\/+*=-^_{}()[]')
        if math_chars > 1 and math_chars >= len(ent) * 0.4:
            return False
        # Must not be a section number or training step pattern
        if re.search(r'^\d+(\.\d+)*\s+', ent):
            return False
        # Reject if it looks like a fragment of a word (e.g., "5we")
        if re.search(r'\d[a-zA-Z]|[a-zA-Z]\d', ent):
            return False
        # Reject if it's just a very common English stopword
        if ent_lower in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'each', 'every', 'all', 'some', 'any', 'no', 'other', 'such', 'own', 'we', 'they', 'it', 'its', 'our', 'your', 'my', 'his', 'her', 'their'}:
            return False
        return True
