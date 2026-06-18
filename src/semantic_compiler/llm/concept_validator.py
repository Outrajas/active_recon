# src/semantic_compiler/llm/concept_validator.py
import re

class ConceptValidator:
    def __init__(self):
        self.known_good = {
            'transformer', 'attention', 'encoder', 'decoder',
            'self-attention', 'multi-head attention', 'positional encoding',
            'residual connection', 'layer normalization', 'feed-forward',
            'bleu', 'wmt', 'adam', 'bytenet', 'convs2s',
            'gpu', 'nmt', 'rnn', 'lstm', 'cnn', 'softmax',
            'dropout', 'training', 'inference', 'beam search',
            'embedding', 'layer', 'stack', 'sub-layer', 'network',
            'encoder-decoder', 'encoder-decoder structure'
        }

    def validate(self, extraction_results: dict) -> dict:
        for chunk_id, objects in extraction_results.items():
            for obj in objects:
                if obj.get('document_role') != 'MAIN_CONTENT':
                    continue
                new_entities = []
                for ent in obj.get('entities', []):
                    valid, _ = self._is_valid_with_reason(ent)
                    if valid:
                        new_entities.append(ent)
                obj['entities'] = new_entities
        return extraction_results

    def _is_valid(self, ent: str) -> bool:
        valid, _ = self._is_valid_with_reason(ent)
        return valid

    def _is_valid_with_reason(self, ent: str) -> tuple:
        if not ent or len(ent) < 3:
            return False, "empty or too short"
        ent_lower = ent.strip().lower()
        # Allow layer count patterns – delegate to canonicalize later
        if re.match(r'^(\d+|n\s*=\s*\d+)\s+(identical\s+)?layers?$', ent_lower):
            return True, "layer count"
        if re.match(r'^stack\s+of\s+(\d+|n\s*=\s*\d+)\s+layers?$', ent_lower):
            return True, "stack of layers"
        if re.match(r'^(one|two|three|four|five|six|seven|eight|nine|ten)\s+sub-?layers?$', ent_lower):
            return True, "sub-layer count"
        if ent_lower in self.known_good:
            return True, "known_good"
        if re.search(r'\w+-\w+', ent_lower) and len(ent) > 4:
            return True, "hyphenated compound"
        if not any(ch.isalpha() for ch in ent):
            return False, "no alphabetic characters"
        if re.match(r'^[\d\W_]+$', ent):
            return False, "purely numeric or symbolic"
        if len(ent) <= 2:
            return False, "too short (<=2 chars)"
        math_chars = sum(1 for ch in ent if ch in r'\/+*=-^_{}()[]')
        if math_chars > 1 and math_chars >= len(ent) * 0.4:
            return False, "too many math symbols"
        if re.search(r'^\d+(\.\d+)*\s+', ent):
            return False, "starts with number pattern"
        if re.search(r'\d[a-zA-Z]|[a-zA-Z]\d', ent):
            return False, "digit-letter mix"
        if ent_lower in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'each', 'every', 'all', 'some', 'any', 'no', 'other', 'such', 'own', 'we', 'they', 'it', 'its', 'our', 'your', 'my', 'his', 'her', 'their'}:
            return False, "stopword"
        return True, "passed all checks"
