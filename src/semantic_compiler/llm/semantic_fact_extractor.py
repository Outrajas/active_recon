# src/semantic_compiler/llm/semantic_fact_extractor.py
import re
from .concept_validator import ConceptValidator

class SemanticFactExtractor:
    def __init__(self):
        self.validator = ConceptValidator()
        self.patterns = [
            (r'(\b[A-Z][\w\s-]+?)\s+(uses|utilizes|employs|relies on|leverages|is based on)\s+([\w\s-]+?\b)', 'USES'),
            (r'(\b[A-Z][\w\s-]+?)\s+(consists of|contains|comprises|includes|is composed of)\s+([\w\s-]+?\b)', 'CONTAINS'),
            (r'(\b[A-Z][\w\s-]+?)\s+(achieves|obtains|reaches|yields|scores?|attains)\s+(\d+\.?\d*\s*(?:BLEU|score|accuracy|%|percent|perplexity)?)', 'ACHIEVES'),
            (r'(\b[A-Z][\w\s-]+?)\s+(outperforms|beats|surpasses|exceeds|performs better than)\s+([\w\s-]+?\b)', 'OUTPERFORMS'),
            (r'(\b[A-Z][\w\s-]+?)\s+(replaces|substitutes|removes|eliminates)\s+([\w\s-]+?\b)', 'REPLACES'),
            (r'(\b[A-Z][\w\s-]+?)\s+(improves|enhances|boosts|increases)\s+([\w\s-]+?\b)', 'IMPROVES'),
            (r'(\b[A-Z][\w\s-]+?)\s+(causes|leads to|results in|contributes to)\s+([\w\s-]+?\b)', 'CAUSES'),
            (r'(\b[A-Z][\w\s-]+?)\s+(enables|allows|permits)\s+([\w\s-]+?\b)', 'ENABLES'),
            (r'(\b[A-Z][\w\s-]+?)\s+(depends on|relies on)\s+([\w\s-]+?\b)', 'DEPENDS_ON'),
            (r'(\b[A-Z][\w\s-]+?)\s+(follows|adopts)\s+([\w\s-]+?\b)', 'FOLLOWS'),
        ]

    def extract_facts(self, sentence: str, salient_entity: str = None) -> list:
        facts = []
        for pattern, relation in self.patterns:
            for match in re.finditer(pattern, sentence, re.IGNORECASE | re.DOTALL):
                subj = match.group(1).strip()
                obj = match.group(3).strip()
                subj = re.sub(r'\s+', ' ', subj).strip()
                obj = re.sub(r'\s+', ' ', obj).strip()
                if salient_entity and subj.lower() in {'we', 'our model', 'this architecture', 'the model', 'this work', 'our approach'}:
                    subj = salient_entity
                if not self.validator._is_valid(subj) or not self.validator._is_valid(obj):
                    continue
                facts.append({
                    'subject': subj,
                    'relation': relation,
                    'object': obj,
                    'evidence': sentence,
                    'confidence': 0.85
                })
        return facts
