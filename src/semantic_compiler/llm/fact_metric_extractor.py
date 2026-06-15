# src/semantic_compiler/llm/fact_metric_extractor.py
import re
from typing import List, Dict, Tuple

class FactMetricExtractor:
    def __init__(self):
        self.metric_patterns = [
            (r'(\d+\.?\d*)\s*(BLEU|perplexity|accuracy|F1|loss|error rate|seconds|hours|days|steps|epochs|GPUs?)\b', None),
            (r'(\d+\.?\d*)\s*(%|percent)', 'percentage'),
            (r'(\d+)\s*(GPUs?|hours|days|steps)\b', None),
        ]

    def enrich(self, extraction_results: dict) -> dict:
        for chunk_id, objects in extraction_results.items():
            for obj in objects:
                if obj.get('document_role') != 'MAIN_CONTENT':
                    continue
                text = obj.get('text', '')
                metrics = []
                for pattern, unit_override in self.metric_patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        value = float(match.group(1))
                        unit = match.group(2).lower() if not unit_override else unit_override
                        dataset = self._detect_dataset(text)
                        metrics.append({
                            "metric": unit,
                            "value": value,
                            "dataset": dataset,
                            "confidence": 0.9
                        })
                obj['metrics'] = metrics
        return extraction_results

    def _detect_dataset(self, text: str) -> str:
        if re.search(r'WMT\s*14|WMT\s*2014|English.to.German|English.to.French|EN.DE|EN.FR', text, re.IGNORECASE):
            return "WMT14"
        if re.search(r'newstest', text, re.IGNORECASE):
            return "newstest"
        return "unknown"
