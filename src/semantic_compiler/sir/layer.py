# src/semantic_compiler/sir/layer.py
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Layer:
    id: str
    data: Dict[str, Any]
