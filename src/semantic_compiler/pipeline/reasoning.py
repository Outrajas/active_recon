# src/semantic_compiler/pipeline/reasoning.py
from semantic_compiler.reasoning.diagnostic_engine import DiagnosticEngine

class ReasoningStage:
    """Stage 7: Run diagnostic reasoning on the consolidated graph."""
    def __init__(self, engine: DiagnosticEngine = None):
        self.engine = engine or DiagnosticEngine()

    def run(self, graph):
        return self.engine.execute(graph)
