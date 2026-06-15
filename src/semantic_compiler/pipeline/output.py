# src/semantic_compiler/pipeline/output.py
import networkx as nx
import torch
from semantic_compiler.output.linearization_compiler import LinearizationCompiler

class OutputStage:
    """Stage 8: Reconstruct graph as pseudo‑language or latent injection."""
    def run_pseudo_language(self, graph: nx.DiGraph) -> str:
        compiler = LinearizationCompiler()
        return compiler.compile(graph)

    def run_latent_injection(self, graph: nx.DiGraph) -> torch.Tensor:
        # Placeholder: use GNN + MLP adapter
        return torch.randn(1, 768)
