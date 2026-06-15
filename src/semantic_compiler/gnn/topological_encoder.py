# src/semantic_compiler/gnn/topological_encoder.py
import torch
import torch.nn as nn

class TopologicalEncoder(nn.Module):
    """GNN that encodes a knowledge graph into a single embedding vector."""
    def __init__(self, in_feats: int, hidden_feats: int, out_feats: int):
        super().__init__()
        self.conv1 = nn.Linear(in_feats, hidden_feats)
        self.conv2 = nn.Linear(hidden_feats, out_feats)

    def forward(self, graph):
        # Placeholder: flatten node features (if any) and pool
        # In production, replace with message-passing over the graph
        return torch.randn(1, self.conv2.out_features)
