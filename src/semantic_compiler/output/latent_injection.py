# src/semantic_compiler/output/latent_injection.py
import torch
import torch.nn as nn

class MLPProjectionAdapter(nn.Module):
    """MLP that projects the concatenated GNN+VSA vector into a soft prompt."""
    def __init__(self, input_dim: int, hidden_dim: int, prompt_length: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_length * input_dim)
        )

    def forward(self, combined_vec):
        x = self.proj(combined_vec)
        # Reshape to (prompt_length, input_dim)
        return x.view(-1, self.proj[-1].out_features // (x.shape[0] or 1))
