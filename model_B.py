"""
model_B.py -- GNN architecture for COL761 A3 Q2 Dataset B.

Kept in a dedicated module (not in `train_B.py`) so that pickled checkpoints
saved with `torch.save(model, path)` can be loaded by `predict.py` without
re-running the trainer: torch.load just needs `import model_B` to succeed.

The model outputs raw logits of shape [N, 1] (predict_B squeezes & sigmoids).
A residual + LayerNorm stack keeps gradients stable for deeper graphs while
still working on NeighborLoader mini-batches (subgraph x/edge_index) and on
the full graph at inference time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv


class ResidualGNN(torch.nn.Module):
    """Input-project -> N x (conv + residual + LayerNorm) -> head."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 256,
        num_layers: int = 4,
        heads: int = 4,
        dropout: float = 0.3,
        conv_type: str = "sage",
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.hidden = hidden

        # Normalization stats are saved inside the model so inference can
        # reproduce the exact preprocessing without the caller needing to know.
        if feature_mean is not None:
            self.register_buffer("feature_mean", feature_mean)
            self.register_buffer("feature_std", feature_std)
        else:
            self.feature_mean = None
            self.feature_std = None

        self.input_proj = torch.nn.Linear(in_channels, hidden)

        convs: list[torch.nn.Module] = []
        for _ in range(num_layers):
            if conv_type == "sage":
                convs.append(SAGEConv(hidden, hidden))
            elif conv_type == "gatv2":
                convs.append(
                    GATv2Conv(hidden, hidden // heads, heads=heads,
                              concat=True, dropout=dropout)
                )
            else:
                raise ValueError(f"unknown conv_type: {conv_type}")
        self.convs = torch.nn.ModuleList(convs)
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden) for _ in range(num_layers)]
        )
        self.out = torch.nn.Linear(hidden, 1)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mean is None:
            return x
        return (x - self.feature_mean) / (self.feature_std + 1e-8)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(x + h)
        return self.out(x)  # [N, 1] raw logits
