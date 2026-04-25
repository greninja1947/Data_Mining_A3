from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class LinkPredictor(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        hidden: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)

        convs: list[torch.nn.Module] = []
        convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            convs.append(SAGEConv(hidden, hidden))
        self.convs = torch.nn.ModuleList(convs)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / (self.feature_std + 1e-8)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self._normalize(x)
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def decode(self, z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        src, dst = edges[:, 0], edges[:, 1]
        h = torch.cat([z[src], z[dst]], dim=-1)
        return self.decoder(h).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edges: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edges)
