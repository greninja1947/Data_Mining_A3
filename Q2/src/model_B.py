import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv


class ResidualGNN(torch.nn.Module):

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

        # always register as buffers so they move with .to(device)
        self.register_buffer("feature_mean",
                             feature_mean if feature_mean is not None
                             else torch.zeros(in_channels))
        self.register_buffer("feature_std",
                             feature_std if feature_std is not None
                             else torch.ones(in_channels))

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = (x - self.feature_mean) / (self.feature_std + 1e-8)
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = norm(x + h)
        return self.out(x)
