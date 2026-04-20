"""
models.py  –  GNN model definitions for COL761 Assignment 3.
Imported by both train_*.py and predict.py so torch.save/load resolves classes.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv


# ─────────────────────────────────────────────────────────────────────────────
# Dataset A: GATv2 node classifier (7-class, Accuracy)
# ─────────────────────────────────────────────────────────────────────────────

class GATv2NodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, heads=8, num_classes=7, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.lin = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.lin(x)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset B: Deep GATv2 node classifier (binary, ROC-AUC)
# ─────────────────────────────────────────────────────────────────────────────

class DeepGATv2(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, heads=4, num_layers=4, dropout=0.5):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden * heads)
        self.convs = torch.nn.ModuleList([
            GATv2Conv(hidden * heads, hidden, heads=heads, concat=True, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden * heads) for _ in range(num_layers)
        ])
        self.dropout = dropout
        self.out = torch.nn.Linear(hidden * heads, 1)

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.elu(conv(x, edge_index)), p=self.dropout, training=self.training)
            x = norm(x + h)
        return self.out(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset C: GraphSAGE + MLP decoder for link prediction (Hits@50)
# ─────────────────────────────────────────────────────────────────────────────

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = dropout
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def decode(self, z, edges):
        src, dst = edges[:, 0], edges[:, 1]
        return self.decoder(torch.cat([z[src], z[dst]], dim=-1)).squeeze(-1)

    def forward(self, x, edge_index, edge_pairs):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_pairs)
