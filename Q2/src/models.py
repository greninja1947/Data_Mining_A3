import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


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


class NormalizedGATv2(torch.nn.Module):
    def __init__(self, mean_x, std_x, base):
        super().__init__()
        self.register_buffer("mean_x", mean_x)
        self.register_buffer("std_x", std_x)
        self.base = base

    def forward(self, x, edge_index):
        x = (x - self.mean_x) / (self.std_x + 1e-7)
        return self.base(x, edge_index)
