# COL761 Assignment 3 — Question 2: Graph Neural Networks Cheat Sheet

---

## What is the Problem?

You are given **3 graphs** and must train a GNN model on each. The grading is **competitive** — your score = (your metric) / (class best metric). So the better your model, the more marks.

| Graph | Task | Output | Metric | Time Limit | Marks |
|-------|------|--------|--------|------------|-------|
| A | Node Classification | 7 classes | Accuracy ↑ | 1 hour | 14 |
| B | Node Classification | 2 classes | ROC-AUC ↑ | 2 hours | 28 |
| C | Link Prediction | Edge exists? | Hits@50 ↑ | 2 hours | 28 |

---

## What Are We Supposed to Do?

1. **Train a GNN model** on each graph using the provided training data.
2. **Save the model** as `cs1230459_model_A.pt`, `cs1230459_model_B.pt`, `cs1230459_model_C.pt`.
3. **Generate predictions** on test data using `predict.py`.
4. **Submit** the code in the required directory structure.

---

## The Pipeline (Step-by-Step)

```
Load Dataset → Build GNN Model → Train → Validate → Save Model → Predict → Evaluate
```

### Step 1: Load Dataset
```bash
python load_dataset.py --dataset A --data_dir /path/to/public_datasets
python load_dataset.py --dataset B --data_dir /path/to/public_datasets
python load_dataset.py --dataset C --data_dir /path/to/public_datasets
```

### Step 2: Train Your Model
Write your training script (e.g., `train_A.py`, `train_B.py`, `train_C.py`).
Save the trained model as: `<kerberos>_model_<dataset>.pt`
Example: `cs1230459_model_A.pt`

### Step 3: Generate Predictions
```bash
# Node classification (A or B):
python predict.py --dataset A --task node \
  --data_dir /path/to/public_datasets \
  --model_path /path/to/models/ \
  --output_dir /path/to/outputs \
  --kerberos cs1230459

# Link prediction (C):
python predict.py --dataset C --task link \
  --data_dir /path/to/public_datasets \
  --model_path /path/to/models/ \
  --output_dir /path/to/outputs \
  --kerberos cs1230459
```
This creates: `cs1230459_predictions_A.pt`, `cs1230459_predictions_C.pt` etc.

### Step 4: Evaluate (on validation set)
```bash
python evaluate.py --dataset A --task node \
  --data_dir /path/to/public_datasets \
  --output_dir /path/to/outputs \
  --kerberos cs1230459
```

---

## What Does the Data Look Like?

### Graphs A and B (Node Classification)
```python
data.x             # node features [N, F]
data.edge_index    # all edges [2, E]
data.y             # labels for labeled nodes only [L]
data.labeled_nodes # which nodes have labels [L]
data.train_mask    # bool mask for training nodes [L]
data.val_mask      # bool mask for validation nodes [L]
data.num_classes   # 7 for A, 2 for B
```

### Graph C (Link Prediction)
```python
data.x          # node feature embeddings [N, D]
data.edge_index # training graph [2, 2M]
data.train_pos  # positive training edges [M, 2]
data.train_neg  # negative training edges [M, 2]
data.valid_pos  # positive validation edges [V, 2]
data.valid_neg  # 500 hard negatives per valid pos [V, 500, 2]
```

---

## Recommended Architectures

---

### Graph A — GATv2 (1 hour budget)

**Why GATv2?** Graph Attention Network v2 uses dynamic attention (each edge weight depends on both source and target features). Consistently top accuracy on multi-class node classification. Trains fast enough for 1 hour.

**Architecture:**
```
Input features → GATv2Conv (256 hidden, 8 heads) → ELU → Dropout(0.6)
              → GATv2Conv (256 hidden, 8 heads) → ELU → Dropout(0.6)
              → Linear → 7-class output
```

**Implementation:**
```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATv2NodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, heads=8, num_classes=7, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=1, concat=False, dropout=dropout)
        self.lin = torch.nn.Linear(hidden, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.lin(x)

# Training loop (use only train_mask nodes for loss):
model = GATv2NodeClassifier(in_channels=data.x.shape[1], num_classes=7)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # data.y is labels only for labeled_nodes, so:
    labeled_out = out[data.labeled_nodes]
    loss = F.cross_entropy(labeled_out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

**Key tricks:**
- Normalize features: `x = (x - x.mean(0)) / (x.std(0) + 1e-8)`
- Label smoothing: `F.cross_entropy(..., label_smoothing=0.1)`
- DropEdge: randomly drop some edges during training

---

### Graph B — Deep GATv2 / RevGAT (2 hour budget)

**Why deeper?** With 2 hours, you can afford a deeper, more expressive model. More layers = better ROC-AUC for binary classification. Use residual connections to avoid over-smoothing.

**Architecture:**
```
Input → Linear (project to 256) → [GATv2Conv (256, 4 heads) + Residual] x 4 → Linear → 1 output (logit)
```

**Implementation:**
```python
from torch_geometric.nn import GATv2Conv
import torch, torch.nn.functional as F

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
            x = norm(x + F.dropout(F.elu(conv(x, edge_index)), p=self.dropout, training=self.training))
        return self.out(x).squeeze(-1)  # raw logit

# Binary classification with ROC-AUC:
# Loss: BCEWithLogitsLoss (handles class imbalance better)
pos_weight = torch.tensor([(train_mask.sum() - pos_count) / pos_count])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Evaluate with:
from sklearn.metrics import roc_auc_score
probs = torch.sigmoid(model(data.x, data.edge_index)[data.labeled_nodes][data.val_mask])
auc = roc_auc_score(data.y[data.val_mask].cpu(), probs.detach().cpu())
```

**Key tricks:**
- Use `BCEWithLogitsLoss` with `pos_weight` for imbalanced binary labels
- Output raw logits, not sigmoid — `sklearn.roc_auc_score` works on logits too
- Save best val ROC-AUC checkpoint

---

### Graph C — BUDDY / NCN for Link Prediction (2 hour budget)

**Why BUDDY?** Hits@50 rewards models that correctly rank true edges above hard negatives. BUDDY precomputes structural features (common neighbors, Adamic-Adar) using hashing — this is much faster than SEAL (subgraph extraction) but nearly as accurate.

**Alternative: NCN (Neighbor Completion Network)** — more accurate but ~3x slower.

**Simpler baseline: GraphSAGE + dot product decoder**
```
Node embeddings via GraphSAGE → dot product for edge score → rank edges
```

**BUDDY-style Implementation (simplified):**
```python
from torch_geometric.nn import SAGEConv
import torch, torch.nn.functional as F

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden=256, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = dropout
        # MLP decoder for edge scoring (better than raw dot product)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def encode(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def decode(self, z, edges):
        # edges: [E, 2]
        src, dst = edges[:, 0], edges[:, 1]
        return self.decoder(torch.cat([z[src], z[dst]], dim=-1)).squeeze(-1)

# Training:
model = LinkPredictor(in_channels=data.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    pos_scores = model.decode(z, data.train_pos)
    neg_scores = model.decode(z, data.train_neg)
    loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_scores, neg_scores]),
        torch.cat([torch.ones(len(data.train_pos)), torch.zeros(len(data.train_neg))]).to(z.device)
    )
    loss.backward()
    optimizer.step()
```

**Evaluating Hits@50:**
```python
def hits_at_k(z, model, valid_pos, valid_neg, k=50):
    # valid_neg shape: [V, 500, 2]
    hits = 0
    for i in range(len(valid_pos)):
        pos_score = model.decode(z, valid_pos[i:i+1]).item()
        neg_scores = model.decode(z, valid_neg[i]).detach()  # [500]
        rank = (neg_scores >= pos_score).sum().item() + 1
        if rank <= k:
            hits += 1
    return hits / len(valid_pos)
```

**Key tricks:**
- Use MLP decoder over concatenated embeddings (better than dot product for Hits@K)
- Hard negative mining: model already provides 500 hard negatives per positive in `valid_neg`
- Try **structural features as additional input**: compute degree, common neighbors, Jaccard coefficient and concatenate to `x`

---

## Submission Structure

```
cs1230459/
├── Q1/
│   └── submission.py
└── Q2/
    ├── src/
    │   ├── load_dataset.py
    │   ├── predict.py
    │   ├── evaluate.py
    │   ├── train_A.py         ← your training code
    │   ├── train_B.py
    │   └── train_C.py
    └── requirements.txt
```

Models saved as: `cs1230459_model_A.pt`, `cs1230459_model_B.pt`, `cs1230459_model_C.pt`

---

## Quick Reference: Which Architecture for What?

| Scenario | Architecture | PyG Class |
|----------|-------------|-----------|
| Fast, accurate node classification | GATv2 | `GATv2Conv` |
| Deep node classification (avoid over-smoothing) | GCNII | `GCN2Conv` |
| Binary node classification (ROC-AUC) | Deep GATv2 + residual | `GATv2Conv` |
| Link prediction (fast, good Hits@K) | GraphSAGE + MLP decoder | `SAGEConv` |
| Link prediction (best accuracy) | BUDDY / NCN | external impl |

---

## Common Mistakes to Avoid

1. **Don't use `data.y` directly** — it's only for labeled nodes. Use `data.labeled_nodes` to index into node predictions.
2. **Don't softmax before cross-entropy** — `F.cross_entropy` applies softmax internally.
3. **For ROC-AUC** — output raw logits, not probabilities. `roc_auc_score` works on raw scores.
4. **For Hits@50** — rank the positive edge against ALL 500 hard negatives per query, not just random ones.
5. **Save best val checkpoint** — don't save the last epoch model; save when val metric peaks.
