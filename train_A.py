"""
train_A.py  –  GATv2 node classification for Dataset A (7-class, Accuracy)

Usage
-----
    python train_A.py --data_dir /path/to/public_datasets \
                      --model_dir /path/to/save/models \
                      --kerberos cs1230459 \
                      --time_limit 3600
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from load_dataset import load_dataset
from models import GATv2NodeClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds   = load_dataset("A", args.data_dir)
    data = ds[0].to(device)

    # Normalise features
    x_mean = data.x.mean(0)
    x_std  = data.x.std(0) + 1e-8
    data.x = (data.x - x_mean) / x_std

    num_classes = ds.num_classes
    # hidden=8 per head × 8 heads = 64-dim hidden (standard GAT/Cora config).
    # 256×8=2048-dim causes 6.9M params vs 140 training nodes → severe overfit.
    model = GATv2NodeClassifier(
        in_channels=data.x.shape[1],
        hidden=8,
        heads=8,
        num_classes=num_classes,
        dropout=0.6,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    labeled_nodes = data.labeled_nodes
    train_idx = labeled_nodes[data.train_mask]
    val_idx   = labeled_nodes[data.val_mask]

    best_val_acc = 0.0
    best_state   = None
    t0 = time.time()

    epoch = 0
    pbar = tqdm(total=int(args.time_limit), desc="Training A", unit="s",
                bar_format="{l_bar}{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}] {postfix}")

    while True:
        elapsed = time.time() - t0
        if elapsed >= args.time_limit:
            break

        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)                        # [N, C]
        loss = F.cross_entropy(out[train_idx], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch += 1

        # update bar to current elapsed seconds
        pbar.n = min(elapsed, args.time_limit)
        pbar.refresh()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                pred   = logits[val_idx].argmax(dim=1)
                val_acc = (pred == data.y[data.val_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}",
                             val=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")

    pbar.n = args.time_limit
    pbar.refresh()
    pbar.close()
    print(f"\nTraining done. Best val acc: {best_val_acc:.4f}")

    # Restore best weights for saving
    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    model.eval()

    os.makedirs(args.model_dir, exist_ok=True)
    out_path = os.path.join(args.model_dir, f"{args.kerberos}_model_A.pt")
    torch.save(model, out_path)
    print(f"Model saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--model_dir",  default="models")
    parser.add_argument("--kerberos",   default="cs1230459")
    parser.add_argument("--time_limit", type=float, default=3600,
                        help="Training time budget in seconds")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
