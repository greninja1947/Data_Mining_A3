import argparse
import os
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from load_dataset import load_dataset
from models import GATv2NodeClassifier, NormalizedGATv2


def drop_edge(edge_index, drop_rate, training):
    if not training or drop_rate == 0.0:
        return edge_index
    mask = torch.rand(edge_index.size(1), device=edge_index.device) >= drop_rate
    return edge_index[:, mask]


def build_model(data, dataset, args):
    mean_x = data.x.mean(0).cpu()
    std_x = (data.x.std(0) + 1e-7).cpu()
    base = GATv2NodeClassifier(
        in_channels=data.x.shape[1],
        hidden=args.hidden,
        heads=args.heads,
        num_classes=dataset.num_classes,
        dropout=args.dropout,
    )
    return NormalizedGATv2(mean_x, std_x, base)


def build_optimizer(model, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


def evaluate_val(model, data, val_idx):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[val_idx].argmax(dim=1)
        val_acc = (pred == data.y[data.val_mask]).float().mean().item()
    return val_acc


def save_model(model, args):
    out_path = os.path.join(args.model_dir, f"{args.kerberos}_model_A.pt")
    torch.save(model, out_path)
    print(f"Model saved to {out_path}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dataset = load_dataset("A", args.data_dir)
    data = dataset[0].to(device)

    labeled_nodes = data.labeled_nodes
    val_idx = labeled_nodes[data.val_mask]
    train_idx = labeled_nodes[data.train_mask]

    model = build_model(data, dataset, args).to(device)
    optimizer = build_optimizer(model, args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)

    best_state = None
    best_val_acc = 0.0
    t0 = time.time()
    epoch = 0

    pbar = tqdm(
        total=int(args.time_limit),
        desc="Training A",
        unit="s",
        bar_format="{l_bar}{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}] {postfix}",
    )

    while True:
        elapsed = time.time() - t0
        if elapsed >= args.time_limit:
            break

        epoch += 1
        model.train()
        optimizer.zero_grad()
        ei = drop_edge(data.edge_index, args.drop_edge, model.training)
        out = model(data.x, ei)
        loss = F.cross_entropy(out[train_idx], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.n = min(elapsed, args.time_limit)
        pbar.refresh()

        if epoch % 10 == 0:
            val_acc = evaluate_val(model, data, val_idx)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            pbar.set_postfix(
                epoch=epoch,
                loss=f"{loss.item():.4f}",
                val=f"{val_acc:.4f}",
                best=f"{best_val_acc:.4f}",
            )

    pbar.n = args.time_limit
    pbar.refresh()
    pbar.close()
    print(f"\nTraining done. Best val acc: {best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.cpu()
    model.eval()
    save_model(model, args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--kerberos", default="cs1230459")
    parser.add_argument("--time_limit", type=float, default=300)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--wd", type=float, default=0.001)
    parser.add_argument("--t_max", type=int, default=1000)
    parser.add_argument("--drop_edge", type=float, default=0.2)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
    return parser


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
