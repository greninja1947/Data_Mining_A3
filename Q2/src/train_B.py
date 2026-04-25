

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from load_dataset import load_dataset
from model_B import ResidualGNN  



def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_training_data(raw: Data) -> tuple[Data, torch.Tensor, torch.Tensor]:
    """Dense y over all N nodes (unlabeled = -1) for NeighborLoader."""
    n = int(raw.num_nodes)
    y_full = torch.full((n,), -1, dtype=torch.long)
    y_full[raw.labeled_nodes] = raw.y.long()

    train_idx = raw.labeled_nodes[raw.train_mask]
    val_idx   = raw.labeled_nodes[raw.val_mask]

    data = Data(
        x=raw.x.float(),
        edge_index=raw.edge_index.contiguous(),
        y=y_full,
        num_nodes=n,
    )
    return data, train_idx, val_idx


def compute_feature_stats(x: torch.Tensor,
                          sample_nodes: int = 200_000) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    idx = torch.randperm(n)[:sample_nodes] if n > sample_nodes else torch.arange(n)
    sample = x[idx].float()
    return sample.mean(dim=0), sample.std(dim=0)


@torch.no_grad()
def evaluate_auc(
    model: torch.nn.Module,
    data: Data,
    node_idx: torch.Tensor,
    device: torch.device,
    infer_batch_size: int,
    full_graph_eval: bool,
) -> float:
    model.eval()

    if full_graph_eval:
        x  = data.x.to(device, non_blocking=True)
        ei = data.edge_index.to(device, non_blocking=True)
        logits = model(x, ei).squeeze(-1)
        scores = logits[node_idx].detach().cpu().numpy()
        labels = data.y[node_idx].cpu().numpy()
        del x, ei, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return float(roc_auc_score(labels, scores))

    y_true_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []
    loader = NeighborLoader(
        data,
        num_neighbors=[25, 15],
        batch_size=infer_batch_size,
        input_nodes=node_idx,
        shuffle=False,
        num_workers=0,
    )
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out   = model(batch.x, batch.edge_index).squeeze(-1)
        seeds = out[: batch.batch_size]
        y_true_all.append(batch.y[: batch.batch_size].cpu().numpy())
        y_score_all.append(seeds.detach().cpu().numpy())

    return float(roc_auc_score(np.concatenate(y_true_all), np.concatenate(y_score_all)))



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--model_dir",  required=True)
    p.add_argument("--kerberos",   default="cs1230216")
    p.add_argument("--time_limit", type=float, default=7200.0)

    p.add_argument("--conv_type",  choices=["sage", "gatv2"], default="sage")
    p.add_argument("--hidden",     type=int,   default=256)
    p.add_argument("--num_layers", type=int,   default=4)
    p.add_argument("--heads",      type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.3)

    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=0.0)
    p.add_argument("--batch_size",      type=int,   default=2048)
    p.add_argument("--eval_batch_size", type=int,   default=8192)
    p.add_argument("--num_neighbors",   type=int, nargs="+", default=[15, 10, 10, 10])
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--eval_every",      type=int,   default=1)
    p.add_argument("--max_epochs",      type=int,   default=10_000)

    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--force_cpu", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.force_cpu)
    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, f"{args.kerberos}_model_B.pt")

    t0 = time.time()
    print(f"[setup] device={device}  time_budget={args.time_limit:.0f}s", flush=True)
    print(f"[setup] loading dataset B ...", flush=True)

    ds  = load_dataset("B", args.data_dir)
    raw = ds[0]
    print(f"[setup] loaded in {time.time()-t0:.1f}s: nodes={raw.num_nodes:,} "
          f"edges={raw.num_edges:,} feat_dim={raw.x.shape[1]}", flush=True)

    data, train_idx, val_idx = build_training_data(raw)
    del raw
    print(f"[setup] train_nodes={train_idx.numel():,}  val_nodes={val_idx.numel():,}",
          flush=True)


    train_labels = data.y[train_idx].float()
    pos = float(train_labels.sum())
    neg = float(train_labels.numel() - pos)
    pos_weight_val = neg / max(pos, 1.0)
    print(f"[setup] pos={int(pos):,}  neg={int(neg):,}  pos_weight={pos_weight_val:.3f}",
          flush=True)

    mean, std = compute_feature_stats(data.x)

    fanout = list(args.num_neighbors)
    if len(fanout) < args.num_layers:
        fanout += [fanout[-1]] * (args.num_layers - len(fanout))
    fanout = fanout[: args.num_layers]

    train_loader = NeighborLoader(
        data,
        num_neighbors=fanout,
        batch_size=args.batch_size,
        input_nodes=train_idx,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    model = ResidualGNN(
        in_channels=data.x.shape[1],
        hidden=args.hidden,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        conv_type=args.conv_type,
        feature_mean=mean,
        feature_std=std,
    ).to(device)
    print(f"[setup] model={args.conv_type}  hidden={args.hidden}  "
          f"layers={args.num_layers}  params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion  = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc   = -1.0
    best_epoch = -1
    full_graph_eval = (device.type == "cuda")

    def budget_left() -> float:
        return args.time_limit - (time.time() - t0)

    def save_best(auc: float, epoch: int) -> None:
        nonlocal best_auc, best_epoch
        best_auc, best_epoch = auc, epoch
        torch.save(model.to("cpu"), save_path)
        model.to(device)
        print(f"[ckpt] saved  AUC={auc:.4f}  ->  {save_path}", flush=True)

    epoch = 0
    stopped_reason = "max_epochs"
    try:
        while epoch < args.max_epochs:
            if budget_left() <= 0:
                stopped_reason = "time_limit"
                break
            epoch += 1
            model.train()
            t_ep = time.time()
            total_loss, total_n = 0.0, 0

            for batch in train_loader:
                if budget_left() <= 0:
                    stopped_reason = "time_limit"
                    break
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad()
                out    = model(batch.x, batch.edge_index).squeeze(-1)
                logits = out[: batch.batch_size]
                target = batch.y[: batch.batch_size].float()
                loss   = criterion(logits, target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * batch.batch_size
                total_n    += batch.batch_size

            print(f"[epoch {epoch:04d}] loss={total_loss/max(total_n,1):.4f}  "
                  f"ep={time.time()-t_ep:.1f}s  "
                  f"elapsed={time.time()-t0:.1f}s/{args.time_limit:.0f}s",
                  flush=True)

            if stopped_reason == "time_limit":
                break

            if epoch % args.eval_every == 0 or budget_left() <= 0:
                try:
                    auc = evaluate_auc(model, data, val_idx, device,
                                       args.eval_batch_size, full_graph_eval)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    full_graph_eval = False
                    auc = evaluate_auc(model, data, val_idx, device,
                                       args.eval_batch_size, False)
                print(f"[val ] AUC={auc:.4f}  best={best_auc:.4f}@{best_epoch}",
                      flush=True)
                if auc > best_auc:
                    save_best(auc, epoch)

    except KeyboardInterrupt:
        stopped_reason = "keyboard_interrupt"

    if best_auc < 0:
        torch.save(model.to("cpu"), save_path)
        print(f"[ckpt] no val AUC seen — saved current weights to {save_path}", flush=True)

    print(f"[done] {stopped_reason}  epochs={epoch}  "
          f"best_val_auc={best_auc:.4f}@{best_epoch}  "
          f"wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
