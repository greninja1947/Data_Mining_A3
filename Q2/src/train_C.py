"""
train_C.py -- trainer for COL761 A3 Q2 Dataset C (link prediction, Hits@50).

The training loop is controlled by a wall-clock budget and keeps the best
validation Hits@50 checkpoint at <kerberos>_model_C.pt.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from load_dataset import load_dataset
from model_C import LinkPredictor 


DEFAULT_HIDDEN = 256
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_NEG_MODE = "mixed"




def pick_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_random_neg(num_nodes: int, num_samples: int,
                      pos_set: set, device: torch.device) -> torch.Tensor:
    out = torch.empty((num_samples, 2), dtype=torch.long)
    i = 0
    while i < num_samples:
        remaining = num_samples - i
        candidates = torch.randint(0, num_nodes, (int(remaining * 1.5) + 8, 2))
        for u, v in candidates.tolist():
            if u == v or (u, v) in pos_set or (v, u) in pos_set:
                continue
            out[i] = torch.tensor([u, v])
            i += 1
            if i >= num_samples:
                break
    return out.to(device)


def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int) -> float:
    n_neg_higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return (n_neg_higher < k).float().mean().item()


@torch.no_grad()
def evaluate_val(model: LinkPredictor, x: torch.Tensor, edge_index: torch.Tensor,
                 valid_pos: torch.Tensor, valid_neg: torch.Tensor,
                 device: torch.device, k: int = 50,
                 chunk: int = 5000) -> tuple[float, float]:
    model.eval()
    z = model.encode(x, edge_index)

    pos_scores = model.decode(z, valid_pos.to(device))  # [P]
    P, K, _ = valid_neg.shape
    flat = valid_neg.view(P * K, 2).to(device)
    neg_scores_parts = []
    for i in range(0, flat.shape[0], chunk):
        neg_scores_parts.append(model.decode(z, flat[i:i + chunk]))
    neg_scores = torch.cat(neg_scores_parts).view(P, K)

    h = hits_at_k(pos_scores, neg_scores, k)
    return float(h), float(pos_scores.mean().item())



def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--kerberos", default="cs1230216")
    p.add_argument("--time_limit", type=float, default=60.0)

    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    p.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)

    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--max_epochs", type=int, default=10_000)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--neg_mode", choices=["given", "random", "mixed"], default=DEFAULT_NEG_MODE)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--patience_sec", type=float, default=0.0)
    p.add_argument("--min_improve", type=float, default=0.0)
    p.add_argument("--log_every", type=int, default=0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_cpu", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.force_cpu)
    os.makedirs(args.model_dir, exist_ok=True)
    save_path = os.path.join(args.model_dir, f"{args.kerberos}_model_C.pt")

    t0 = time.time()
    print(f"[setup] device={device}  time_budget={args.time_limit:.0f}s", flush=True)
    print(f"[setup] loading dataset C ...", flush=True)
    ds = load_dataset("C", args.data_dir)
    print(f"[setup] loaded in {time.time() - t0:.1f}s: {ds}", flush=True)

    if not hasattr(ds, "x"):
        raise RuntimeError(
            "Dataset C has no `x`. Expected a `gnn_feature` file in the C/ directory."
        )

    x = ds.x.float()
    edge_index = ds.edge_index.long()
    num_nodes = int(x.shape[0])

    train_pos = ds.train_pos.long()
    train_neg = ds.train_neg.long() if hasattr(ds, "train_neg") else None
    valid_pos = ds.valid_pos.long()
    valid_neg = ds.valid_neg.long()  # [V, 500, 2]

    print(f"[setup] nodes={num_nodes:,}  feat_dim={x.shape[1]}  "
          f"edges={edge_index.shape[1]:,}", flush=True)
    print(f"[setup] train_pos={train_pos.shape[0]}  "
          f"train_neg={0 if train_neg is None else train_neg.shape[0]}  "
          f"valid_pos={valid_pos.shape[0]}  valid_neg={valid_neg.shape[0]}x{valid_neg.shape[1]}",
          flush=True)

    mean = x.mean(dim=0)
    std = x.std(dim=0)

    x_dev = x.to(device)
    ei_dev = edge_index.to(device)
    train_pos_dev = train_pos.to(device)
    train_neg_dev = train_neg.to(device) if train_neg is not None else None

    pos_set = set()
    for u, v in train_pos.tolist():
        pos_set.add((u, v))

    model = LinkPredictor(
        in_channels=x.shape[1],
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        feature_mean=mean,
        feature_std=std,
    ).to(device)
    print(f"[setup] model=SAGE+MLP  hidden={args.hidden}  layers={args.num_layers}  "
          f"params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    if args.patience_sec > 0:
        print(
            f"[setup] early_stop patience={args.patience_sec:.0f}s  "
            f"min_improve={args.min_improve:g}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_hits = -1.0
    best_epoch = -1
    last_improve_t: float | None = None
    last_hits: float | None = None

    def budget_left() -> float:
        return args.time_limit - (time.time() - t0)

    def save_best(h: float, epoch: int) -> None:
        nonlocal best_hits, best_epoch
        best_hits = h
        best_epoch = epoch
        model_cpu = model.to("cpu")
        torch.save(model_cpu, save_path)
        model.to(device)

    def update_progress(pbar: tqdm, loss_value: float | None = None) -> None:
        pbar.n = min(time.time() - t0, args.time_limit)
        postfix = {"epoch": epoch}
        if loss_value is not None:
            postfix["loss"] = f"{loss_value:.4f}"
        if last_hits is not None:
            postfix[f"H@{args.k}"] = f"{last_hits:.4f}"
        if best_hits >= 0:
            postfix["best"] = f"{best_hits:.4f}@{best_epoch}"
        pbar.set_postfix(postfix)

    epoch = 0
    stopped_reason = "max_epochs"
    pbar = tqdm(
        total=max(1, int(args.time_limit)),
        desc="Training C",
        unit="s",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}] {postfix}",
    )
    try:
        while epoch < args.max_epochs:
            if budget_left() <= 0:
                stopped_reason = "time_limit"
                break
            if (
                args.patience_sec > 0
                and last_improve_t is not None
                and (time.time() - last_improve_t) >= args.patience_sec
            ):
                stopped_reason = "no_improve_patience"
                break
            epoch += 1
            model.train()
            optimizer.zero_grad()

            z = model.encode(x_dev, ei_dev)

            pos_scores = model.decode(z, train_pos_dev)
            neg_edges_list = []
            if args.neg_mode in ("given", "mixed") and train_neg_dev is not None:
                neg_edges_list.append(train_neg_dev)
            if args.neg_mode in ("random", "mixed"):
                sampled = sample_random_neg(
                    num_nodes, train_pos.shape[0], pos_set, device
                )
                neg_edges_list.append(sampled)
            if not neg_edges_list:
                sampled = sample_random_neg(
                    num_nodes, train_pos.shape[0], pos_set, device
                )
                neg_edges_list.append(sampled)
            neg_edges = torch.cat(neg_edges_list, dim=0)
            neg_scores = model.decode(z, neg_edges)

            logits = torch.cat([pos_scores, neg_scores])
            target = torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores),
            ])
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            if epoch % args.eval_every == 0 or budget_left() <= 0:
                hits, _ = evaluate_val(
                    model, x_dev, ei_dev, valid_pos, valid_neg, device,
                    k=args.k,
                )
                last_hits = hits
                if hits > (best_hits + args.min_improve):
                    save_best(hits, epoch)
                    last_improve_t = time.time()
                update_progress(pbar, loss.item())
            elif args.log_every > 0 and epoch % args.log_every == 0:
                update_progress(pbar, loss.item())
    except KeyboardInterrupt:
        stopped_reason = "keyboard_interrupt"
    finally:
        pbar.n = min(args.time_limit, time.time() - t0)
        pbar.close()

    if best_hits < 0:
        print("[ckpt] no validation Hits measured; saving current weights as fallback",
              flush=True)
        torch.save(model.to("cpu"), save_path)
        model.to(device)

    print(f"[done] stop_reason={stopped_reason}  epochs={epoch}  "
          f"best_val_hits@{args.k}={best_hits:.4f}@{best_epoch}  "
          f"wall_time={time.time() - t0:.1f}s  model={save_path}", flush=True)


if __name__ == "__main__":
    main()
