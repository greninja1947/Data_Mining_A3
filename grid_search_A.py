"""
grid_search_A.py  –  Hyperparameter grid search for Dataset A (GATv2)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  EDIT THESE to control the search:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Grid definition ───────────────────────────────────────────────────────────
#   Each list is [low, ..., high] with whatever steps you want.

HIDDEN_SIZES   = [8,16,32]       # per-head hidden dim    (params ∝ hidden × heads)
HEAD_COUNTS    = [4,8,16]            # number of attention heads
LEARNING_RATES = [0.001, 0.005, 0.01]  # Adam lr
DROPOUTS       = [0.4, 0.6]       # dropout probability
WEIGHT_DECAYS  = [5e-4, 1e-3]     # L2 regularisation

# ── Training config ───────────────────────────────────────────────────────────
TRAIN_TIME_SEC   = 180    # seconds per run  (3 min default)
STAGNATION_SEC   = 40     # seconds without val improvement → early stop
EVAL_EVERY       = 10     # evaluate every N epochs

# ── Test-mode overrides (used with --test flag) ───────────────────────────────
TEST_HIDDEN_SIZES   = [8, 16]
TEST_HEAD_COUNTS    = [4]
TEST_LEARNING_RATES = [0.005]
TEST_DROPOUTS       = [0.6]
TEST_WEIGHT_DECAYS  = [5e-4]
TEST_TRAIN_TIME_SEC = 30    # 30 s per run → finishes in ~1 min

# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import itertools
import os
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from load_dataset import load_dataset
from models import GATv2NodeClassifier


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "grid_search_results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "A_grid_results.csv")

CSV_FIELDS = [
    "hidden", "heads", "lr", "dropout", "weight_decay",
    "num_params", "best_val_acc", "best_epoch",
    "epochs_run", "time_taken_s",
    "stagnated", "stagnation_at_s",
]


# ─────────────────────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def run_one(data, labeled_nodes, train_idx, val_idx,
            hidden, heads, lr, dropout, weight_decay,
            train_time, stagnation_sec, eval_every):

    torch.manual_seed(42)
    model = GATv2NodeClassifier(
        in_channels=data.x.shape[1],
        hidden=hidden,
        heads=heads,
        num_classes=int(data.y.max().item()) + 1,
        dropout=dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = count_params(model)

    best_val_acc   = 0.0
    best_epoch     = 0
    last_improved  = time.time()
    stagnated      = False
    stagnation_at  = None
    epoch          = 0
    t0             = time.time()

    pbar = tqdm(
        total=int(train_time),
        desc=f"  h={hidden} H={heads} lr={lr} do={dropout} wd={weight_decay}",
        unit="s",
        bar_format="{l_bar}{bar}| {n:.0f}/{total}s [{elapsed}<{remaining}] {postfix}",
        leave=False,
    )

    while True:
        elapsed = time.time() - t0
        if elapsed >= train_time:
            break

        # ── early stop on stagnation ──────────────────────────────────────
        idle = time.time() - last_improved
        if idle >= stagnation_sec:
            stagnated     = True
            stagnation_at = round(elapsed, 1)
            pbar.set_postfix_str(f"STAGNATED at {stagnation_at}s | best={best_val_acc:.4f}")
            pbar.n = elapsed
            pbar.refresh()
            break

        model.train()
        opt.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[data.train_mask])
        loss.backward()
        opt.step()
        epoch += 1

        pbar.n = min(elapsed, train_time)
        pbar.refresh()

        if epoch % eval_every == 0:
            model.eval()
            with torch.no_grad():
                pred    = model(data.x, data.edge_index)[val_idx].argmax(1)
                val_acc = (pred == data.y[data.val_mask]).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc  = val_acc
                best_epoch    = epoch
                last_improved = time.time()

            pbar.set_postfix(ep=epoch, val=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}",
                             idle=f"{time.time()-last_improved:.0f}s")

    pbar.n = int(train_time)
    pbar.refresh()
    pbar.close()

    return {
        "hidden":         hidden,
        "heads":          heads,
        "lr":             lr,
        "dropout":        dropout,
        "weight_decay":   weight_decay,
        "num_params":     n_params,
        "best_val_acc":   round(best_val_acc, 4),
        "best_epoch":     best_epoch,
        "epochs_run":     epoch,
        "time_taken_s":   round(time.time() - t0, 1),
        "stagnated":      stagnated,
        "stagnation_at_s": stagnation_at if stagnated else "no",
    }


def notify(message):
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "Grid Search A" sound name "Glass"'],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(__file__), "public_datasets"))
    parser.add_argument("--test", action="store_true", help="Quick smoke-test with tiny grid")
    args = parser.parse_args()

    # ── select grid ───────────────────────────────────────────────────────────
    if args.test:
        hidden_list   = TEST_HIDDEN_SIZES
        head_list     = TEST_HEAD_COUNTS
        lr_list       = TEST_LEARNING_RATES
        dropout_list  = TEST_DROPOUTS
        wd_list       = TEST_WEIGHT_DECAYS
        train_time    = TEST_TRAIN_TIME_SEC
        print("[ TEST MODE — small grid ]")
    else:
        hidden_list   = HIDDEN_SIZES
        head_list     = HEAD_COUNTS
        lr_list       = LEARNING_RATES
        dropout_list  = DROPOUTS
        wd_list       = WEIGHT_DECAYS
        train_time    = TRAIN_TIME_SEC

    combos = list(itertools.product(hidden_list, head_list, lr_list, dropout_list, wd_list))
    total  = len(combos)
    est_min = total * train_time / 60
    print(f"\nGrid search: {total} combinations × up to {train_time}s each")
    print(f"Max total time: ~{est_min:.0f} min  (early stops will cut this down)\n")

    # ── load data once ────────────────────────────────────────────────────────
    print("Loading dataset A ...")
    ds   = load_dataset("A", args.data_dir)
    data = ds[0]
    data.x = (data.x - data.x.mean(0)) / (data.x.std(0) + 1e-8)

    labeled    = data.labeled_nodes
    train_idx  = labeled[data.train_mask]
    val_idx    = labeled[data.val_mask]

    # ── run grid ──────────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for i, (hidden, heads, lr, dropout, wd) in enumerate(combos, 1):
        n_params_est = hidden * heads * data.x.shape[1] * 2   # rough estimate
        print(f"\n[{i:>3}/{total}] hidden={hidden} heads={heads} lr={lr} "
              f"dropout={dropout} wd={wd}  (~{n_params_est:,} params est.)")

        row = run_one(
            data, labeled, train_idx, val_idx,
            hidden=hidden, heads=heads, lr=lr,
            dropout=dropout, weight_decay=wd,
            train_time=train_time,
            stagnation_sec=STAGNATION_SEC,
            eval_every=EVAL_EVERY,
        )
        results.append(row)

        status = f"stagnated@{row['stagnation_at_s']}s" if row["stagnated"] else "ran full time"
        print(f"  → best_val={row['best_val_acc']:.4f}  "
              f"epochs={row['epochs_run']}  params={row['num_params']:,}  [{status}]")

    # ── save CSV ──────────────────────────────────────────────────────────────
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    # ── print leaderboard ─────────────────────────────────────────────────────
    results.sort(key=lambda r: r["best_val_acc"], reverse=True)

    print(f"\n{'='*70}")
    print(f"  TOP 5 CONFIGS  (full results → {RESULTS_CSV})")
    print(f"{'='*70}")
    print(f"  {'Rank':<5} {'val_acc':<9} {'hidden':<8} {'heads':<7} "
          f"{'lr':<8} {'dropout':<9} {'wd':<8} {'params':<10} {'stagnated'}")
    print(f"  {'-'*65}")
    for rank, r in enumerate(results[:5], 1):
        stag = f"yes@{r['stagnation_at_s']}s" if r["stagnated"] else "no (ran full)"
        print(f"  {rank:<5} {r['best_val_acc']:<9.4f} {r['hidden']:<8} {r['heads']:<7} "
              f"{r['lr']:<8} {r['dropout']:<9} {r['weight_decay']:<8} "
              f"{r['num_params']:<10,} {stag}")

    not_stagnated = [r for r in results if not r["stagnated"]]
    if not_stagnated:
        print(f"\n  ⚠  {len(not_stagnated)} config(s) did NOT stagnate within {train_time}s "
              f"— may benefit from longer training:")
        for r in not_stagnated:
            print(f"     hidden={r['hidden']} heads={r['heads']} lr={r['lr']} "
                  f"dropout={r['dropout']} wd={r['weight_decay']}  "
                  f"best={r['best_val_acc']:.4f}")

    print(f"\nResults saved to {RESULTS_CSV}")

    notify(f"Done! {total} runs. Best val_acc={results[0]['best_val_acc']:.4f} "
           f"(h={results[0]['hidden']} H={results[0]['heads']} lr={results[0]['lr']})")


if __name__ == "__main__":
    main()
