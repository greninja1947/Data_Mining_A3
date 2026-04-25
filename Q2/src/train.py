"""
train.py -- unified training entry point for COL761 A3 evaluators.

Required evaluator command:
    python train.py --dataset A|B|C --task node|link \
        --data_dir /absolute/path/to/datasets \
        --model_dir /path/to/models \
        --kerberos YOUR_KERBEROS
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import train_C


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COL761 A3 training dispatcher")
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--time_limit", type=float, default=None)

    # Dataset A knobs. Defaults are filled after dataset dispatch.
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--wd", type=float, default=None)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--drop_edge", type=float, default=None)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default=None)

    # Dataset C knobs. Defaults are filled after dataset dispatch.
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--neg_mode", choices=["given", "random", "mixed"], default=None)
    parser.add_argument("--max_epochs", type=int, default=10_000)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true")
    return parser


def _validate_task(args: argparse.Namespace) -> None:
    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        raise SystemExit(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )


def _train_a(args: argparse.Namespace) -> None:
    from train_A import train as train_a

    train_a(
        SimpleNamespace(
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            kerberos=args.kerberos,
            time_limit=300.0 if args.time_limit is None else args.time_limit,
            hidden=16 if args.hidden is None else args.hidden,
            heads=4 if args.heads is None else args.heads,
            dropout=0.6 if args.dropout is None else args.dropout,
            lr=0.005 if args.lr is None else args.lr,
            wd=0.001 if args.wd is None else args.wd,
            t_max=1000 if args.t_max is None else args.t_max,
            drop_edge=0.2 if args.drop_edge is None else args.drop_edge,
            optimizer="adam" if args.optimizer is None else args.optimizer,
        )
    )


def _train_c(args: argparse.Namespace) -> None:
    c_argv: list[str] = [
        "--data_dir", args.data_dir,
        "--model_dir", args.model_dir,
        "--kerberos", args.kerberos,
        "--time_limit", str(7200.0 if args.time_limit is None else args.time_limit),
        "--hidden", str(train_C.DEFAULT_HIDDEN if args.hidden is None else args.hidden),
        "--num_layers", str(train_C.DEFAULT_NUM_LAYERS if args.num_layers is None else args.num_layers),
        "--dropout", str(train_C.DEFAULT_DROPOUT if args.dropout is None else args.dropout),
        "--lr", str(train_C.DEFAULT_LR if args.lr is None else args.lr),
        "--weight_decay", str(
            train_C.DEFAULT_WEIGHT_DECAY if args.weight_decay is None else args.weight_decay
        ),
        "--neg_mode", train_C.DEFAULT_NEG_MODE if args.neg_mode is None else args.neg_mode,
        "--max_epochs", str(args.max_epochs),
        "--eval_every", str(args.eval_every),
        "--k", str(args.k),
        "--seed", str(args.seed),
    ]
    if args.force_cpu:
        c_argv.append("--force_cpu")
    train_C.main(c_argv)


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_task(args)

    if args.dataset == "A":
        _train_a(args)
    elif args.dataset == "C":
        _train_c(args)
    else:
        raise SystemExit("Dataset B training is not implemented in this dispatcher.")


if __name__ == "__main__":
    main()

