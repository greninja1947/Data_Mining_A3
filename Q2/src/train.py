from __future__ import annotations

import argparse
from types import SimpleNamespace


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
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
            time_limit=2700.0,
            hidden=16,
            heads=4,
            dropout=0.6,
            lr=0.005,
            wd=0.001,
            t_max=1000,
            drop_edge=0.2,
            optimizer="adam",
        )
    )


def _train_b(args: argparse.Namespace) -> None:
    import train_B

    b_argv: list[str] = [
        "--data_dir",  args.data_dir,
        "--model_dir", args.model_dir,
        "--kerberos",  args.kerberos,
        "--time_limit", "3600.0",
    ]
    train_B.main(b_argv)


def _train_c(args: argparse.Namespace) -> None:
    import train_C

    c_argv: list[str] = [
        "--data_dir", args.data_dir,
        "--model_dir", args.model_dir,
        "--kerberos", args.kerberos,
        "--time_limit", "5400.0",
    ]
    train_C.main(c_argv)


def main() -> None:
    args = build_arg_parser().parse_args()
    _validate_task(args)

    if args.dataset == "A":
        _train_a(args)
    elif args.dataset == "B":
        _train_b(args)
    elif args.dataset == "C":
        _train_c(args)


if __name__ == "__main__":
    main()
