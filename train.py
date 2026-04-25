"""
train.py  –  COL761 Assignment 3 unified training entry point.

Train command (as required by evaluators):
    python train.py --dataset A|B|C --task node|link \
        --data_dir /absolute/path/to/datasets \
        --model_dir /path/to/models \
        --kerberos YOUR_KERBEROS \
        [--time_limit SECONDS]
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="COL761 A3 training dispatcher")
    parser.add_argument("--dataset",    required=True, choices=["A", "B", "C"])
    parser.add_argument("--task",       required=True, choices=["node", "link"])
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--model_dir",  default="models")
    parser.add_argument("--kerberos",   default="cs1230459")
    parser.add_argument("--time_limit", type=float, default=300)
    args = parser.parse_args()

    # validate task <-> dataset pairing
    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    if args.dataset == "A":
        from train_A import train as train_fn
        import types
        train_args = types.SimpleNamespace(
            data_dir    = args.data_dir,
            model_dir   = args.model_dir,
            kerberos    = args.kerberos,
            time_limit  = args.time_limit,
            hidden      = 16,
            heads       = 4,
            dropout     = 0.6,
            lr          = 0.005,
            wd          = 0.001,
            t_max       = 1000,
            drop_edge   = 0.2,
            optimizer   = "adam",
        )
        train_fn(train_args)

    elif args.dataset == "B":
        raise NotImplementedError("Dataset B training not yet implemented.")

    elif args.dataset == "C":
        raise NotImplementedError("Dataset C training not yet implemented.")


if __name__ == "__main__":
    main()
