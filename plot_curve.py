"""
plot_curve.py  –  Plot metric vs training time from a trainloop CSV.

Usage:
    python plot_curve.py --csv trainloop_results/A_results.csv \
                         --part A --out trainloop_results/A_curve.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


METRIC_LABEL = {"A": "Accuracy", "B": "ROC-AUC", "C": "Hits@50"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",  required=True)
    parser.add_argument("--part", required=True, choices=["A", "B", "C"])
    parser.add_argument("--out",  required=True)
    args = parser.parse_args()

    times, metrics = [], []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_seconds"]) / 60)   # convert to minutes
            metrics.append(float(row["metric"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, metrics, marker="o", linewidth=2, markersize=6, color="#2196F3")

    # annotate each point
    for t, m in zip(times, metrics):
        ax.annotate(f"{m:.4f}", (t, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("Training time (minutes)", fontsize=12)
    ax.set_ylabel(METRIC_LABEL[args.part], fontsize=12)
    ax.set_title(f"Dataset {args.part} — {METRIC_LABEL[args.part]} vs Training Time", fontsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
