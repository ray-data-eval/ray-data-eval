"""Plots Figure 7a.

Usage:
    python plots/rag.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_NAME = "Ray Data"
FONT_SIZE = 24

STAGES = [("Encoding", "encoding_s", "#b2df8a"),
          ("Retrieval", "retrieval_s", "#fdbf6f"),
          ("Generation", "generation_s", "#fb9a99")]


def read(path):
    """Read a results CSV.

    Returns a list of rows with float values.
    """
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [{k: float(v) for k, v in row.items() if v != ""}
                for row in csv.DictReader(f)]


def main():
    """Draw Figure 7a."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=os.path.join(ROOT, "results-archive"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    args = p.parse_args()

    src = os.path.join(args.results, "rag")
    rows = sorted(read(os.path.join(src, "ray_data_dynamic.csv"))
                  or read(os.path.join(src, "ray_data_static.csv")),
                  key=lambda r: r["num_gpus"])
    staged = read(os.path.join(src, "ray_data_staged.csv"))
    if not rows:
        sys.exit(f"no ray_data_dynamic.csv or ray_data_static.csv under {src}")

    gpu_labels = [f"{int(r['num_gpus'])} GPU" for r in rows]
    gpu_times = [r["jct_min"] for r in rows]
    gpu_labels[0] = f"1 GPU\n{SYSTEM_NAME}\n(dynamic)"

    labels = [f"1 GPU\n{SYSTEM_NAME}\n(staged)"] + gpu_labels
    bar_positions = list(range(len(labels)))

    plt.figure(figsize=(10, 7))

    # Stacked bar for the staged baseline.
    staged_row = staged[0] if staged else {}
    staged_total = staged_row.get("jct_min", 0.0)
    staged_bottom = 0.0
    for stage, col, color in STAGES:
        if col not in staged_row:
            continue
        t = staged_row[col] / 60
        plt.bar(0, t, bottom=staged_bottom, color=color, label=stage)
        plt.text(0, staged_bottom + t / 2, f"{t:.1f}", ha="center", va="center",
                 fontsize=FONT_SIZE - 1)
        staged_bottom += t
    if staged_row and staged_bottom == 0:
        plt.bar(0, staged_total, color="#b2df8a", label=f"{SYSTEM_NAME} (staged)")
    elif staged_bottom and abs(staged_bottom - staged_total) > 0.1:
        print(f"  warning: stages sum to {staged_bottom:.1f} min but the "
              f"logged total is {staged_total:.1f} min")

    plt.bar(bar_positions[1:], gpu_times, color="#A6CEE3",
            label=f"{SYSTEM_NAME} (dynamic)")
    plt.axvline(x=1.5, color="black", linestyle="--", linewidth=1)

    top = max([staged_total] + gpu_times)
    plt.ylim(0, top * 1.1)
    plt.yticks(fontsize=FONT_SIZE)
    plt.xticks(bar_positions, labels, fontsize=FONT_SIZE - 5)
    plt.ylabel("Job Completion Time (minutes)", fontsize=FONT_SIZE)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, t in enumerate([staged_total] + gpu_times):
        plt.text(i, t + 0.3, f"{t:.1f}", ha="center", va="bottom", fontsize=FONT_SIZE)

    plt.legend(fontsize=FONT_SIZE - 2)
    plt.tight_layout()
    save(plt, args.outdir, "rag")

    print()
    for lbl, t in zip(labels, [staged_total] + gpu_times):
        flat = lbl.replace("-\n", "-").replace(chr(10), " ")
        print(f"  {flat:22s} {t:7.1f} min")


if __name__ == "__main__":
    main()
