"""Plots Figure 9.

Usage:
    python plots/memory_pipelining.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import use_style, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Section 5.3.1: (160 tasks x 5s + 800 rows x 0.5s) / 8 vCPUs.
OPTIMAL_S = 150

SYSTEM_NAME = "Ray Data"

# (file, row label), top to bottom as the figure lays them out.
SYSTEMS = [
    ("spark.csv", "Spark"),
    ("flink.csv", "Flink"),
    ("tfdata.csv", "TFData"),
    ("ray_data_no_part.csv", f"{SYSTEM_NAME}(-Part.)"),
    ("ray_data_no_adapt.csv", f"{SYSTEM_NAME}(-Adapt.)"),
    ("ray_data.csv", SYSTEM_NAME),
]


def read(path):
    """Read a results CSV.

    Returns {memory limit in GB: seconds}, omitting runs that OOMed.
    """
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return {int(r["mem_limit_gb"]): float(r["jct_s"])
                for r in csv.DictReader(f) if r.get("jct_s")}


def main():
    """Draw Figure 9."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=os.path.join(ROOT, "results-archive"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    args = p.parse_args()

    src = os.path.join(args.results, "memory_pipelining")
    rows, labels = [], []
    for fname, label in SYSTEMS:
        by_limit = read(os.path.join(src, fname))
        if not by_limit:
            print(f"  skip {label}: no {fname}")
            continue
        rows.append(by_limit)
        labels.append(label)

    if not rows:
        sys.exit(f"no results under {src}\n"
                 f"  the benchmark is experiments/ray_data_eval/microbenchmarks/"
                 f"memory_pipelining/raydata/producer_consumer_gpu.py")

    limits = sorted({m for r in rows for m in r})
    # NaN where a system OOMed; the figure draws those cells grey.
    grid = np.array([[r.get(m, np.nan) for m in limits] for r in rows])

    use_style(figratio=0.62, size=10, width=4.2)
    import seaborn as sns
    _, ax = plt.subplots()
    sns.heatmap(grid, ax=ax, cmap="RdYlGn_r", annot=False, linewidths=0.6,
                linecolor="white", xticklabels=[str(m) for m in limits],
                yticklabels=labels,
                cbar_kws={"label": "Job Completion Time (s)"})
    ax.set_facecolor("lightgrey")          # OOM cells

    # White text on the dark ends of the scale, black in the middle.
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    for i, j in np.ndindex(grid.shape):
        if np.isnan(grid[i, j]):
            continue
        frac = (grid[i, j] - lo) / (hi - lo)
        ax.text(j + 0.5, i + 0.5, f"{grid[i, j]:.0f}", ha="center", va="center",
                fontsize=9, color="white" if frac < 0.15 or frac > 0.75 else "black")

    ax.set_xlabel("Memory Limit (GB)")
    ax.tick_params(axis="y", rotation=0, length=3)
    ax.tick_params(axis="x", length=0)
    plt.tight_layout()
    save(plt, args.outdir, "memory_pipelining")

    print(f"\n  optimal job completion time {OPTIMAL_S}s\n")
    print(f"  {'system':16s} {'best':>7s} {'x optimal':>10s}   lowest limit it survives")
    for label, by_limit in zip(labels, rows):
        best = min(by_limit.values())
        print(f"  {label:16s} {best:6.0f}s {best / OPTIMAL_S:9.2f}x   {min(by_limit)} GB")

    ray_data = dict(zip(labels, rows)).get(SYSTEM_NAME)
    adapt = dict(zip(labels, rows)).get(f"{SYSTEM_NAME}(-Adapt.)")
    if ray_data and adapt:
        worse = [(m, adapt[m] / ray_data[m] - 1) for m in sorted(ray_data) if m in adapt]
        lo, hi = min(w for _, w in worse), max(w for _, w in worse)
        print(f"\n  Ray Data(-Adapt.) is {lo*100:.0f}-{hi*100:.0f}% worse than Ray Data "
              f"(section 5.3.1 says 10-88%)")


if __name__ == "__main__":
    main()
