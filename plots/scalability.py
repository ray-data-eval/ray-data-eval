"""Plots Figure 10b.

Usage:
    python plots/scalability.py
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

# (file, label, marker),
SERIES = [
    ("ray_data.csv", SYSTEM_NAME, "o"),
    ("ray.csv", "Ray", "s"),
    ("ray_streaming.csv", "Ray Generator", "^"),
]
NODE_TICKS = [1, 2, 4, 8, 16, 32]


def read(path):
    """Read a results CSV.

    Returns ([nodes], [GB/s]).
    """
    with open(path) as f:
        rows = sorted((float(r["num_nodes"]), float(r["throughput_gbs"]))
                      for r in csv.DictReader(f))
    return [n for n, _ in rows], [g for _, g in rows]


def main():
    """Draw Figure 10b."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=os.path.join(ROOT, "results-archive"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    args = p.parse_args()

    src = os.path.join(args.results, "scalability")

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 16
    plt.rcParams["text.usetex"] = False
    plt.figure(figsize=(5, 4))

    drawn = []
    for fname, label, marker in SERIES:
        path = os.path.join(src, fname)
        if not os.path.isfile(path):
            print(f"  skip {label}: no {fname}")
            continue
        nodes, tput = read(path)
        plt.plot(nodes, tput, marker=marker, label=label)
        drawn.append((label, nodes, tput))

    if not drawn:
        sys.exit(f"no scalability CSVs under {src}")

    plt.xscale("log", base=2)
    plt.xticks(NODE_TICKS, NODE_TICKS)
    plt.yticks([10, 20, 30, 40, 50, 60])
    plt.xlabel("Number of Nodes")
    plt.ylabel("Throughput (GB/s)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    save(plt, args.outdir, "scalability")

    print(f"\n  {'system':16s}" + "".join(f"{int(n):>8d}" for n in drawn[0][1]))
    for label, _, tput in drawn:
        print(f"  {label:16s}" + "".join(f"{g:8.1f}" for g in tput))
    print("  (GB/s, against number of nodes)")

    if len(drawn) > 1:
        top_nodes = drawn[0][1][-1]
        ray_data_top = drawn[0][2][-1]
        best_other = max(t[-1] for _, _, t in drawn[1:])
        print(f"\n  at {int(top_nodes)} nodes Ray Data is "
              f"{ray_data_top / best_other:.2f}x the best baseline")


if __name__ == "__main__":
    main()
