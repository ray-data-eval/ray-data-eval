"""Plots Figure 10a.

Usage:
    python plots/partitioning.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import use_style, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOTAL_ROWS = 8192


def read_archive(path):
    """Read a results CSV.

    Returns [(partition size, duration), ...].
    """
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            size = row.get("partition_size_mb") or row.get("num_rows_in_block")
            dur = row.get("duration_s") or row.get("jct_s")
            if size and dur:
                out.append((float(size), float(dur)))
    return sorted(out)


def main():
    """Draw Figure 10a."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=os.path.join(ROOT, "results-archive"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    args = p.parse_args()

    path = os.path.join(args.results, "partitioning", "ray_data.csv")
    data = read_archive(path)
    if not data:
        sys.exit(f"no results at {path}\n"
                 f"  the benchmark is experiments/ray_data_eval/"
                 f"microbenchmarks/partitioning/raydata.py")

    num_rows = [d[0] for d in data]
    throughput = [TOTAL_ROWS / d[1] for d in data]

    use_style(figratio=3 / 4, size=14, width=4)
    plt.figure()
    ax = plt.gca()
    plt.bar(num_rows, throughput, width=[n * 0.5 for n in num_rows])

    ax.set_xscale("log", base=2)
    exps = [0, 3, 5, 7, 10]
    tick_positions = [2 ** n for n in exps]
    tick_labels = ["$2^{" + str(n) + "}$" for n in exps]
    tick_labels[0] = "1"
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    plt.xlabel("Partition size (MB)")
    plt.ylabel("Throughput (rows/s)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    save(plt, args.outdir, "partitioning")

    best = max(zip(throughput, num_rows))
    print(f"\n  peak throughput {best[0]:.1f} rows/s at {best[1]:g} MB")


if __name__ == "__main__":
    main()
