"""Plots ResNet-50 training throughput (section 5.2).

Usage:
    python plots/resnet_training.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import use_style, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Images in the ImageNet training split.
IMAGENET_TRAIN_ROWS = 1281167

SYSTEM_NAME = "Ray Data"

# (file, legend label, colour index)
SERIES = [
    ("ray_data_local.csv", f"{SYSTEM_NAME} (local)", 9),
    ("ray_data_s3.csv", f"{SYSTEM_NAME} (S3)", 1),
    ("tfdata_local.csv", "tf.data (local)", 2),
    ("tfdata_s3.csv", "tf.data (S3)", 3),
]


def process(data, resample="60s"):
    """Turn a cumulative row count into throughput over time.

    Returns (minutes, images/s, mean throughput).
    """
    data = data.copy()
    data["time_from_start"] = pd.to_timedelta(data["time_from_start"].to_numpy(), unit="s")
    data.set_index("time_from_start", inplace=True)
    data["tput"] = data["number_of_rows_finished"].diff() / data.index.total_seconds().diff()
    data["tput"] = data["tput"].fillna(0)
    avg_tput = np.mean(data["tput"])
    tput_resampled = data["tput"].resample(resample).mean().bfill()
    idx = [(item.total_seconds() // 60) for item in tput_resampled.index]
    return idx, tput_resampled.values, avg_tput


def read_column(path, column):
    """Read one numeric column from a CSV.

    Returns a list, empty if the file is absent.
    """
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [float(r[column]) for r in csv.DictReader(f) if r.get(column)]


def main():
    """Draw ResNet-50 training throughput over time."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=os.path.join(ROOT, "results-archive"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    args = p.parse_args()

    src = os.path.join(args.results, "resnet_training")
    colors = use_style(figratio=3 / 4, size=14)

    _, ax = plt.subplots()
    drawn = []
    for fname, label, ci in SERIES:
        path = os.path.join(src, fname)
        if not os.path.isfile(path):
            print(f"  skip {label}: no {fname}")
            continue
        df = pd.read_csv(path)
        idx, values, _ = process(df)
        ax.plot(idx, values, label=label, color=colors[ci])
        last = df.iloc[-1]
        drawn.append((label, last["time_from_start"] / 60,
                      last["number_of_rows_finished"] / last["time_from_start"]))

    if not drawn:
        sys.exit(f"no results under {src}")

    gpu_busy = read_column(os.path.join(src, "gpu_busy_time.csv"), "gpu_busy_s")
    theoretical_max = IMAGENET_TRAIN_ROWS / gpu_busy[0] if gpu_busy else None
    if theoretical_max:
        ax.axhline(y=theoretical_max, color="orange", linestyle="--",
                   label="Max GPU throughput")

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    plt.xlabel("Time (min)")
    plt.ylabel("Throughput (images/s)")
    plt.ylim(0, 400)
    plt.xlim(-3, 72)
    plt.legend(fontsize=10, loc="center right", bbox_to_anchor=(0.5, 0.2, 0.5, 0.5))
    save(plt, args.outdir, "resnet_training")

    if theoretical_max:
        print(f"\n  max GPU throughput {theoretical_max:.1f} images/s "
              f"({IMAGENET_TRAIN_ROWS} images / {gpu_busy[0]:.1f}s of GPU time)")
    print(f"\n  {'system':22s} {'JCT (min)':>10s} {'images/s':>10s} {'% of max':>9s}")
    for label, jct_min, tput in drawn:
        pct = f"{tput / theoretical_max * 100:8.1f}%" if theoretical_max else ""
        print(f"  {label:22s} {jct_min:10.1f} {tput:10.1f} {pct}")


if __name__ == "__main__":
    main()
