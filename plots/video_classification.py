"""Plots Figure 7b.

Usage:
    python plots/video_classification.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import use_style, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Optimal time and max GPU throughput.
INFERENCE_TPUT_MEAN = 31.570071340125125 * 4  # = 126.28 videos/s
OPTIMAL_S = 511.04572511669

SYSTEM_NAME = "Ray Data"

# (file, label, resample interval, colour, linestyle, linewidth)
SERIES = [
    ("ray_data_microbatch.csv", f"{SYSTEM_NAME}-microbatch", "2.5s", "black", "-", 0.05),
    ("ray_data_static.csv", f"{SYSTEM_NAME}-static", "15s", None, "-", None),
    ("ray_data_staged.csv", f"{SYSTEM_NAME}-staged", "30s", "black", "-", None),
    ("flink.csv", "Flink", "35s", None, "--", 1),
    ("spark.csv", "Spark", "50s", "black", "--", 1),
    ("ray_data_dynamic.csv", f"{SYSTEM_NAME}-dynamic", "27s", "blue", "-", None),
]

# Per-series intervals are load-bearing: at one interval the microbatch sawtooth
# flattens out, and that sawtooth is the figure's argument. --resample overrides
# them, keyed by the file's stem.

LEGEND_LEFT = ["Max GPU throughput", "Flink", "Spark"]
LEGEND_RIGHT = [f"{SYSTEM_NAME}-dynamic", f"{SYSTEM_NAME}-static",
                f"{SYSTEM_NAME}-staged", f"{SYSTEM_NAME}-microbatch"]


def get_results_and_preview(results_path: str, resample_interval: str):
    """Turn a cumulative row count into throughput over time.

    Returns (minutes, videos/s, throughput over the whole run, mean of the
    binned curve).
    """
    df = pd.read_csv(results_path)
    actual = df.iloc[-1]["number_of_rows_finished"] / df.iloc[-1]["time_from_start"]

    df["time_from_start"] = pd.to_timedelta(df["time_from_start"].to_numpy(), unit="s")
    df.set_index("time_from_start", inplace=True)
    resampled = df["number_of_rows_finished"].resample(resample_interval).mean().ffill()
    time_vals = resampled.index.total_seconds()
    row_vals = resampled

    tput_vals = row_vals.diff() / time_vals.diff()
    tput_vals = tput_vals.fillna(0)
    filled_time_vals_rd = []
    filled_tput_vals_rd = []

    # Pad from t=0 to the first sample with zeros, so a pipeline with a long
    # startup does not appear to begin at full speed.
    for i in np.arange(0, int(time_vals[0]), 5):
        filled_time_vals_rd.append(i)
        filled_tput_vals_rd.append(0)

    filled_time_vals_rd.extend(time_vals)
    filled_tput_vals_rd.extend(tput_vals)

    smoothed = np.average(filled_tput_vals_rd)
    filled_time_vals_rd = [item / 60 for item in filled_time_vals_rd]
    return filled_time_vals_rd, filled_tput_vals_rd, actual, smoothed


def main():
    """Draw Figure 7b and check section 5.1.2's four claims against it."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=os.path.join(ROOT, "results-archive"))
    p.add_argument("--resample", metavar="NAME=INTERVAL,...",
                   help="resample interval per series, e.g. "
                        "ray_data_dynamic=10s,spark=60s")
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--simple", action="store_true",
                   help="single panel, no broken axis (easier to compare a "
                        "short reviewer-scale run)")
    args = p.parse_args()

    src = os.path.join(args.results, "video_classification")
    colors = use_style(figratio=3 / 4, size=12)

    loaded, stats = [], []
    overrides = {}
    for item in (args.resample or "").split(","):
        if item.strip():
            k, _, v = item.partition("=")
            overrides[k.strip()] = v.strip()
    unknown = set(overrides) - {f[:-4] for f, *_ in SERIES}
    if unknown:
        sys.exit(f"unknown series {', '.join(sorted(unknown))}; expected "
                 + ", ".join(f[:-4] for f, *_ in SERIES))

    for fname, label, interval, color, ls, lw in SERIES:
        interval = overrides.get(fname[:-4], interval)
        path = os.path.join(src, fname)
        if not os.path.isfile(path):
            print(f"  skip {label}: no {fname}")
            continue
        t, y, actual, _ = get_results_and_preview(path, interval)
        loaded.append((t, y, label, color or colors[5], ls, lw))
        # JCT from the raw last row, not t[-1] -- t[-1] is the last resample
        # bucket, which lands short by up to one interval.
        raw = pd.read_csv(path).iloc[-1]
        stats.append((label, float(raw["time_from_start"]), actual))

    if not loaded:
        sys.exit(f"no results under {src}")

    def plot_all(ax):
        """Draw every series on one axis, plus the max-GPU reference line."""
        for t, y, label, color, ls, lw in loaded:
            kw = {"label": label, "color": color, "linestyle": ls}
            if lw is not None:
                kw["linewidth"] = lw
            ax.plot(t, y, **kw)
        ax.axhline(y=INFERENCE_TPUT_MEAN, color="orange", linestyle="--",
                   label="Max GPU throughput")

    if args.simple:
        fig, ax = plt.subplots()
        plot_all(ax)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Throughput (videos/s)")
        ax.set_ylim(0, 200)
        ax.legend(fontsize=6)
    else:
        # Broken x-axis: Spark runs to ~117 min, the rest finish inside 70.
        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [5, 2], "wspace": 0.08})
        plot_all(ax1)
        ax1.set_xlim(0, 70)
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax1.set_ylabel("Throughput (videos/s)")
        plot_all(ax2)
        ax2.set_xlim(90, 125)
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

        ax1.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax1.tick_params(labelright=False)
        ax2.yaxis.tick_right()

        d, d_y = 0.015, 0.015
        d_x1, d_x2 = d * (2 / 5), d
        kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False, linewidth=1.0)
        ax1.plot((1 - d_x1, 1 + d_x1), (-d_y, +d_y), **kwargs)
        ax1.plot((1 - d_x1, 1 + d_x1), (1 - d_y, 1 + d_y), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d_x2, +d_x2), (-d_y, +d_y), **kwargs)
        ax2.plot((-d_x2, +d_x2), (1 - d_y, 1 + d_y), **kwargs)

        ax1.set_xticks([0, 20, 40, 60])
        ax2.set_xticks([100, 120])
        fig.text(0.5, -0.05, "Time (min)", ha="center", va="center")

        fp = FontProperties()
        fp.set_size(7)
        fp.set_stretch("semi-expanded")
        handles, labels = ax2.get_legend_handles_labels()
        idx = {lab: i for i, lab in enumerate(labels)}
        dummy = Line2D([], [], linestyle="None", linewidth=0, alpha=0, label="")
        ordered_h, ordered_l = [], []
        for col in (LEGEND_LEFT, LEGEND_RIGHT):
            for lab in col:
                if lab in idx:
                    ordered_h.append(handles[idx[lab]])
                    ordered_l.append(lab)
            if col is LEGEND_LEFT:
                ordered_h.append(dummy)
                ordered_l.append("")
        fig.legend(ordered_h, ordered_l, loc="upper center",
                   bbox_to_anchor=(0.513, 0.88), ncol=2, frameon=True,
                   handlelength=1.5, columnspacing=0.5, prop=fp)
        plt.ylim(0, 200)

    save(plt, args.outdir, "video_classification")

    print(f"\n  optimal {OPTIMAL_S/60:.2f} min, "
          f"max GPU throughput {INFERENCE_TPUT_MEAN:.1f} videos/s\n")
    print(f"  {'series':20s} {'JCT (min)':>10s} {'videos/s':>10s}")
    for label, jct_s, tput in sorted(stats, key=lambda s: s[1]):
        print(f"  {label:20s} {jct_s/60:10.2f} {tput:10.1f}")

    # The four numbers section 5.1.2 states, recomputed from the data above.
    by = {label: (jct_s, tput) for label, jct_s, tput in stats}
    if f"{SYSTEM_NAME}-dynamic" not in by:
        return
    dyn_s, dyn_tp = by[f"{SYSTEM_NAME}-dynamic"]
    claims = [
        ("of optimal run time", lambda: OPTIMAL_S / dyn_s * 100, "91.0", "%"),
        (f"vs {SYSTEM_NAME}-microbatch",
         lambda: by[f"{SYSTEM_NAME}-microbatch"][0] / dyn_s, "1.95", "x"),
        ("throughput vs Flink", lambda: dyn_tp / by["Flink"][1], "2.6", "x"),
        (f"throughput vs {SYSTEM_NAME}-static",
         lambda: dyn_tp / by[f"{SYSTEM_NAME}-static"][1], "1.29", "x"),
    ]
    print(f"\n  {'paper section 5.1.2':30s} {'this run':>10s} {'paper':>8s}")
    for name, fn, want, unit in claims:
        try:
            got = fn()
        except KeyError:
            continue
        flag = "" if f"{got:.2f}".startswith(want[:3]) else "   <-- differs"
        print(f"  {name:30s} {got:9.2f}{unit} {want:>7s}{unit}{flag}")


if __name__ == "__main__":
    main()
