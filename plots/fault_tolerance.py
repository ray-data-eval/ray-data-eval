"""Plots Figure 7c.

Usage:
    python plots/fault_tolerance.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import use_style, save  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 32
FAILURE_AT_MIN = 15

# Checkpoint interval in seconds.
# The last checkpoint before the failure is the one a restart resumes from.
CKPT_INTERVAL_S = 6 * 60

UNINTERRUPTED = ["node_failure.csv", "executor_failure.csv"]

# For nicer plotting
RESAMPLE = {"executor": 125, "node": 121, "executor_ckpt": 20, "node_ckpt": 50}
SMOOTH_BOUNDS = {"executor_ckpt": (900, 1100), "node_ckpt": (1000, 2000)}

LEGEND_ORDER = [
    "Max GPU throughput",
    "Executor failure",
    "Executor failure, restart from ckpt",
    "Node failure",
    "Node failure, restart from ckpt",
]

SOLID = [
    ("node", "node_failure.csv", "Node failure", "#e31a1c", 1.0),
    ("executor", "executor_failure.csv", "Executor failure", "blue", 1.0),
]
DASHED = [
    ("node_ckpt", ["node_failure_ckpt_seg0.csv", "node_failure_ckpt_seg1.csv",
                   "node_failure_ckpt_seg2.csv"],
     "Node failure, restart from ckpt", "#fb9a99", 1.0),
    ("executor_ckpt", ["executor_failure_ckpt_seg0.csv",
                       "executor_failure_ckpt_seg1.csv"],
     "Executor failure, restart from ckpt", "blue", 0.5),
]


def rate(df, resample):
    """Turn a cumulative row count into throughput over time.

    Returns (minutes, videos/s).
    """
    d = df.copy()
    d["time_from_start"] = d["time_from_start"] - d["time_from_start"].min()
    d["time_from_start"] = pd.to_timedelta(d["time_from_start"].to_numpy(), unit="s")
    d.set_index("time_from_start", inplace=True)
    d = d.resample(f"{resample}s").max().diff().fillna(0) / resample
    return d.index.total_seconds() / 60, d["number_of_rows_finished"].values


def stitch(segments, ckpt_batches):
    """Splice a run's restart segments into one continuous timeline.

    Returns the joined frame, with each restart's redone work deducted.
    """
    segs = [s.copy() for s in segments]
    col = "number_of_rows_finished"
    bounds = (0,) + tuple(ckpt_batches)
    lost = [segs[i][col].max() - (bounds[i + 1] - bounds[i]) * BATCH_SIZE
            for i in range(len(ckpt_batches))]
    for i, amount in enumerate(lost):
        if amount < 0:
            raise SystemExit(
                f"segment {i} is recorded as keeping "
                f"{bounds[i + 1] - bounds[i]} batches but only ran "
                f"{segs[i][col].max() / BATCH_SIZE:.0f}; check the resume "
                f"points in the segment manifest")

    for i, amount in enumerate(lost):
        nxt = segs[i + 1]
        nxt[col] = (nxt[col] - amount).clip(lower=0)
    for i in range(len(lost)):
        segs[i + 1][col] += segs[i][col].max()
    return pd.concat(segs)


def smooth(idx, vals, bounds):
    """Smooth the curve outside `bounds`, leaving the outage between them raw.

    Returns (minutes, videos/s).
    """
    s = pd.Series(vals, index=pd.to_timedelta(np.asarray(idx) * 60, unit="s"))
    lo, hi = (pd.Timedelta(seconds=b) for b in bounds)
    a = s[s.index <= lo].rolling("120s").mean().fillna(0).resample("120s").mean()
    b = s[(s.index > lo) & (s.index <= hi)]
    c = s[s.index > hi].rolling("120s").mean().fillna(0).resample("120s").mean()
    out = pd.concat([a, b, c])
    return out.index.total_seconds() / 60, out.values


def total_batches(src):
    """Batches in a complete run, read from the uninterrupted runs alongside."""
    totals = {int(pd.read_csv(os.path.join(src, f))["number_of_rows_finished"].max()
                  / BATCH_SIZE)
              for f in UNINTERRUPTED if os.path.isfile(os.path.join(src, f))}
    if len(totals) != 1:
        raise SystemExit(f"uninterrupted runs disagree on the total: {totals}")
    return totals.pop()


def batches_at_last_checkpoint(df):
    """Batches a segment had finished at its last checkpoint before failing."""
    boundary = (FAILURE_AT_MIN * 60 // CKPT_INTERVAL_S) * CKPT_INTERVAL_S
    t0 = df["time_from_start"].min()
    done = df[df["time_from_start"] <= t0 + boundary]["number_of_rows_finished"].max()
    return int(done / BATCH_SIZE)


def resume_points(src, names):
    """Batch index each restart resumed from.

    Returns one index per restart.
    """
    dfs = [pd.read_csv(os.path.join(src, n)) for n in names]
    points, acc = [], 0
    for df in dfs[:-1]:
        acc += batches_at_last_checkpoint(df)
        points.append(acc)
    points[-1] = total_batches(src) - int(dfs[-1]["number_of_rows_finished"].max()
                                          / BATCH_SIZE)
    return points


def check_segments(src):
    """Print how each restart's resume point was arrived at."""
    total = total_batches(src)
    print(f"  a complete run is {total} batches\n")
    for _, names, label, *_ in DASHED:
        if not all(os.path.isfile(os.path.join(src, n)) for n in names):
            continue
        resumed = resume_points(src, names)
        ran = [int(pd.read_csv(os.path.join(src, n))["number_of_rows_finished"].max()
                   / BATCH_SIZE) for n in names]
        print(f"  {label}")
        print(f"      {'segment':9s} {'resumed at':>11s} {'ran':>6s} {'kept':>6s} "
              f"{'redone':>7s}   source")
        bounds = [0] + resumed
        for i in range(len(names)):
            end = bounds[i + 1] if i + 1 < len(bounds) else total
            kept = end - bounds[i]
            how = ("start" if i == 0 else
                   "total - batches in the final segment"
                   if i == len(names) - 1 else
                   f"checkpoint at {FAILURE_AT_MIN * 60 // CKPT_INTERVAL_S * CKPT_INTERVAL_S // 60} min")
            flag = "" if 0 <= ran[i] - kept else "   <-- kept more than it ran"
            print(f"      seg{i:<6d} {bounds[i]:11d} {ran[i]:6d} {kept:6d} "
                  f"{ran[i] - kept:7d}   {how}{flag}")
        covered = sum(
            (bounds[i + 1] if i + 1 < len(bounds) else total) - bounds[i]
            for i in range(len(names)))
        print(f"      {'':9s} {'':11s} {'':6s} {covered:6d} accounted for   "
              f"{'OK' if covered == total else 'MISMATCH'}")

        # The last resume point is taken from the total.
        dfs = [pd.read_csv(os.path.join(src, n)) for n in names]
        from_ckpts = sum(batches_at_last_checkpoint(d) for d in dfs[:-1])
        print(f"      last resume point: {resumed[-1]} from the total, "
              f"{from_ckpts} from the checkpoint times "
              f"(delta {resumed[-1] - from_ckpts:+d} batches, "
              f"{(resumed[-1] - from_ckpts) / total * 100:.2f}% of the run)\n")


def load(src):
    """Read every curve in the figure.

    Returns (minutes, videos/s, label, colour, linestyle, linewidth, alpha)
    per curve.
    """
    def read(name):
        path = os.path.join(src, name)
        return pd.read_csv(path) if os.path.isfile(path) else None

    out = []
    for key, names, label, color, alpha in DASHED:
        segs = [read(n) for n in names]
        if any(seg is None for seg in segs):
            print(f"  skip {label}: missing segments")
            continue
        x, y = rate(stitch(segs, resume_points(src, names)), RESAMPLE[key])
        x, y = smooth(x, y, SMOOTH_BOUNDS[key])
        out.append((x, y, label, color, "--", 1.2, alpha))

    for key, fname, label, color, alpha in SOLID:
        df = read(fname)
        if df is None:
            print(f"  skip {label}: no {fname}")
            continue
        x, y = rate(df, RESAMPLE[key])
        out.append((x, np.asarray(y), label, color, "-", None, alpha))
    return out


def main():
    """Draw Figure 7c."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=os.path.join(ROOT, "results-archive"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    p.add_argument("--check-segments", action="store_true",
                   help="verify the checkpoint manifests against the data")
    args = p.parse_args()

    src = os.path.join(args.results, "fault_tolerance")
    if args.check_segments:
        check_segments(src)
        return
    curves = load(src)
    if not curves:
        sys.exit(f"no results under {src}")

    max_gpu = float(pd.read_csv(os.path.join(src, "inference_tput.csv"))
                    ["videos_per_s"].median())

    use_style(figratio=3 / 4, size=12)
    _, ax = plt.subplots()
    ax.axhline(y=max_gpu, color="orange", linestyle="--", label="Max GPU throughput")
    for x, y, label, color, ls, lw, alpha in curves:
        kw = {"label": label, "color": color, "linestyle": ls, "alpha": alpha}
        if lw is not None:
            kw["linewidth"] = lw
        ax.plot(x, y, **kw)
    ax.axvline(x=FAILURE_AT_MIN, color="black", linewidth=1)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    plt.xlabel("Time (min)", fontsize=12)
    plt.ylabel("Throughput (videos/s)", fontsize=12)
    plt.xlim(0, 65)
    plt.ylim(0, 60)
    fp = FontProperties()
    fp.set_size(7.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered = [(by_label[l], l) for l in LEGEND_ORDER if l in by_label]
    # Top right, nudged in off the frame. Above 7.5pt the box grows wide
    # enough to reach left of the failure line and cover it.
    plt.legend([h for h, _ in ordered], [l for _, l in ordered],
               prop=fp, handlelength=1.5, loc="upper right",
               bbox_to_anchor=(0.995, 0.96), borderaxespad=0, framealpha=1.0)
    save(plt, args.outdir, "fault_tolerance")

    print(f"\n  max GPU throughput {max_gpu:.2f} videos/s "
          f"(median of inference_tput.csv)")
    print(f"\n  {'curve':38s} {'plateau':>8s} {'% of max':>9s}")
    for x, y, label, *_ in curves:
        nz = np.asarray(y, dtype=float)
        nz = nz[nz > 1]
        if not len(nz):
            continue
        plateau = float(np.percentile(nz, 75))
        print(f"  {label:38s} {plateau:8.2f} {plateau/max_gpu*100:8.1f}%")


if __name__ == "__main__":
    main()
