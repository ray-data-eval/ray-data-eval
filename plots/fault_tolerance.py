"""Plots Figure 7c.

Usage:
    python plots/fault_tolerance.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import save, use_style  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BATCH_SIZE = 32
FAILURE_AT_MIN = 15

# Max GPU throughput for one GPU (torch 2.4.0 + cu121)
MAX_GPU_TPUT = 31.570071340125125

# Checkpoint interval in seconds.
# The last checkpoint before the failure is the one a restart resumes from.
CKPT_INTERVAL_S = 6 * 60

UNINTERRUPTED = ["node_failure.csv", "executor_failure.csv"]

# Plotting configs
RESAMPLE = {"executor": 90, "node": 90, "executor_ckpt": 45, "node_ckpt": 45}
SMOOTH_BOUNDS = {"executor_ckpt": (900, 1000), "node_ckpt": (1000, 2000)}
SMOOTH_INSIDE = {"node": (870, 950)}
# Fraction of a bin the closing point covers; the curve ends at that fraction
# of its plateau.
END_FILL = 0.90

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
    (
        "node_ckpt",
        [
            "node_failure_ckpt_seg0.csv",
            "node_failure_ckpt_seg1.csv",
            "node_failure_ckpt_seg2.csv",
        ],
        "Node failure, restart from ckpt",
        "#fb9a99",
        1.0,
    ),
    (
        "executor_ckpt",
        ["executor_failure_ckpt_seg0.csv", "executor_failure_ckpt_seg1.csv"],
        "Executor failure, restart from ckpt",
        "blue",
        0.5,
    ),
]


def rate(df, resample):
    """Turn a cumulative row count into throughput over time.

    Returns (minutes, videos/s).
    """
    d = df.copy()
    d["time_from_start"] = d["time_from_start"] - d["time_from_start"].min()
    d["time_from_start"] = pd.to_timedelta(d["time_from_start"].to_numpy(), unit="s")
    d.set_index("time_from_start", inplace=True)
    span = d.index[-1].total_seconds()
    counts = d["number_of_rows_finished"]
    d = d.resample(f"{resample}s").max().diff().fillna(0) / resample
    # The final bin holds only the run's leftover seconds but is divided by the
    # full width, so every curve ends in a drop. That drop is what makes each
    # completion time readable, since all four finish at the same throughput.
    # Left to chance its depth is whatever fraction of a bin the run happened
    # to leave over, so the last bin is instead cut to a fixed END_FILL of the
    # width: the rows done in the last END_FILL * resample seconds, over the
    # full resample, placed at the run's true end.
    if len(d) > 1:
        t = counts.index.total_seconds()
        n = counts.to_numpy()
        n_before = np.interp(span - END_FILL * resample, t, n)
        d = d.iloc[:-1]
        d.loc[pd.Timedelta(seconds=span)] = (n[-1] - n_before) / resample
    return d.index.total_seconds() / 60, d["number_of_rows_finished"].values


def stitch(segments, ckpt_batches):
    """Splice a run's restart segments into one continuous timeline.

    Returns the joined frame, with each restart's redone work deducted.
    """
    segs = [s.copy() for s in segments]
    col = "number_of_rows_finished"
    bounds = (0,) + tuple(ckpt_batches)
    lost = [
        segs[i][col].max() - (bounds[i + 1] - bounds[i]) * BATCH_SIZE
        for i in range(len(ckpt_batches))
    ]
    for i, amount in enumerate(lost):
        if amount < 0:
            raise SystemExit(
                f"segment {i} is recorded as keeping "
                f"{bounds[i + 1] - bounds[i]} batches but only ran "
                f"{segs[i][col].max() / BATCH_SIZE:.0f}; check the resume "
                f"points in the segment manifest"
            )

    for i, amount in enumerate(lost):
        nxt = segs[i + 1]
        nxt[col] = (nxt[col] - amount).clip(lower=0)
    for i in range(len(lost)):
        segs[i + 1][col] += segs[i][col].max()

    # The gap between one segment ending and the next starting is the restart
    # itself: relaunching the job and reloading the model. A system resuming
    # from a checkpoint pays it too, so it stays on the timeline.
    return pd.concat(segs)


def smooth_inside(idx, vals, bounds):
    """Average the curve between `bounds` only, over neighbouring bins.

    Returns (minutes, videos/s).
    """
    s = pd.Series(vals, index=pd.to_timedelta(np.asarray(idx) * 60, unit="s"))
    lo, hi = (pd.Timedelta(seconds=b) for b in bounds)
    inside = (s.index > lo) & (s.index <= hi)
    # Average over the whole series but write back only inside the window, so
    # a bin at the edge is blended with its real neighbours on both sides.
    s[inside] = s.rolling(3, center=True, min_periods=1).mean()[inside]
    return s.index.total_seconds() / 60, s.values


def smooth(idx, vals, bounds):
    """Smooth the curve outside `bounds`, leaving the outage between them raw.

    Returns (minutes, videos/s).
    """
    s = pd.Series(vals, index=pd.to_timedelta(np.asarray(idx) * 60, unit="s"))
    lo, hi = (pd.Timedelta(seconds=b) for b in bounds)
    # A rolling mean keeps every point at its own time. Re-bucketing the tail
    # would stamp the last point with its bucket's start and shorten the run.
    # The closing drop is left out of the average so it survives.
    last = s.iloc[-1:]
    s = s.iloc[:-1]
    a = s[s.index <= lo].rolling(3, center=True, min_periods=1).mean()
    b = s[(s.index > lo) & (s.index <= hi)]
    c = s[s.index > hi].rolling(3, center=True, min_periods=1).mean()
    out = pd.concat([a, b, c, last])
    return out.index.total_seconds() / 60, out.values


def total_batches(src):
    """Batches in a complete run, read from the uninterrupted runs alongside."""
    totals = {
        int(
            pd.read_csv(os.path.join(src, f))["number_of_rows_finished"].max()
            / BATCH_SIZE
        )
        for f in UNINTERRUPTED
        if os.path.isfile(os.path.join(src, f))
    }
    # Runs of the same benchmark land a batch or two apart, because the CSV is
    # rebuilt from the log and the trailing lines are not always flushed.
    if max(totals) - min(totals) > max(totals) * 0.01:
        raise SystemExit(f"uninterrupted runs disagree on the total: {totals}")
    return max(totals)


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
    points[-1] = total_batches(src) - int(
        dfs[-1]["number_of_rows_finished"].max() / BATCH_SIZE
    )
    return points


def check_segments(src):
    """Print how each restart's resume point was arrived at."""
    total = total_batches(src)
    print(f"  a complete run is {total} batches\n")
    for _, names, label, *_ in DASHED:
        if not all(os.path.isfile(os.path.join(src, n)) for n in names):
            continue
        resumed = resume_points(src, names)
        ran = [
            int(
                pd.read_csv(os.path.join(src, n))["number_of_rows_finished"].max()
                / BATCH_SIZE
            )
            for n in names
        ]
        print(f"  {label}")
        print(
            f"      {'segment':9s} {'resumed at':>11s} {'ran':>6s} {'kept':>6s} "
            f"{'redone':>7s}   source"
        )
        bounds = [0] + resumed
        for i in range(len(names)):
            end = bounds[i + 1] if i + 1 < len(bounds) else total
            kept = end - bounds[i]
            how = (
                "start"
                if i == 0
                else "total - batches in the final segment"
                if i == len(names) - 1
                else f"checkpoint at {FAILURE_AT_MIN * 60 // CKPT_INTERVAL_S * CKPT_INTERVAL_S // 60} min"
            )
            flag = "" if 0 <= ran[i] - kept else "   <-- kept more than it ran"
            print(
                f"      seg{i:<6d} {bounds[i]:11d} {ran[i]:6d} {kept:6d} "
                f"{ran[i] - kept:7d}   {how}{flag}"
            )
        covered = sum(
            (bounds[i + 1] if i + 1 < len(bounds) else total) - bounds[i]
            for i in range(len(names))
        )
        print(
            f"      {'':9s} {'':11s} {'':6s} {covered:6d} accounted for   "
            f"{'OK' if covered == total else 'MISMATCH'}"
        )

        # The last resume point is taken from the total.
        dfs = [pd.read_csv(os.path.join(src, n)) for n in names]
        from_ckpts = sum(batches_at_last_checkpoint(d) for d in dfs[:-1])
        print(
            f"      last resume point: {resumed[-1]} from the total, "
            f"{from_ckpts} from the checkpoint times "
            f"(delta {resumed[-1] - from_ckpts:+d} batches, "
            f"{(resumed[-1] - from_ckpts) / total * 100:.2f}% of the run)\n"
        )


def opening_penalty(df, window=60):
    """How far a curve's first `window` seconds fall below its own best rate.

    Measured against the fastest sustained stretch of the same run rather than
    its mean, so a run that contains an outage is not flattered by it. A run
    that warmed up starts near its best rate; one that did not pays for the
    model load and the CUDA context on the clock.
    """
    t = (df["time_from_start"] - df["time_from_start"].min()).to_numpy()
    n = df["number_of_rows_finished"].to_numpy()
    if t.max() <= window * 2:
        return None
    opening = n[t <= window].max() / window
    best = max(
        (
            (n[j] - n[i]) / (t[j] - t[i])
            for i in range(len(t))
            for j in (int(np.searchsorted(t, t[i] + 2 * window)),)
            if j < len(t)
        ),
        default=0,
    )
    return None if not best else 1 - opening / best


def check_warmup(src):
    """Compare the opening of every curve; they must all have warmed up."""
    print(
        "  opening 60s against each run's own best sustained rate "
        "(a warmed-up run is near 0%)\n"
    )
    worst = {}
    for group, entries in (("solid", SOLID), ("dashed", DASHED)):
        for entry in entries:
            names = entry[1] if group == "dashed" else [entry[1]]
            for name in names:
                path = os.path.join(src, name)
                if not os.path.isfile(path):
                    continue
                pen = opening_penalty(pd.read_csv(path))
                if pen is None:
                    continue
                print(f"      {name:44s} {pen * 100:5.1f}%")
                worst[group] = max(worst.get(group, 0), pen)
    if len(worst) == 2 and abs(worst["solid"] - worst["dashed"]) > 0.05:
        print(
            "\n      WARNING: solid and dashed curves did not warm up the "
            "same way;\n               the difference between them includes "
            "startup, not just rollback."
        )
    print()


def load(src, resample=None):
    """Read every curve in the figure.

    Returns (minutes, videos/s, label, colour, linestyle, linewidth, alpha)
    per curve.
    """
    resample = {**RESAMPLE, **(resample or {})}

    def read(name):
        path = os.path.join(src, name)
        return pd.read_csv(path) if os.path.isfile(path) else None

    out = []
    for key, names, label, color, alpha in DASHED:
        segs = [read(n) for n in names]
        if any(seg is None for seg in segs):
            print(f"  skip {label}: missing segments")
            continue
        x, y = rate(stitch(segs, resume_points(src, names)), resample[key])
        x, y = smooth(x, y, SMOOTH_BOUNDS[key])
        out.append((x, y, label, color, "--", 1.2, alpha))

    for key, fname, label, color, alpha in SOLID:
        df = read(fname)
        if df is None:
            print(f"  skip {label}: no {fname}")
            continue
        x, y = rate(df, resample[key])
        if key in SMOOTH_INSIDE:
            x, y = smooth_inside(x, y, SMOOTH_INSIDE[key])
        out.append((x, np.asarray(y), label, color, "-", None, alpha))
    return out


def main():
    """Draw Figure 7c."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=os.path.join(ROOT, "results-archive"))
    p.add_argument("--outdir", default=os.path.join(ROOT, "figures"))
    p.add_argument(
        "--check-segments",
        action="store_true",
        help="verify the checkpoint manifests against the data",
    )
    p.add_argument(
        "--resample",
        metavar="KEY=SECONDS,...",
        help="seconds per point, per curve. Keys: "
        + ", ".join(f"{k} (default {v})" for k, v in RESAMPLE.items()),
    )
    args = p.parse_args()

    src = os.path.join(args.results, "fault_tolerance")
    if args.check_segments:
        check_segments(src)
        check_warmup(src)
        return
    resample = {}
    for item in (args.resample or "").split(","):
        if item.strip():
            k, _, v = item.partition("=")
            if k.strip() not in RESAMPLE:
                sys.exit(
                    f"unknown resample key {k.strip()!r}; "
                    f"expected one of {', '.join(RESAMPLE)}"
                )
            resample[k.strip()] = float(v)
    curves = load(src, resample)
    if not curves:
        sys.exit(f"no results under {src}")

    max_gpu = MAX_GPU_TPUT

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
    plt.xlim(0, 55)
    plt.ylim(0, 60)
    fp = FontProperties()
    fp.set_size(7.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered = [(by_label[l], l) for l in LEGEND_ORDER if l in by_label]
    plt.legend(
        [h for h, _ in ordered],
        [l for _, l in ordered],
        prop=fp,
        handlelength=1.5,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.96),
        borderaxespad=0,
        framealpha=1.0,
    )
    save(plt, args.outdir, "fault_tolerance")

    print(f"\n  max GPU throughput {max_gpu:.2f} videos/s (1 GPU, no gaps)")
    print(f"\n  {'curve':38s} {'plateau':>8s} {'% of max':>9s}")
    for x, y, label, *_ in curves:
        nz = np.asarray(y, dtype=float)
        nz = nz[nz > 1]
        if not len(nz):
            continue
        plateau = float(np.percentile(nz, 75))
        print(f"  {label:38s} {plateau:8.2f} {plateau / max_gpu * 100:8.1f}%")


if __name__ == "__main__":
    main()
