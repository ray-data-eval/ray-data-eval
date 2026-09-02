"""Produces every figure.

Usage:
    python plots/all.py
    python plots/all.py --results results     # from your own runs
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = [
    ("rag", "7a", "RAG"),
    ("video_classification", "7b", "Video classification"),
    ("fault_tolerance", "7c", "Fault tolerance"),
    ("resnet_training", "8a", "ResNet-50 training"),
    ("memory_pipelining", "9", "Synthetic pipelining"),
    ("partitioning", "10a", "Partition size"),
    ("scalability", "10b", "Scalability"),
]


def main():
    """Run every per-figure script in turn."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--results", default=None)
    p.add_argument("--outdir", default=None)
    args, extra = p.parse_known_args()

    failed = []
    for name, fig, title in FIGURES:
        cmd = [sys.executable, os.path.join(HERE, f"{name}.py")] + extra
        if args.results:
            cmd += ["--results", args.results]
        if args.outdir:
            cmd += ["--outdir", args.outdir]
        print(f"\n=== Figure {fig} - {title} " + "=" * max(0, 40 - len(title)))
        if subprocess.run(cmd).returncode:
            failed.append((fig, name))

    print()
    if failed:
        print(f"  {len(failed)} of {len(FIGURES)} failed: "
              + ", ".join(f"{f} ({n})" for f, n in failed))
        return 1
    print(f"  all {len(FIGURES)} figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
