#!/usr/bin/env python3
"""Generate synthetic videos in place of the Kinetics-700-2020 test split.

    python scripts/setup/make_synthetic_videos.py --out /tmp/kinetics-synth --count 6500

Kinetics is a list of YouTube URLs, so it cannot be redistributed, and link
rot means a fresh download will not match the 64,535 videos in the paper.
This generates clips with the same shape and roughly the same encoded size,
so Figures 7b and 7c run without the real data.

Although absolute videos/s might differ from the paper due to different
decode costs, and the classifier's predictions are meaningless on noise, the
relative ordering of the systems is unaffected.

A marker file, <out>.SYNTHETIC.json, is written beside the tree.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# Matches the Kinetics-700 distribution: ~10 s clips at 25 fps, 340x256
# before the pipeline's 224x224 crop.
DEFAULT_SECONDS = 10
DEFAULT_FPS = 25
DEFAULT_WIDTH = 340
DEFAULT_HEIGHT = 256

NUM_CLASSES = 700


def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def make_video(path: str, seconds: int, fps: int, width: int, height: int) -> None:
    """One clip of coloured noise, H.264 encoded like the real dataset.

    Noise rather than a solid colour on purpose: a static frame compresses to
    almost nothing, which would make decode cost -- the thing the CPU stage
    actually spends its time on -- unrepresentative.
    """
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi",
         "-i", f"nullsrc=s={width}x{height}:d={seconds}:r={fps}",
         "-vf", "geq=random(1)*255:128:128",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         path],
        check=True, capture_output=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--count", type=int, default=6500,
                   help="number of videos (64535 matches the full test split)")
    p.add_argument("--seconds", type=int, default=DEFAULT_SECONDS)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    args = p.parse_args()

    if not have_ffmpeg():
        sys.exit("error: ffmpeg is required.\n"
                 "  apt-get install -y ffmpeg   /   brew install ffmpeg")

    os.makedirs(args.out, exist_ok=True)
    print(f"\nGenerating {args.count} synthetic videos into {args.out}")
    print(f"  {args.width}x{args.height}, {args.seconds}s @ {args.fps}fps\n")

    for i in range(args.count):
        cls = f"class_{i % NUM_CLASSES:03d}"
        cls_dir = os.path.join(args.out, cls)
        os.makedirs(cls_dir, exist_ok=True)
        path = os.path.join(cls_dir, f"synth_{i:06d}.mp4")
        if not os.path.exists(path):
            make_video(path, args.seconds, args.fps, args.width, args.height)
        if (i + 1) % 100 == 0 or i + 1 == args.count:
            print(f"\r  {i + 1}/{args.count}", end="", flush=True)

    # Marker so synthetic runs are labelled in result files.
    with open(os.path.join(os.path.dirname(args.out.rstrip("/")) or ".",
                           os.path.basename(args.out.rstrip("/")) +
                           ".SYNTHETIC.json"), "w") as f:
        json.dump({"synthetic": True, "count": args.count,
                   "seconds": args.seconds, "fps": args.fps,
                   "width": args.width, "height": args.height,
                   "note": "Substitute for Kinetics-700-2020. Relative system "
                           "ordering holds; absolute throughput does not."}, f, indent=2)

    total_mb = sum(os.path.getsize(os.path.join(r, n))
                   for r, _, fs in os.walk(args.out) for n in fs) / 1024**2
    print(f"\n\nDone: {args.count} videos, {total_mb:.0f} MB")
    print(f"Use with:  export KINETICS_DIR={args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
