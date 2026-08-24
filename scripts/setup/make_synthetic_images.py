#!/usr/bin/env python3
"""Generate synthetic JPEGs in place of ImageNet, for Figure 8a.

    python scripts/setup/make_synthetic_images.py --out /tmp/imagenet-synth --count 50000

ImageNet requires registration and cannot be redistributed, and approval can
take a few days. This produces a directory tree with the same layout, one
directory per class with JPEGs inside, so the Figure 8a pipeline runs without
it.

Although absolute images/s might differ from the paper due to different JPEG
decode costs, the comparison between Ray Data and tf.data is unaffected,
since both read the same data.

A marker file, <out>.SYNTHETIC.json, is written beside the tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# ray_data_eval lives under experiments/, not on the path when run directly.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments"))

# ImageNet training JPEGs average roughly 110 KB at around 500x375.
DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 375
NUM_CLASSES = 1000
JPEG_QUALITY = 90


def make_image(path: str, width: int, height: int, rng: np.random.Generator) -> None:
    """A smoothly-varying noise image, JPEG-encoded.

    Pure white noise would defeat the DCT and produce implausibly large files
    that decode unusually slowly. Low-frequency gradients plus fine noise land
    much closer to a photograph's compression behaviour, which is what makes
    the decode cost representative.
    """
    from PIL import Image

    ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = np.stack([
        np.sin(6.283 * (xs * rng.uniform(0.5, 3) + ys * rng.uniform(0.5, 3))),
        np.cos(6.283 * (xs * rng.uniform(0.5, 3) - ys * rng.uniform(0.5, 3))),
        np.sin(6.283 * (xs + ys) * rng.uniform(0.5, 3)),
    ], axis=-1)
    arr = (base * 0.5 + 0.5) * 220 + rng.normal(0, 12, base.shape)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").save(
        path, "JPEG", quality=JPEG_QUALITY)


def imagenet_wnids(n: int) -> list:
    """The first `n` real ImageNet class IDs.

    Returns WNIDs like n01440764. The trainer maps these to class indices and
    raises KeyError on anything else, so invented names such as n00000000 do
    not work.
    """
    from ray_data_eval.image_training.e2e_training.util import SORTED_WNIDS
    return SORTED_WNIDS[:n]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=50_000,
                   help="number of images (ImageNet train is ~1.28M)")
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--classes", type=int, default=NUM_CLASSES,
                   help="how many ImageNet classes to spread images over")
    p.add_argument("--val-fraction", type=float, default=0.02,
                   help="share of images placed under val/ (ImageNet is ~4%%)")
    args = p.parse_args()

    try:
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit("error: Pillow is required.\n  pip install Pillow")

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    print(f"\nGenerating {args.count} synthetic JPEGs into {args.out}")
    print(f"  {args.width}x{args.height}, quality {JPEG_QUALITY}")
    print(f"  layout: {args.out}/{{train,val}}/<class>/*.JPEG\n")

    # ImageNet's layout: <root>/{train,val}/<class>/*.JPEG. The trainer joins
    # "train" and "val" onto the path it is given, so a flat tree fails with
    # FileNotFoundError before any image is read.
    n_val = max(1, int(args.count * args.val_fraction))
    wnids = imagenet_wnids(args.classes)
    for i in range(args.count):
        cls = wnids[i % len(wnids)]
        split = "val" if i < n_val else "train"
        cls_dir = os.path.join(args.out, split, cls)
        os.makedirs(cls_dir, exist_ok=True)
        path = os.path.join(cls_dir, f"synth_{i:07d}.JPEG")
        if not os.path.exists(path):
            make_image(path, args.width, args.height, rng)
        if (i + 1) % 500 == 0 or i + 1 == args.count:
            print(f"\r  {i + 1}/{args.count}", end="", flush=True)

    # Written beside the tree, not inside it: a stray non-image file in the
    # image directory breaks any reader that globs the whole tree.
    with open(os.path.join(os.path.dirname(args.out.rstrip("/")) or ".",
                           os.path.basename(args.out.rstrip("/")) +
                           ".SYNTHETIC.json"), "w") as f:
        json.dump({"synthetic": True, "count": args.count,
                   "width": args.width, "height": args.height,
                   "note": "Substitute for ImageNet. The Ray Data-vs-tf.data "
                           "comparison holds; absolute images/s does not."},
                  f, indent=2)

    total_mb = sum(os.path.getsize(os.path.join(r, n))
                   for r, _, fs in os.walk(args.out) for n in fs) / 1024**2
    print(f"\n\nDone: {args.count} images, {total_mb:.0f} MB "
          f"({total_mb * 1024 / max(args.count, 1):.0f} KB each)")
    print(f"Use with:  export IMAGENET_DIR={args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
