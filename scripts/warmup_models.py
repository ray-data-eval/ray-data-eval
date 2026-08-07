#!/usr/bin/env python3
"""Cache model weights locally, before running the GPU experiments.

    python scripts/warmup_models.py

Run once per machine. The experiment scripts load from a local path rather
than resolving the hub at task time, so the cache must already be populated.
Downloading here also keeps it off the clock during a timed run, and stops
every worker process fetching at once.
"""

from __future__ import annotations

import argparse
import os
import sys

VIDEOMAE_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
MODEL_CACHE = "/tmp/videomae_model"
PREPROCESSOR_CACHE = "/tmp/videomae_processor"


def warmup_videomae() -> None:
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

    print(f"  VideoMAE -> {MODEL_CACHE}")
    VideoMAEForVideoClassification.from_pretrained(VIDEOMAE_ID).save_pretrained(MODEL_CACHE)
    VideoMAEImageProcessor.from_pretrained(VIDEOMAE_ID).save_pretrained(PREPROCESSOR_CACHE)


def warmup_stable_diffusion() -> None:
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    sd = "stabilityai/stable-diffusion-2-1"
    print(f"  Stable Diffusion components ({sd})")
    for cls, sub in ((AutoencoderKL, "vae"), (UNet2DConditionModel, "unet"),
                     (CLIPTextModel, "text_encoder"), (CLIPTokenizer, "tokenizer")):
        cls.from_pretrained(sd, subfolder=sub)


def warmup_rag() -> None:
    from transformers import AutoModel, AutoTokenizer

    encoder = os.environ.get("RAG_ENCODER", "facebook/contriever")
    print(f"  RAG encoder ({encoder})")
    AutoTokenizer.from_pretrained(encoder)
    AutoModel.from_pretrained(encoder)
    print("  NOTE: Llama-3-8B is gated on Hugging Face. Accept its licence and")
    print("        run `huggingface-cli login`, or set RAG_MODEL to a substitute.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--all", action="store_true", help="every model, not just VideoMAE")
    args = p.parse_args()

    print("\nCaching model weights...\n")
    steps = [("VideoMAE", warmup_videomae)]
    if args.all:
        steps += [("Stable Diffusion", warmup_stable_diffusion), ("RAG", warmup_rag)]

    failed = 0
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {name} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"\n{'Some models failed to cache.' if failed else 'Done.'}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
