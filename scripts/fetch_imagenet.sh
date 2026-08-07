#!/usr/bin/env bash
# Fetch ImageNet (ILSVRC2012) for Figure 8a.
#
# ImageNet cannot be redistributed. You must register at image-net.org and
# accept its terms, which take up to a few days to be approved. If that is not
# practical, scripts/make_synthetic_images.py generates a substitute
# that runs the same pipeline.

set -euo pipefail

DEST="${IMAGENET_DIR:-$HOME/imagenet}"

cat <<MSG

ImageNet (ILSVRC2012) -- Figure 8a
==================================

This dataset is NOT redistributable, so this script cannot download it for
you. To obtain it:

  1. Register at https://image-net.org/download.php and accept the terms.
  2. Download ILSVRC2012_img_train.tar (about 138 GB).
  3. Extract into per-class directories:

       mkdir -p $DEST/train && cd $DEST/train
       tar -xf /path/to/ILSVRC2012_img_train.tar
       for f in n*.tar; do
         d="\${f%.tar}"; mkdir -p "\$d"; tar -xf "\$f" -C "\$d"; rm "\$f"
       done

  4. export IMAGENET_DIR=$DEST/train

Approval can take a few days. If you would rather not wait, the synthetic
substitute exercises the identical pipeline:

  python scripts/make_synthetic_images.py --out $DEST/synthetic --count 50000
  export IMAGENET_DIR=$DEST/synthetic

Although absolute throughput might differ from the paper with synthetic data
due to different JPEG decode costs, the *relative* comparison between
Ray Data and tf.data is unaffected.

MSG

if [ -d "$DEST/train" ]; then
  count=$(find "$DEST/train" -name '*.JPEG' 2>/dev/null | head -200000 | wc -l | tr -d ' ')
  echo "Found $DEST/train with $count+ JPEGs."
else
  echo "Not found: $DEST/train"
fi
echo
