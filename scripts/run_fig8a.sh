#!/usr/bin/env bash
# Figure 8a: ResNet-50 training, Ray Data against tf.data, local and S3.
#
#   bash scripts/run_fig8a.sh [dataset-root]
#
# Needs 1 GPU node. The dataset root defaults to $IMAGENET_DIR, or to the
# synthetic substitute from scripts/setup/make_synthetic_images.py.
set -u

DATA="${1:-${IMAGENET_DIR:-/tmp/imagenet-synth}}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
DEST=${RESULTS_DIR:-$ROOT/results}/resnet_training
mkdir -p "$DEST"

[ -d "$DATA/train" ] || {
    echo "no $DATA/train; run scripts/setup/fetch_imagenet.sh or"
    echo "  python scripts/setup/make_synthetic_images.py --out $DATA --count 2000 --classes 10"
    exit 1
}

export RAY_DEDUP_LOGS=0
export PYTHONPATH=$ROOT/experiments
cd experiments/ray_data_eval/image_training/e2e_training

echo "== Ray Data =="
python ray_data_e2e_training.py -a resnet50 -b 128 "$DATA" \
    > e2e_training_s3_g5_xlarge_batch_256.out 2>&1
echo "== tf.data =="
python tf_data_e2e_training.py -b 256 --local "$DATA" \
    > tf_data_e2e_training_g5_xlarge_batch_256.out 2>&1

for f in *.csv; do [ -f "$f" ] && cp "$f" "$DEST/$f"; done
echo
echo "Wrote $DEST"
echo "Plot with:  python plots/resnet_training.py --results results"
