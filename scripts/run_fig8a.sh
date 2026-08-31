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

# The benchmark joins an existing Ray with ray.init("auto"); on a fresh
# single node there is none, so start one.
ray status >/dev/null 2>&1 || { echo "starting a local Ray"; ray start --head --disable-usage-stats >/dev/null; }

cd experiments/ray_data_eval/image_training/e2e_training

check() {
    # A phase that wrote no CSV failed, whatever its exit code said.
    ls $1 >/dev/null 2>&1 && return 0
    echo "  FAILED: no $1 produced; last lines of $2:"
    tail -8 "$2" | sed "s/^/    /"
    exit 1
}

echo "== Ray Data =="
python ray_data_e2e_training.py -a resnet50 -b 128 "$DATA" \
    > e2e_training_s3_g5_xlarge_batch_256.out 2>&1
check "ray_data_*.csv" e2e_training_s3_g5_xlarge_batch_256.out
echo "== tf.data =="
python tf_data_e2e_training.py -b 256 --local "$DATA" \
    > tf_data_e2e_training_g5_xlarge_batch_256.out 2>&1
check "tfdata_*.csv" tf_data_e2e_training_g5_xlarge_batch_256.out

for f in *.csv; do [ -f "$f" ] && cp "$f" "$DEST/$f"; done
echo
echo "Wrote $DEST"
echo "Plot with:  python plots/resnet_training.py --results results"
