#!/usr/bin/env bash
# Figure 8a: ResNet-50 training, Ray Data against tf.data, local and S3.
#
#   bash scripts/run_fig8a.sh [dataset-root]
#   IMAGENET_S3=s3://bucket/prefix bash scripts/run_fig8a.sh [dataset-root]   # also runs the _s3 pair
#
# Needs 1 GPU node.
set -u

DATA="${1:-${IMAGENET_DIR:-/tmp/imagenet-synth}}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$PWD
DEST=${RESULTS_DIR:-$ROOT/results}
case "$DEST" in /*) ;; *) DEST=$ROOT/$DEST ;; esac
DEST=$DEST/resnet_training
mkdir -p "$DEST"

case "$DATA" in
s3://*) ;;  # existence is checked by the trainers at read time
*)
    [ -d "$DATA/train" ] || {
        echo "no $DATA/train; download it (see the Figure 8a section of README.md) or"
        echo "  python scripts/setup/make_synthetic_images.py --out $DATA --count 2000 --classes 10"
        exit 1
    }
    ;;
esac

export RAY_DEDUP_LOGS=0
export AWS_REGION=${AWS_REGION:-us-west-2}
export PYTHONPATH=$ROOT/experiments

ray status >/dev/null 2>&1 || { echo "starting a local Ray"; ray start --head --disable-usage-stats >/dev/null; }

cd experiments/ray_data_eval/image_training/e2e_training

RD=e2e_training_s3_g5_xlarge_batch_256
TF=tf_data_e2e_training_g5_xlarge_batch_128

check() {
    [ -f "$1.csv" ] && return 0
    echo "  FAILED: no $1.csv produced; last lines of $1.out:"
    tail -8 "$1.out" | sed "s/^/    /"
    exit 1
}

# tf.data uses batch 128 (section 5.2.1).
series() {  # series <local|s3> <ray-data-root> <tf.data-root> [tf.data flags]
    local SERIES=$1 RAYDATA=$2 TFDATA=$3; shift 3
    rm -f "$RD.csv" "$TF.csv"
    echo "== Ray Data ($SERIES) =="
    python ray_data_e2e_training.py -a resnet50 -b 256 --epochs 1 "$RAYDATA" > "$RD.out" 2>&1
    check "$RD"
    cp "$RD.csv" "$DEST/ray_data_$SERIES.csv"
    echo "Wrote $DEST/ray_data_$SERIES.csv"
    [ -n "$TFDATA" ] || return 0
    echo "== tf.data ($SERIES) =="
    python tf_data_e2e_training.py -b 128 "$@" "$TFDATA" > "$TF.out" 2>&1
    check "$TF"
    cp "$TF.csv" "$DEST/tfdata_$SERIES.csv"
    echo "Wrote $DEST/tfdata_$SERIES.csv"
}

case "$DATA" in
s3://*)
    LOCAL=${IMAGENET_DIR:-$HOME/imagenet}
    if [ -d "$LOCAL/train" ]; then
        IMAGENET_S3=$DATA series s3 "$DATA" "$LOCAL"
    else
        echo "no $LOCAL/train: skipping tf.data (S3), which lists the files from the local copy"
        series s3 "$DATA" ""
    fi
    ;;
*)
    series local "$DATA" "$DATA" --local
    [ -n "${IMAGENET_S3:-}" ] && series s3 "$IMAGENET_S3" "$DATA"
    ;;
esac

echo
echo "Plot with:  python plots/resnet_training.py --results results"
