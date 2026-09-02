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
export PYTHONPATH=$ROOT/experiments

# The benchmark joins an existing Ray with ray.init("auto"); on a fresh
# single node there is none, so start one.
ray status >/dev/null 2>&1 || { echo "starting a local Ray"; ray start --head --disable-usage-stats >/dev/null; }

cd experiments/ray_data_eval/image_training/e2e_training

# Both trainers name their outputs after the published S3/g5 run whatever
# the data source; the series is decided here and the files renamed for
# plots/resnet_training.py.
case "$DATA" in s3://*) SERIES=s3 ;; *) SERIES=local ;; esac
RD=e2e_training_s3_g5_xlarge_batch_256
TF=tf_data_e2e_training_g5_xlarge_batch_256
rm -f "$RD.csv" "$TF.csv"

check() {
    # A phase that wrote no CSV failed, whatever its exit code said.
    [ -f "$1.csv" ] && return 0
    echo "  FAILED: no $1.csv produced; last lines of $1.out:"
    tail -8 "$1.out" | sed "s/^/    /"
    exit 1
}

# The published series is one epoch; the trainer defaults to 90.
echo "== Ray Data =="
python ray_data_e2e_training.py -a resnet50 -b 128 --epochs 1 "$DATA" > "$RD.out" 2>&1
check "$RD"
cp "$RD.csv" "$DEST/ray_data_$SERIES.csv"
echo "== tf.data =="
python tf_data_e2e_training.py -b 256 --local "$DATA" > "$TF.out" 2>&1
check "$TF"
cp "$TF.csv" "$DEST/tfdata_$SERIES.csv"

echo
echo "Wrote $DEST/ray_data_$SERIES.csv and $DEST/tfdata_$SERIES.csv"
echo "Plot with:  python plots/resnet_training.py --results results"
