#!/usr/bin/env bash
# Figure 7b: video classification, four systems.
#
#   bash scripts/run_fig7b.sh
#
# Needs 4 GPU nodes and about 4 hours. Results land in
# results/video_classification/. Flink and Spark are not run here.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
BENCH=experiments/ray_data_eval/video_inference/ray_data_pipeline_map.py
OUT=video_inference_s3_g5_xlarge_batch_32
DEST=results/video_classification
mkdir -p "$DEST"
export RAY_DEDUP_LOGS=0

run() {
    name=$1; shift
    echo "== $name =="
    env "$@" python "$BENCH" --source s3 > "$OUT.out" 2>&1
    if [ -f "$OUT.csv" ]; then
        mv "$OUT.csv" "$DEST/$name.csv"
        echo "  wrote $DEST/$name.csv"
    else
        echo "  FAILED: no CSV; see $OUT.out"
    fi
}

run ray_data_dynamic
run ray_data_microbatch \
    RAY_DATA_CTX_SCHEDULING_POLICY=microbatch \
    RAY_DATA_CTX_MICROBATCH_SIZE=32 \
    RAY_DATA_CTX_MICROBATCH_GROUP_SIZE=1 \
    RAY_DATA_CTX_MICROBATCH_STAGE_BARRIER=strict
run ray_data_staged \
    RAY_DATA_CTX_SCHEDULING_POLICY=microbatch \
    RAY_DATA_CTX_MICROBATCH_SIZE=32 \
    RAY_DATA_CTX_MICROBATCH_GROUP_SIZE=4 \
    RAY_DATA_CTX_MICROBATCH_STAGE_BARRIER=relaxed
run cameo_llf \
    RAY_DATA_CTX_SCHEDULING_POLICY=llf_v2 \
    RAY_DATA_CTX_LLF_DISABLE_ADMISSION_CONTROL=True

echo
echo "Plot with:  python plots/video_classification.py --results results"
