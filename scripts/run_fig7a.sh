#!/usr/bin/env bash
# Figure 7a: retrieval-augmented generation.
#
#   bash scripts/run_fig7a.sh
#
# Needs the raydata-rag environment and 1 node with 8 GPUs; GPU_COUNT caps the
# sweep for smaller nodes (GPU_COUNT=2 runs the 1- and 2-GPU points).
# VLLM_EXTRA_ARGS is appended to every run; 24 GB GPUs need
# VLLM_EXTRA_ARGS="--max-model-len 4096 --gpu-memory-utilization 0.95 --enforce-eager".
# DATASET, KB and MODEL override the paths; see the README for how to build
# the knowledge base.
set -u

GPU_COUNT="${GPU_COUNT:-8}"
DATASET="${DATASET:-qa/web-train.json}"
KB="${KB:-kb}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
PROMPTS="${PROMPTS:-100000}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
BENCH=experiments/ray_data_eval/rag/benchmark_rag.py
DEST=${RESULTS_DIR:-results}/rag
mkdir -p "$DEST"

[ -f "${KB}_kb.index" ] || { echo "no ${KB}_kb.index; build the knowledge base first"; exit 1; }

# ray_data_dynamic joins an existing Ray with ray.init("auto"); on a fresh
# single node there is none, so start one.
ray status >/dev/null 2>&1 || { echo "starting a local Ray"; ray start --head --disable-usage-stats >/dev/null; }

run() {
    mode=$1; dp=$2
    echo "== $mode, data-parallel $dp =="
    python "$BENCH" --dataset "$DATASET" --kb-prefix "$KB" --model "$MODEL" \
        --mode "$mode" --num-prompts "$PROMPTS" --nprobe 256 --topk 5 \
        --data-parallel-size "$dp" --output-dir "$DEST" ${VLLM_EXTRA_ARGS:-}
}

for n in 1 2 4 8; do
    [ "$n" -le "$GPU_COUNT" ] && run ray_data_dynamic "$n"
done
run staged_batch 1

echo
echo "Job completion times are in $DEST/<timestamp>-<mode>-dp<N>-.../log.log"
echo "Plot with:  python plots/rag.py --results results"
