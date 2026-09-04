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

# One row per (mode, GPU count) from the latest run of each, for plots/rag.py.
python - "$DEST" <<'PY'
import csv, glob, os, re, sys
dest = sys.argv[1]
def grab(text, key):
    m = re.findall(rf"{key}: ([0-9.]+) s", text)
    return float(m[-1]) if m else None
latest = {}
for d in sorted(glob.glob(os.path.join(dest, "*-dp*-nprobe*"))):
    m = re.search(r"-(ray_data_dynamic|staged_batch)-dp(\d+)-", d)
    if m and os.path.exists(os.path.join(d, "log.log")):
        latest[(m.group(1), int(m.group(2)))] = d
rows = {"ray_data_dynamic": [], "staged_batch": []}
for (mode, dp), d in sorted(latest.items()):
    text = open(os.path.join(d, "log.log")).read()
    jct = grab(text, "Elapsed Time")
    if jct is None:
        continue
    row = {"num_gpus": dp, "jct_min": round(jct / 60, 3), "jct_s": jct}
    if mode == "staged_batch":
        row.update(encoding_s=grab(text, "Encoding time"),
                   retrieval_s=grab(text, "Retrieving time"),
                   generation_s=grab(text, "Generation time"))
    rows[mode].append(row)
for mode, name in (("ray_data_dynamic", "ray_data_dynamic.csv"), ("staged_batch", "ray_data_staged.csv")):
    if rows[mode]:
        with open(os.path.join(dest, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[mode][0]))
            w.writeheader(); w.writerows(rows[mode])
        print(f"  wrote {dest}/{name}")
PY

echo
echo "Plot with:  python plots/rag.py --results results"
