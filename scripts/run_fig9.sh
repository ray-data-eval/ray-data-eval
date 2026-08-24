#!/usr/bin/env bash
# Run the Figure 9 memory sweep. Needs the raydata-fig9 environment from
# scripts/setup/install_ray_data_fig9.sh, and about 35 minutes.
#
#   bash scripts/run_fig9.sh [outdir]
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:-results/memory_pipelining}"
BENCH=experiments/ray_data_eval/microbenchmarks/memory_pipelining/raydata/producer_consumer_gpu.py

export PYTHONPATH="$PWD/experiments"
export NUM_VIDEOS=160 FRAMES_PER_VIDEO=5 FRAME_SIZE_MB=100
export NUM_CPUS=8 NUM_GPUS=4 TIME_UNIT=0.5

mkdir -p "$OUT"
echo "mem_limit_gb,jct_s" > "$OUT/ray_data.csv"

for mem in 8 10 12 14 16 6; do
  rm -rf /tmp/ray/session_* 2>/dev/null || true
  python "$BENCH" --mem-limit "$mem" > "$OUT/mem_${mem}.out" 2>&1 || true
  t=$(grep -oE 'Total time: [0-9.]+' "$OUT/mem_${mem}.out" | grep -oE '[0-9.]+' | head -1)
  echo "${mem},${t:-}" >> "$OUT/ray_data.csv"
  printf '  %2s GB  %s s\n' "$mem" "${t:-FAILED}"
done

echo
echo "Wrote $OUT/ray_data.csv"
echo "Published values: 8-16 GB about 199-200 s, 6 GB about 945 s."
echo "Plot with:  python plots/memory_pipelining.py --results \$(dirname $OUT)"
