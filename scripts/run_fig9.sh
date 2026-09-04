#!/usr/bin/env bash
# Run the Figure 9 memory sweep. Needs the raydata-fig9 environment from
# scripts/setup/install_ray_data_fig9.sh, and about 35 minutes per variant.
#
#   bash scripts/run_fig9.sh [outdir] [variant ...]
#
# Variants: default (Ray Data), no_adapt (Ray Data(-Adapt.)), no_part
# (Ray Data(-Part.)). MEMS="16 8" runs a subset of the sweep.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:-results/memory_pipelining}"; shift || true
VARIANTS="${*:-default}"
BENCH=experiments/ray_data_eval/microbenchmarks/memory_pipelining/raydata/producer_consumer_gpu.py

export PYTHONPATH="$PWD/experiments"
export NUM_VIDEOS=160 FRAMES_PER_VIDEO=5 FRAME_SIZE_MB=100
export NUM_CPUS=8 NUM_GPUS=4 TIME_UNIT=0.5

mkdir -p "$OUT"
for variant in $VARIANTS; do
  case "$variant" in
    default) csv=ray_data.csv ;;
    no_adapt) csv=ray_data_no_adapt.csv ;;
    no_part) csv=ray_data_no_part.csv ;;
    *) echo "unknown variant: $variant (default, no_adapt, no_part)"; exit 1 ;;
  esac
  echo "== $variant =="
  echo "mem_limit_gb,jct_s,host_mem_growth_mb" > "$OUT/$csv"
  for mem in ${MEMS:-8 10 12 14 16 6}; do
    rm -rf /tmp/ray/session_* 2>/dev/null || true
    log="$OUT/${variant}_mem_${mem}.out"
    python "$BENCH" --variant "$variant" --mem-limit "$mem" > "$log" 2>&1 || true
    t=$(grep -oE 'Total time: [0-9.]+' "$log" | grep -oE '[0-9.]+' | head -1 || true)
    first=$(grep -aoE 'Used Memory: [0-9.]+' "$log" | grep -oE '[0-9.]+' | head -1 || true)
    peak=$(grep -aoE 'Used Memory: [0-9.]+' "$log" | grep -oE '[0-9.]+' | sort -n | tail -1 || true)
    # Greyed out when the run's host memory grew past the limit; KEEP_EXCEEDED=1 keeps it.
    grew=$(python3 -c "print(round(${peak:-0}-${first:-0}))")
    if [ "$grew" -gt $((mem*1024)) ] && [ -z "${KEEP_EXCEEDED:-}" ]; then shown=""; else shown="$t"; fi
    echo "${mem},${shown:-},${grew}" >> "$OUT/$csv"
    printf '  %2s GB  %-10s +%s MB host memory%s\n' "$mem" "${t:-FAILED} s" "$grew" "$([ "$grew" -gt $((mem*1024)) ] && echo '  (exceeded the limit)')"
  done
  echo "Wrote $OUT/$csv"
done

echo
echo "Published values, Ray Data: 8-16 GB about 199-200 s, 6 GB about 290 s (260-300 s depending on instance)."
echo "Plot with:  python plots/memory_pipelining.py --results \$(dirname $OUT)"
