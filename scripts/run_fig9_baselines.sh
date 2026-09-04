#!/usr/bin/env bash
# Run the Figure 9 baselines (tf.data, Spark, Flink) over the memory sweep.
# Needs scripts/setup/setup_baselines.sh, and about 2 hours.
#
#   bash scripts/run_fig9_baselines.sh [outdir] [system ...]
#   MEMS="16 8" bash scripts/run_fig9_baselines.sh   # subset of the sweep
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
OUT="${1:-results/memory_pipelining}"; shift || true
SYSTEMS="${*:-tfdata spark flink}"
MP=experiments/ray_data_eval/microbenchmarks/memory_pipelining

export NUM_VIDEOS=160 FRAMES_PER_VIDEO=5 FRAME_SIZE_MB=100
export NUM_CPUS=8 NUM_GPUS=4 TIME_UNIT=0.5

mkdir -p "$OUT"
for sys in $SYSTEMS; do
  echo "== $sys =="
  echo "mem_limit_gb,jct_s,host_mem_growth_mb" > "$OUT/$sys.csv"
  for mem in ${MEMS:-16 14 12 10 8 6}; do
    log="$PWD/$OUT/${sys}_mem_${mem}.out"
    timeout 2400 bash "$MP/$sys/launch.sh" "$mem" "$log" >/dev/null 2>&1 || true
    t=$(grep -aoE 'Run time: [0-9.]+' "$log" | grep -oE '[0-9.]+' | head -1 || true)
    first=$(grep -aoE 'Used Memory: [0-9.]+' "$log" | grep -oE '[0-9.]+' | head -1 || true)
    peak=$(grep -aoE 'Used Memory: [0-9.]+' "$log" | grep -oE '[0-9.]+' | sort -n | tail -1 || true)
    # Greyed out when the run's host memory grew past the limit; KEEP_EXCEEDED=1 keeps it.
    grew=$(python3 -c "print(round(${peak:-0}-${first:-0}))")
    if [ "$grew" -gt $((mem*1024)) ] && [ -z "${KEEP_EXCEEDED:-}" ]; then shown=""; else shown="$t"; fi
    echo "${mem},${shown:-},${grew}" >> "$OUT/$sys.csv"
    printf '  %2s GB  %-10s +%s MB host memory%s\n' "$mem" "${t:-FAILED} s" "$grew" "$([ "$grew" -gt $((mem*1024)) ] && echo '  (exceeded the limit)')"
  done
done

echo
echo "Wrote $OUT/{$(echo $SYSTEMS | tr ' ' ',')}.csv"
echo "Plot with:  python plots/memory_pipelining.py --results \$(dirname $OUT)"
