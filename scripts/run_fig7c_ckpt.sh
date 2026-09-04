#!/usr/bin/env bash
# Figure 7c, dashed curves: Ray Data emulating global checkpoint-restart.
# The job takes an "empty" checkpoint every 6 minutes; on a failure it is
# killed and restarted from the last checkpoint, redoing the work since.
#
#   bash scripts/run_fig7c_ckpt.sh executor  <cpu-node-ssh> <head-ip>
#   bash scripts/run_fig7c_ckpt.sh node      <cpu-node-ssh> <head-ip>
#
# Writes <mode>_failure_ckpt_seg<N>.csv, one per (re)start, to
# results/fault_tolerance/.
set -u

MODE="${1:?usage: run_fig7c_ckpt.sh <executor|node> <cpu-node-ssh> <head-ip>}"
CPU_NODE="${2:?missing cpu-node ssh target}"
HEAD_IP="${3:?missing head ip}"

FAILURE_AT_S="${FAILURE_AT_S:-900}"
RECOVER_AT_S="${RECOVER_AT_S:-1800}"
CKPT_INTERVAL_S="${CKPT_INTERVAL_S:-360}"
CONDA_ENV="${CONDA_ENV:-raydata}"
RAY_START_ARGS="${RAY_START_ARGS:-}"

remote() {
    ssh -o BatchMode=yes "$CPU_NODE" \
        "source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $*"
}
BENCH=experiments/ray_data_eval/video_inference/ray_data_pipeline_map.py
OUT=video_inference_s3_g5_xlarge_batch_32

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export RAY_DEDUP_LOGS=0
DEST=${RESULTS_DIR:-results}/fault_tolerance
mkdir -p "$DEST"

T0=""            # start of the whole experiment (first measured run)
SKIP=0           # inputs already processed at the last checkpoint
SEG=0

start_segment() {
    rm -f "$OUT.csv" "$OUT.out"
    RAY_DATA_SKIP_FILES=$SKIP python "$BENCH" --source s3 > "$OUT.out" 2>&1 &
    BENCH_PID=$!
    local t=""
    for _ in $(seq 1 600); do
        t=$(grep -a "\[Start Time\]" "$OUT.out" 2>/dev/null | tail -1 | awk '{print int($3)}')
        [ -n "$t" ] && break
        kill -0 "$BENCH_PID" 2>/dev/null || { echo "  segment $SEG exited during warmup"; tail -5 "$OUT.out"; exit 1; }
        sleep 2
    done
    [ -n "$t" ] || { echo "  segment $SEG never started"; kill "$BENCH_PID"; exit 1; }
    SEG_T0=$t; [ -n "$T0" ] || T0=$t
    echo "== segment $SEG started at t+$(( SEG_T0 - T0 ))s, skipping $SKIP inputs =="
}

finish_segment() {
    # save the segment's CSV on the experiment's clock
    ray_data_postprocess
    python - "$OUT.csv" "$DEST/${MODE}_failure_ckpt_seg$SEG.csv" $(( SEG_T0 - T0 )) <<'PY'
import sys, csv
src, dst, shift = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = list(csv.DictReader(open(src)))
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["time_from_start", "number_of_rows_finished"])
    w.writeheader()
    for r in rows:
        w.writerow({"time_from_start": float(r["time_from_start"]) + shift,
                    "number_of_rows_finished": r["number_of_rows_finished"]})
PY
    cp "$OUT.out" "$DEST/${MODE}_failure_ckpt_seg$SEG.out"
    echo "  wrote $DEST/${MODE}_failure_ckpt_seg$SEG.csv"
    SEG=$((SEG + 1))
}

ray_data_postprocess() {
    # the benchmark writes its CSV on a normal exit; after a kill, do it here
    [ -f "$OUT.csv" ] || python - "$OUT.out" <<'PY'
import sys
sys.path.insert(0, "experiments/ray_data_eval/video_inference")
from ray_data_pipeline_helpers import postprocess
postprocess(sys.argv[1])
PY
}

kill_segment() {
    # the failure: the job dies and must restart from its last checkpoint
    kill "$BENCH_PID" 2>/dev/null; wait "$BENCH_PID" 2>/dev/null
    local ran=$(( $(date +%s) - SEG_T0 ))
    local at=$(( ran / CKPT_INTERVAL_S * CKPT_INTERVAL_S ))
    ray_data_postprocess
    local done_at
    done_at=$(python - "$OUT.csv" "$at" <<'PY'
import sys, csv
rows = [(float(r["time_from_start"]), int(r["number_of_rows_finished"])) for r in csv.DictReader(open(sys.argv[1]))]
at = float(sys.argv[2]); t0 = rows[0][0]
print(max([n for t, n in rows if t - t0 <= at] or [0]))
PY
)
    echo "  segment $SEG ran ${ran}s; last checkpoint at ${at}s had $done_at rows done"
    SKIP=$(( SKIP + done_at ))
}

wait_until() {
    while [ $(( $(date +%s) - T0 )) -lt "$1" ]; do
        kill -0 "$BENCH_PID" 2>/dev/null || return 1
        sleep 5
    done
}

start_segment
wait_until "$FAILURE_AT_S" || { echo "  finished before the failure"; wait "$BENCH_PID"; finish_segment; exit 0; }
case "$MODE" in
executor)
    echo "== t+$(( $(date +%s) - T0 ))s: executor failure; global rollback =="
    ssh -o BatchMode=yes "$CPU_NODE" "pkill -9 -n -f '^ray::' || true"
    kill_segment; finish_segment
    start_segment
    ;;
node)
    echo "== t+$(( $(date +%s) - T0 ))s: node failure; global rollback =="
    remote "ray stop" 2>&1 | tail -1 | sed 's/^/  /'
    kill_segment; finish_segment
    start_segment
    if wait_until "$RECOVER_AT_S"; then
        echo "== t+$(( $(date +%s) - T0 ))s: node rejoins; global rollback =="
        remote "ray start --address=$HEAD_IP:6379 $RAY_START_ARGS" 2>&1 | tail -1 | sed 's/^/  /'
        kill_segment; finish_segment
        start_segment
    fi
    ;;
*) echo "unknown mode: $MODE"; kill "$BENCH_PID"; exit 1 ;;
esac

echo "== waiting for the final segment =="
wait "$BENCH_PID"; echo "  finished after $(( ($(date +%s) - T0) / 60 )) min"
finish_segment
