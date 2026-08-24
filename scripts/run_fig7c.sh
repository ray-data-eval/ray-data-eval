#!/usr/bin/env bash
# Figure 7c: inject a failure into the video classification run.
#
#   bash scripts/run_fig7c.sh executor  <cpu-node-ssh> <head-ip>
#   bash scripts/run_fig7c.sh node      <cpu-node-ssh> <head-ip>
#
# Run this on the GPU node. <cpu-node-ssh> is an ssh target for the CPU-only
# node; <head-ip> is this node's address, used to rejoin it.
#
# Executor failure kills one worker process. Node failure stops Ray on the
# CPU-only node and rejoins it 15 minutes later, which is what the paper's
# "disconnects and reconnects" means -- the instance itself keeps running.
#
# FAILURE_AT_S and RECOVER_AT_S override the timings for a quick check; the
# published run uses the defaults. CONDA_ENV names the environment to activate
# on the CPU node, and RAY_START_ARGS is appended when rejoining it.
set -u

MODE="${1:?usage: run_fig7c.sh <executor|node> <cpu-node-ssh> <head-ip>}"
CPU_NODE="${2:?missing cpu-node ssh target}"
HEAD_IP="${3:?missing head ip}"

FAILURE_AT_S="${FAILURE_AT_S:-900}"     # t=15 min
RECOVER_AT_S="${RECOVER_AT_S:-1800}"    # t=30 min
CONDA_ENV="${CONDA_ENV:-raydata}"
RAY_STOP_ARGS="${RAY_STOP_ARGS:-}"
RAY_START_ARGS="${RAY_START_ARGS:-}"

# ssh does not start a login shell with conda on PATH, so activate it by hand.
remote() {
    ssh -o BatchMode=yes "$CPU_NODE" \
        "source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $*"
}
BENCH=experiments/ray_data_eval/video_inference/ray_data_pipeline_map.py
OUT=video_inference_s3_g5_xlarge_batch_32

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export RAY_DEDUP_LOGS=0

echo "== starting the benchmark =="
python "$BENCH" --source s3 > "$OUT.out" 2>&1 &
BENCH_PID=$!

# Anchor the clock to the "[Start Time]" the benchmark prints, not to launch:
# the warmup comes first, and the figure's x axis starts after it.
echo "  waiting for the measured run to start"
T0=""
for _ in $(seq 1 600); do
    T0=$(grep -a "\[Start Time\]" "$OUT.out" 2>/dev/null | tail -1 | awk '{print int($3)}')
    [ -n "$T0" ] && break
    kill -0 "$BENCH_PID" 2>/dev/null || { echo "  benchmark exited during warmup"; wait "$BENCH_PID"; exit 1; }
    sleep 2
done
[ -n "$T0" ] || { echo "  benchmark never reached the measured run"; kill "$BENCH_PID"; exit 1; }
echo "  measured run started at $(( $(date +%s) - T0 ))s ago; injecting at t+${FAILURE_AT_S}s"
wait_until() {
    local target=$1
    while [ $(( $(date +%s) - T0 )) -lt "$target" ]; do
        kill -0 "$BENCH_PID" 2>/dev/null || { echo "  benchmark exited early"; return 1; }
        sleep 5
    done
}

if ! wait_until "$FAILURE_AT_S"; then wait "$BENCH_PID"; exit 1; fi

case "$MODE" in
executor)
    echo "== t+$(( $(date +%s) - T0 ))s: killing one worker process on $CPU_NODE =="
    # -n takes the newest match, so exactly one worker dies. The pattern only
    # matches Ray workers, which set their process title to "ray::<task>".
    ssh -o BatchMode=yes "$CPU_NODE" \
        "pkill -9 -n -f '^ray::' && echo '  killed one worker' || echo '  no worker matched'"
    ;;
node)
    echo "== t+$(( $(date +%s) - T0 ))s: disconnecting $CPU_NODE =="
    # Graceful, not --force: a deregistered node tells the cluster its objects
    # are gone, so dependent tasks fail and re-execute at once. Under SIGKILL the
    # loss is discovered through failed fetches and recovery stalls for minutes.
    remote "ray stop ${RAY_STOP_ARGS}" 2>&1 | tail -2 | sed 's/^/  /'

    if wait_until "$RECOVER_AT_S"; then
        echo "== t+$(( $(date +%s) - T0 ))s: rejoining $CPU_NODE =="
        remote "ray start --address=$HEAD_IP:6379 $RAY_START_ARGS" 2>&1 | tail -2 | sed 's/^/  /'
    fi
    ;;
*)
    echo "unknown mode: $MODE (expected 'executor' or 'node')"; kill "$BENCH_PID"; exit 1 ;;
esac

echo "== waiting for the run to finish =="
wait "$BENCH_PID"
echo "  benchmark exited $?  after $(( ($(date +%s) - T0) / 60 )) min"

DEST=results/fault_tolerance
mkdir -p "$DEST"
[ -f "$OUT.csv" ] && mv "$OUT.csv" "$DEST/${MODE}_failure.csv" && echo "  wrote $DEST/${MODE}_failure.csv"
