#!/usr/bin/env bash
# Figure 10b: scalability against node count.
#
#   bash scripts/run_fig10b.sh <num-nodes>
#
# Throughput is recorded against node count, so run this, add nodes, and run it
# again. --num_nodes must match the cluster or the points are not comparable.
set -u

NODES="${1:?usage: run_fig10b.sh <num-nodes>}"
SIZE="${SIZE:-5}"
CPUS_PER_NODE="${CPUS_PER_NODE:-8}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
BENCH=experiments/ray_data_eval/microbenchmarks/scalability/benchmark_scalability.py
DEST=results/scalability
mkdir -p "$DEST"

run() {
    mode=$1; name=$2
    echo "== $name, $NODES nodes =="
    python "$BENCH" --mode "$mode" --size "$SIZE" \
        --cpus_per_node "$CPUS_PER_NODE" --num_nodes "$NODES" \
        2>&1 | tee "$DEST/${name}_${NODES}.out"
    t=$(grep -oE "Throughput in GB/s: [0-9.]+" "$DEST/${name}_${NODES}.out" \
        | grep -oE "[0-9.]+" | tail -1)
    [ -f "$DEST/$name.csv" ] || echo "num_nodes,throughput_gbs" > "$DEST/$name.csv"
    echo "${NODES},${t:-}" >> "$DEST/$name.csv"
    echo "  $NODES nodes: ${t:-FAILED} GB/s"
}

run ray_data           ray_data
run ray_original       ray
run ray_original_streaming ray_streaming

echo
echo "Plot with:  python plots/scalability.py --results results"
