#!/usr/bin/env bash
# Figure 10a: partition size sweep.
#
#   bash scripts/run_fig10a.sh
#
# Needs 8 CPU cores, no GPU, no cluster. About 20 minutes.
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DEST=${RESULTS_DIR:-results}/partitioning
mkdir -p "$DEST"

# The benchmark asks for a 15 GB object store, so /dev/shm has to have that
# much free. A previous run that was killed leaves its plasma files behind.
STORE_GB="${PARTITION_STORE_GB:-15}"
FREE_SHM=$(df -B1 --output=avail /dev/shm | tail -1)
TOTAL_SHM=$(df -B1 --output=size /dev/shm | tail -1)
NEED=$(( (STORE_GB + 1) * 1000000000 ))
if [ "$FREE_SHM" -lt "$NEED" ]; then
    echo "only $((FREE_SHM / 1000000000)) GB free in /dev/shm; the ${STORE_GB} GB object store does not fit."
    if [ "$TOTAL_SHM" -lt "$NEED" ]; then
        echo "This machine's /dev/shm is too small for the full benchmark; use an"
        echo "8-vCPU, 32 GB node (e.g. m7i.2xlarge), or run the short version with a"
        echo "smaller store:  PARTITION_STORE_GB=4 PARTITION_NUM_ROWS=2048 PARTITION_SIZES=1,64,1024 $0"
    else
        echo "A Ray left over from an earlier experiment is holding it. Clear it with:"
        echo "    ray stop --force && rm -rf /tmp/ray/session_*"
    fi
    exit 1
fi

python experiments/ray_data_eval/microbenchmarks/partitioning/raydata.py \
    2>&1 | tee "$DEST/raydata.out"

# The benchmark prints one line per size; turn them into the CSV the plot reads.
echo "num_rows_in_block,duration_s" > "$DEST/ray_data.csv"
grep -oE "num_rows_in_block=[0-9]+, duration=[0-9.]+" "$DEST/raydata.out" \
    | sed -E 's/num_rows_in_block=([0-9]+), duration=([0-9.]+)/\1,\2/' \
    | tail -n +2 >> "$DEST/ray_data.csv"

# tee hides the benchmark's exit status, so check the output instead: an empty
# CSV used to be reported as a successful run.
if [ "$(wc -l < "$DEST/ray_data.csv")" -lt 2 ]; then
    echo "the sweep produced no measurements; the benchmark failed. Last lines:"
    tail -15 "$DEST/raydata.out" | sed 's/^/    /'
    exit 1
fi

echo
echo "Wrote $DEST/ray_data.csv"
echo "Plot with:  python plots/partitioning.py --results results"
