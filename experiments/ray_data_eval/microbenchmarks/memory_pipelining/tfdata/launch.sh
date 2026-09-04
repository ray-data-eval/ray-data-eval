#!/usr/bin/env bash
# tf.data baseline for Figure 9.
#
#   bash tfdata/launch.sh <mem-limit-gb> [log-file]
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f "$ROOT/third_party/env.sh" ] && . "$ROOT/third_party/env.sh"
export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
export MALLOC_ARENA_MAX=2
mem_limit=${1:?mem limit in GB}
log_file=${2:-mem-limit-$mem_limit.log}

echo "Running $mem_limit GB -> $log_file"
python -u producer_consumer_gpu.py --mem-limit "$mem_limit" > "$log_file" 2>&1
