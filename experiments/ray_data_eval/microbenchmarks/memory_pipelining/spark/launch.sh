#!/usr/bin/env bash
# Spark baseline for Figure 9.
#
#   bash spark/launch.sh <mem-limit-gb> [log-file]
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
cd "$(dirname "${BASH_SOURCE[0]}")"
[ -f "$ROOT/third_party/env.sh" ] && . "$ROOT/third_party/env.sh"
SPARK_HOME="${SPARK_HOME:-$(python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')}"
export SPARK_HOME
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-$(command -v python)}"
export SPARK_EVENTS_PATH="$ROOT/logs/spark-events"
export SPARK_TRACE_EVENT_PATH="$ROOT/logs/spark-trace-events"
mkdir -p "$SPARK_EVENTS_PATH" "$SPARK_TRACE_EVENT_PATH"
export SPARK_EVENTS_FILEURL=file://$SPARK_EVENTS_PATH
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$SPARK_EVENTS_FILEURL"
export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=4 -Dspark.worker.resource.gpu.discoveryScript=$PWD/gpu_discovery.sh -Dspark.executor.instances=8 -Dspark.executor.cores=1"
export PYTHONPATH=$(dirname "$PWD"):$PYTHONPATH
mem_limit=${1:?mem limit in GB}
log_file=${2:-mem-limit-$mem_limit.log}

"$SPARK_HOME/sbin/stop-history-server.sh" >/dev/null 2>&1
"$SPARK_HOME/sbin/stop-master.sh" >/dev/null 2>&1
"$SPARK_HOME/sbin/stop-worker.sh" >/dev/null 2>&1
"$SPARK_HOME/sbin/start-history-server.sh"
"$SPARK_HOME/sbin/start-master.sh" --host localhost
"$SPARK_HOME/sbin/start-worker.sh" spark://localhost:7077
sleep 5

echo "Running $mem_limit GB -> $log_file"
python -u producer_consumer_gpu.py --mem-limit "$mem_limit" --stage-level-scheduling > "$log_file" 2>&1

"$SPARK_HOME/sbin/stop-history-server.sh" >/dev/null 2>&1
"$SPARK_HOME/sbin/stop-master.sh" >/dev/null 2>&1
"$SPARK_HOME/sbin/stop-worker.sh" >/dev/null 2>&1
