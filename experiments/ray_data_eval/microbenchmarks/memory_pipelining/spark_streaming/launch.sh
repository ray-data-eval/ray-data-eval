set -ex

# Paths come from third_party/env.sh (scripts/setup/setup_baselines.sh), falling back
# to the installed pyspark and the active interpreter.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)
[ -f "$ROOT/third_party/env.sh" ] && . "$ROOT/third_party/env.sh"
SPARK_HOME="${SPARK_HOME:-$(python -c 'import os, pyspark; print(os.path.dirname(pyspark.__file__))')}"
export SPARK_HOME
export PYSPARK_PYTHON="${PYSPARK_PYTHON:-$(command -v python)}"
export SPARK_EVENTS_PATH="$ROOT/logs/spark-events"
export SPARK_TRACE_EVENT_PATH="$ROOT/logs/spark-trace-events"
mkdir -p "$SPARK_EVENTS_PATH" "$SPARK_TRACE_EVENT_PATH"
export SPARK_EVENTS_FILEURL=file://$SPARK_EVENTS_PATH
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$SPARK_EVENTS_FILEURL"

# export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=4 -Dspark.worker.resource.gpu.discoveryScript=./gpu_discovery.sh -Dspark.executor.instances=8 -Dspark.executor.cores=1"

"$SPARK_HOME/sbin/stop-history-server.sh"
"$SPARK_HOME/sbin/start-history-server.sh"

export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
python -u producer_consumer_gpu.py --mem-limit 4 > mem-limit-4.log 2>&1
# python -u producer_consumer_gpu.py --mem-limit 4 > mem-limit-4.log 2>&1
