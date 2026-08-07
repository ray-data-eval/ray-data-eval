set -ex

export SPARK_EVENTS_PATH=/home/ubuntu/ray-data-eval/logs/spark-events
export SPARK_TRACE_EVENT_PATH=/home/ubuntu/ray-data-eval/logs/spark-trace-events
export SPARK_EVENTS_FILEURL=file://$SPARK_EVENTS_PATH
export SPARK_HOME=/home/ubuntu/miniconda3/envs/ray-data/lib/python3.10/site-packages/pyspark
export PYSPARK_PYTHON=/home/ubuntu/miniconda3/envs/ray-data/bin/python
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$SPARK_EVENTS_FILEURL"

# export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=4 -Dspark.worker.resource.gpu.discoveryScript=./gpu_discovery.sh -Dspark.executor.instances=8 -Dspark.executor.cores=1"

/home/ubuntu/miniconda3/envs/ray-data/lib/python3.10/site-packages/pyspark/sbin/stop-history-server.sh
/home/ubuntu/miniconda3/envs/ray-data/lib/python3.10/site-packages/pyspark/sbin/start-history-server.sh

export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
python -u producer_consumer_gpu.py --mem-limit 4 > mem-limit-4.log 2>&1
# python -u producer_consumer_gpu.py --mem-limit 4 > mem-limit-4.log 2>&1
