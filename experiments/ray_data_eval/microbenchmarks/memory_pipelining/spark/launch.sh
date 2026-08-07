export SPARK_EVENTS_PATH=/home/ray/default/ray-data-eval/logs/spark-events
export SPARK_TRACE_EVENT_PATH=/home/ray/default/ray-data-eval/logs/spark-trace-events
export SPARK_EVENTS_FILEURL=file://$SPARK_EVENTS_PATH
export SPARK_HOME=./spark-3.5.1-bin-hadoop3
export PYSPARK_PYTHON=/home/ray/anaconda3/bin/python
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=$SPARK_EVENTS_FILEURL"

# export SPARK_WORKER_OPTS="-Dspark.worker.resource.gpu.amount=4 -Dspark.worker.resource.gpu.discoveryScript=./gpu_discovery.sh -Dspark.executor.instances=8 -Dspark.executor.cores=1"
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH

./spark-3.5.1-bin-hadoop3/sbin/stop-history-server.sh
./spark-3.5.1-bin-hadoop3/sbin/start-history-server.sh
python producer_consumer_gpu.py --stage-level-scheduling
