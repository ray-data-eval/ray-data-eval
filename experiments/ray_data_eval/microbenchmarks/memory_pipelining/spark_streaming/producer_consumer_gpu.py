import time
from pyspark.streaming import StreamingContext
from pyspark.streaming.listener import StreamingListener
from pyspark.resource.requests import TaskResourceRequests
from pyspark.resource import ResourceProfileBuilder
import argparse
from setting import (
    TIME_UNIT,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    NUM_CPUS,
    FRAME_SIZE_B,
    limit_cpu_memory,
)
from pyspark.sql import SparkSession
import os


def start_spark_streaming(mem_limit, stage_level_scheduling):
    # https://spark.apache.org/docs/latest/configuration.html

    spark_config = (
        SparkSession.builder.config("spark.driver.host", "127.0.0.1")
        .appName("Local Spark Example")
        .master("spark://localhost:7077")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
    )

    if not stage_level_scheduling:
        assert False, "Please use stage_level_scheduling"

        spark_config = (
            spark_config.config("spark.executor.cores", 1)
            .config("spark.cores.max", NUM_CPUS)
            .config("spark.executor.instances", NUM_CPUS)
            .config("spark.executor.memory", f"{int(mem_limit * 1024 / 8)}m")
            # Allocate 1 GPU per executor and 1 GPU per task.
            .config("spark.executor.resource.gpu.amount", 1)
            .config("spark.task.resource.gpu.amount", 1)
            .config("spark.driver.memory", "1g")
        )
    else:
        if mem_limit <= 16:
            spark_config = (
                spark_config.config("spark.dynamicAllocation.enabled", "false")
                .config("spark.executor.instances", 1)
                .config("spark.executor.cores", 1)
                .config("spark.executor.memory", "1g")
                # Allocate 1 GPU per executor.
                .config("spark.executor.resource.gpu.amount", 1)
                .config("spark.driver.memory", "1g")
            )
        else:
            spark_config = (
                spark_config.config("spark.dynamicAllocation.enabled", "false")
                .config("spark.executor.instances", 4)
                .config("spark.executor.cores", 2)
                .config("spark.executor.memory", f"{int(mem_limit / 4 * 1024)}m")
                # Allocate 1 GPU per executor.
                .config("spark.executor.resource.gpu.amount", 1)
                .config("spark.driver.memory", "1g")
            )

    spark = spark_config.getOrCreate()
    sc = spark.sparkContext
    ssc = StreamingContext(sc, 0.1)  # Set batch interval to 0.1 seconds
    return sc, ssc


def producer(row):
    print("producer")
    time.sleep(TIME_UNIT * 10)
    for j in range(FRAMES_PER_VIDEO):
        data = b"1" * FRAME_SIZE_B
        yield (data, row * FRAMES_PER_VIDEO + j)


def consumer(row):
    print("consumer")
    time.sleep(TIME_UNIT)
    data = b"2" * FRAME_SIZE_B
    return (data,)


def inference(row):
    print("inference")
    time.sleep(TIME_UNIT)
    return 1


def run_spark_data(ssc, mem_limit, stage_level_scheduling):
    if stage_level_scheduling:
        # For the CPU stages, request 1 CPU and 0.5 GPU. This will run 8 concurrent tasks.
        # cpu_task_requests = TaskResourceRequests().cpus(1).resource("gpu", 0.5)
        # For the GPU stages, request 1 CPU and 1 GPU. This will run 4 concurrent tasks.
        gpu_task_requests = TaskResourceRequests().cpus(1).resource("gpu", 1)

        builder = ResourceProfileBuilder()
        # cpu_task_profile = builder.require(cpu_task_requests).build
        gpu_task_profile = builder.require(gpu_task_requests).build

    start = time.perf_counter()
    BATCH_SIZE = NUM_VIDEOS // 5 if mem_limit >= 10 else 4
    rdd_queue = [
        ssc.sparkContext.range(i, i + BATCH_SIZE) for i in range(0, NUM_VIDEOS, BATCH_SIZE)
    ]
    input_stream = ssc.queueStream(rdd_queue)

    # Apply the producer UDF
    producer_stream = input_stream.flatMap(producer)

    def process_batch(rdd):
        if rdd.isEmpty():
            run_time = time.perf_counter() - start
            print(f"\nRun time: {run_time:.2f} seconds")
            ssc.stop(stopSparkContext=True, stopGraceFully=True)
            return

        consumer_rdd = rdd.map(lambda x: consumer(x))

        if stage_level_scheduling:
            # Call repartition to force a new stage for stage level scheduling
            inference_rdd = consumer_rdd.repartition(consumer_rdd.count())
            inference_rdd = inference_rdd.map(lambda x: inference(x)).withResources(
                gpu_task_profile
            )
        else:
            inference_rdd = consumer_rdd.map(lambda x: inference(x))

        total_processed = inference_rdd.count()
        print(f"Total length of data processed in batch: {total_processed:,}")

    producer_stream.foreachRDD(process_batch)


def bench(mem_limit, stage_level_scheduling):
    sc, ssc = start_spark_streaming(mem_limit, stage_level_scheduling)
    listener = StreamingListener()
    ssc.addStreamingListener(listener)

    run_spark_data(ssc, mem_limit, stage_level_scheduling)
    ssc.start()
    ssc.awaitTerminationOrTimeout(3600)
    print("Stopping streaming context.")
    ssc.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=30
    )
    parser.add_argument(
        "--stage-level-scheduling",
        action="store_true",
        required=False,
        help="Whether to enable stage level scheduling",
        default=False,
    )
    args = parser.parse_args()
    limit_cpu_memory(args.mem_limit)

    # import multiprocessing
    # Start memory usage logging in a separate process
    # logging_process = multiprocessing.Process(target=log_memory_usage_process, args=(2, args.mem_limit))  # Log every 2 seconds
    # logging_process.start()

    # assert not args.stage_level_scheduling, "Receive error: TaskResourceProfiles are only supported for Standalone, Yarn and Kubernetes cluster for now when dynamic allocation is disabled."

    bench(args.mem_limit, args.stage_level_scheduling)
    # logging_process.terminate()
