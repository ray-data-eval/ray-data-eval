import time

from pyspark.sql import SparkSession
from pyspark import StorageLevel
from pyspark.resource.requests import TaskResourceRequests
from pyspark.resource import ResourceProfileBuilder
import os
import argparse

from setting import TIME_UNIT, NUM_CPUS, FRAMES_PER_VIDEO, NUM_VIDEOS, FRAME_SIZE_B, busy_loop


def start_spark(stage_level_scheduling: bool):
    spark_config = (
        SparkSession.builder.config("spark.driver.host", "127.0.0.1")
        .appName("Local Spark Example")
        .master("spark://127.0.0.1:7077")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", os.getenv("SPARK_EVENTS_FILEURL"))
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
    )
    if not stage_level_scheduling:
        spark_config = (
            spark_config.config("spark.executor.cores", 1)
            .config("spark.cores.max", NUM_CPUS)
            .config("spark.executor.instances", NUM_CPUS)
            # Allocate 1 GPU per executor and 1 GPU per task.
            .config("spark.executor.resource.gpu.amount", 1)
            .config("spark.task.resource.gpu.amount", 1)
        )
    else:
        spark_config = (
            spark_config.config("spark.dynamicAllocation.enabled", "false")
            .config("spark.executor.instances", 4)
            .config("spark.executor.cores", 2)
            .config("spark.executor.memory", "4g")
            # Allocate 1 GPU per executor.
            .config("spark.executor.resource.gpu.amount", 1)
        )

    spark = spark_config.getOrCreate()
    return spark


def producer_udf(row):
    busy_loop(TIME_UNIT * 10)
    for j in range(FRAMES_PER_VIDEO):
        data = b"1" * FRAME_SIZE_B
        assert len(data) == FRAME_SIZE_B
        yield (data, row * FRAMES_PER_VIDEO + j)


def consumer_udf(row):
    busy_loop(TIME_UNIT)
    data = b"2" * FRAME_SIZE_B
    return (data,)


def inference_udf(row):
    busy_loop(TIME_UNIT)
    return 1


def run_spark_data(
    spark, stage_level_scheduling: bool = False, cache: bool = False, cache_disk: bool = False
):
    start = time.perf_counter()

    if stage_level_scheduling:
        # For the CPU stages, request 1 CPU and 0.5 GPU. This will run 8 concurrent tasks.
        # cpu_task_requests = TaskResourceRequests().cpus(1).resource("gpu", 0.5)
        # For the GPU stages, request 1 CPU and 1 GPU. This will run 4 concurrent tasks.
        gpu_task_requests = TaskResourceRequests().cpus(1).resource("gpu", 1)

        builder = ResourceProfileBuilder()
        # cpu_task_profile = builder.require(cpu_task_requests).build
        gpu_task_profile = builder.require(gpu_task_requests).build

    rdd = spark.sparkContext.range(
        start=0, end=NUM_VIDEOS, numSlices=NUM_VIDEOS
    )  # Set NUM_VIDEOS as the number of partitions.

    producer_rdd = rdd.flatMap(producer_udf)
    # if stage_level_scheduling:
    #     producer_rdd = producer_rdd.withResources(cpu_task_profile)
    if cache:
        producer_rdd = producer_rdd.cache()
    elif cache_disk:
        producer_rdd = producer_rdd.persist(StorageLevel.MEMORY_AND_DISK)
        print(producer_rdd.getStorageLevel())

    consumer_rdd = producer_rdd.map(consumer_udf)
    # if stage_level_scheduling:
    #     consumer_rdd = consumer_rdd.withResources(cpu_task_profile)
    if cache:
        consumer_rdd = consumer_rdd.cache()
    elif cache_disk:
        consumer_rdd = consumer_rdd.persist(StorageLevel.MEMORY_AND_DISK)
        print(consumer_rdd.getStorageLevel())

    if stage_level_scheduling:
        # Call repartition to force a new stage for stage level scheduling
        inference_rdd = consumer_rdd.repartition(NUM_VIDEOS)
        inference_rdd = inference_rdd.map(lambda row: (inference_udf(row),)).withResources(
            gpu_task_profile
        )
    else:
        inference_rdd = consumer_rdd.map(lambda row: (inference_udf(row),))

    print("inference_df.count()", inference_rdd.count())

    run_time = time.perf_counter() - start
    print(f"Run time: {run_time:.2f} seconds")


def bench(stage_level_scheduling, cache, cache_disk):
    spark = start_spark(stage_level_scheduling)
    run_spark_data(spark, stage_level_scheduling, cache, cache_disk)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage-level-scheduling",
        action="store_true",
        required=False,
        help="Whether to enable stage level scheduling",
        default=False,
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        required=False,
        help="Whether to cache intermediate datasets in memory",
        default=False,
    )
    parser.add_argument(
        "--cache_disk",
        action="store_true",
        required=False,
        help="Whether to cache intermediate datasets in memory and disk",
        default=False,
    )
    args = parser.parse_args()

    bench(args.stage_level_scheduling, args.cache, args.cache_disk)
