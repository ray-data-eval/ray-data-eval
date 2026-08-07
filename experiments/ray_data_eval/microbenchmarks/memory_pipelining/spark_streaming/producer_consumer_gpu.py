import time
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.listener import StreamingListener
from pyspark.resource.requests import TaskResourceRequests
from pyspark.resource import ResourceProfileBuilder
import resource
import argparse
from setting import TIME_UNIT, FRAMES_PER_VIDEO, NUM_VIDEOS, NUM_CPUS, FRAME_SIZE_B, GB


def limit_cpu_memory(mem_limit):
    # limit cpu memory with resources
    mem_limit_bytes = mem_limit * GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))


class CustomStreamingListener(StreamingListener):
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def onBatchStarted(self, batchStarted):
        if self.start_time is None:
            self.start_time = batchStarted.batchInfo().submissionTime() / 1000.0
            print(f"Start time: {self.start_time:.2f}")

    def onBatchCompleted(self, batchCompleted):
        self.end_time = batchCompleted.batchInfo().submissionTime() / 1000.0
        print(
            f"\nTotal runtime of streaming computation: {self.end_time - self.start_time:.2f} seconds"
        )


def start_spark_streaming(executor_memory, stage_level_scheduling):
    # https://spark.apache.org/docs/latest/configuration.html

    if executor_memory < 10:
        # Using rlimit will oom.
        # Instead, I just set executor.memory.
        # The total memory should be 1 * 2g + 2g = 4g.
        # limit_cpu_memory(executor_memory)
        assert executor_memory == 4
        conf = (
            SparkConf()
            # .set("spark.dynamicAllocation.enabled", "false")
            .set("spark.cores.max", 4)
            .set("spark.executor.cores", 4)
            .set("spark.executor.instances", 1)
            .set("spark.executor.memory", "1g")
            .set("spark.driver.memory", "2g")
            .set("spark.scheduler.mode", "FAIR")
            # .set("spark.executor.resource.gpu.amount", 1)
            # .set("spark.task.resource.gpu.amount", 1)
        )
    elif not stage_level_scheduling:
        conf = (
            SparkConf()
            .set("spark.dynamicAllocation.enabled", "false")
            .set("spark.cores.max", NUM_CPUS)
            .set("spark.executor.cores", 1)
            .set("spark.executor.instances", NUM_CPUS)
            .set("spark.executor.memory", "5g")
            .set("spark.driver.memory", "8g")
            # .set("spark.scheduler.mode", "FAIR")
            # .set("spark.executor.resource.gpu.amount", 1)
        )
    else:
        conf = (
            SparkConf()
            .set("spark.dynamicAllocation.enabled", "false")
            .set("spark.cores.max", NUM_CPUS)
            .set("spark.executor.cores", 2)
            .set("spark.executor.instances", 4)
            .set("spark.executor.memory", "5g")
            .set("spark.driver.memory", "8g")
            .set("spark.executor.resource.gpu.amount", 1)
        )

    BATCH_INTERVAL = 0.1  # seconds
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, BATCH_INTERVAL)
    return sc, ssc


def producer(row):
    print('producer')
    time.sleep(TIME_UNIT * 10)
    for j in range(FRAMES_PER_VIDEO):
        data = b"1" * FRAME_SIZE_B
        yield (data, row * FRAMES_PER_VIDEO + j)


def consumer(row):
    print('consumer')
    time.sleep(TIME_UNIT)
    data = b"2" * FRAME_SIZE_B
    return (data,)


def inference(row):
    print('inference')
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
            print(f"\nTotal runtime of streaming computation: {run_time:.2f} seconds")
            ssc.stop(stopSparkContext=True, stopGraceFully=False)
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

    assert not args.stage_level_scheduling, "Receive error: TaskResourceProfiles are only supported for Standalone, Yarn and Kubernetes cluster for now when dynamic allocation is disabled."

    bench(args.mem_limit, args.stage_level_scheduling)
