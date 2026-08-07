import time
from pyflink.common.typeinfo import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common import Configuration
from pyflink.datastream.functions import FlatMapFunction, RuntimeContext, MapFunction
import argparse
import resource

from setting import (
    GB,
    EXECUTION_MODE,
    TIME_UNIT,
    NUM_CPUS,
    NUM_GPUS,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    NUM_FRAMES_TOTAL,
    FRAME_SIZE_B,
)


def limit_cpu_memory(mem_limit):
    # limit cpu memory with resources
    mem_limit_bytes = mem_limit * GB
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit_bytes, mem_limit_bytes))


class Producer(FlatMapFunction):
    def open(self, runtime_context: RuntimeContext):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def flat_map(self, value):
        time.sleep(TIME_UNIT * 10)
        for _ in range(FRAMES_PER_VIDEO):
            yield b"1" * FRAME_SIZE_B


class Consumer(MapFunction):
    def open(self, runtime_context):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, value):
        time.sleep(TIME_UNIT)
        return b"2" * FRAME_SIZE_B


class Inference(MapFunction):
    def open(self, runtime_context):
        self.task_info = runtime_context.get_task_name_with_subtasks()
        self.task_index = runtime_context.get_index_of_this_subtask()

    def map(self, value):
        time.sleep(TIME_UNIT)
        return 1


def run_flink(env, mem_limit):
    start = time.perf_counter()
    items = list(range(NUM_VIDEOS))
    ds = env.from_collection(items, type_info=Types.INT())

    producer = Producer()
    ds = ds.flat_map(producer, output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        NUM_CPUS // 2 if mem_limit >= 10 else 3
    )

    ds = ds.map(Consumer(), output_type=Types.PICKLED_BYTE_ARRAY()).set_parallelism(
        NUM_CPUS // 2 if mem_limit >= 10 else 3
    )

    ds = ds.map(Inference(), output_type=Types.LONG()).set_parallelism(
        NUM_GPUS if mem_limit >= 10 else 3
    )

    count = 0
    for length in ds.execute_and_collect():
        count += length
        print(f"Processed {count}/{NUM_FRAMES_TOTAL}")

    end = time.perf_counter()
    print(f"Total rows: {count:,}")
    print(f"Time: {end - start:.4f}s")


def run_experiment(mem_limit):
    config = Configuration()
    config.set_string("python.execution-mode", EXECUTION_MODE)

    # Set memory limit for the task manager
    # https://nightlies.apache.org/flink/flink-docs-master/docs/deployment/memory/mem_setup/
    mem_limit_mb = mem_limit * 1024  # Convert GB to MB
    config.set_string("taskmanager.memory.process.size", f"{mem_limit_mb // 2}m")
    config.set_string("jobmanager.memory.process.size", f"{mem_limit_mb // 2}m")
    # This will oom.
    # limit_cpu_memory(mem_limit)
    env = StreamExecutionEnvironment.get_execution_environment(config)
    run_flink(env, mem_limit)


def main(mem_limit):
    run_experiment(mem_limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=20
    )
    args = parser.parse_args()

    main(args.mem_limit)
