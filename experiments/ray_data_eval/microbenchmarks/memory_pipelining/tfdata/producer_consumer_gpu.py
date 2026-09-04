import tensorflow as tf
import time
import os
import numpy as np
import argparse
from setting import (
    GB,
    TIME_UNIT,
    NUM_CPUS,
    NUM_GPUS,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    NUM_FRAMES_TOTAL,
    FRAME_SIZE_B,
    limit_cpu_memory,
)
import sys

TF_PROFILER_LOGS = "logs/tf"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
parent_directory = os.path.abspath("..")
sys.path.append(parent_directory)


def bench(mem_limit):
    # oom at 4GB
    # limit_cpu_memory(mem_limit)

    options = tf.data.Options()
    # https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutotuneOptions
    options.autotune.enabled = True
    options.autotune.cpu_budget = NUM_CPUS + NUM_GPUS
    # When autotuning is enabled (through autotune), determines the RAM budget to use. Values greater than the available RAM in bytes may result in OOM. If None, defaults to half of the available RAM in bytes.
    # Doesn't work at the moment.
    options.autotune.ram_budget = mem_limit * GB

    def producer_fn(idx):
        time.sleep(10 * TIME_UNIT)
        for i in range(FRAMES_PER_VIDEO):
            data = {
                "idx": idx * FRAMES_PER_VIDEO + i,
                "data": np.full(FRAME_SIZE_B, i, dtype=np.uint8),
            }
            yield data

    def consumer_fn(idx, data):
        time.sleep(TIME_UNIT)
        return np.zeros(FRAME_SIZE_B, dtype=np.uint8)

    def inference_fn(data):
        time.sleep(TIME_UNIT)
        return 1

    start = time.perf_counter()
    items = list(range(NUM_VIDEOS))
    ds = tf.data.Dataset.from_tensor_slices(items)

    if mem_limit <= 12:
        p = 1
    elif mem_limit <= 14:
        p = 2
    else:
        p = tf.data.experimental.AUTOTUNE

    # flat_map doesn't have num_parallel_calls
    ds = ds.with_options(options).interleave(
        lambda item: tf.data.Dataset.from_generator(
            producer_fn,
            args=(item,),
            output_signature={
                "idx": tf.TensorSpec(shape=(), dtype=tf.int64),
                "data": tf.TensorSpec(shape=(FRAME_SIZE_B,), dtype=tf.uint8),
            },
            name="producer",
        ),
        block_length=1,
        num_parallel_calls=p,
        name="producer_interleave",
    )

    ds = ds.with_options(options).map(
        lambda items: tf.numpy_function(
            consumer_fn,
            inp=[items["idx"], items["data"]],
            Tout=tf.uint8,
            name="consumer",
        ),
        num_parallel_calls=p,
        name="consumer_map",
    )

    ds = ds.with_options(options).map(
        lambda items: tf.numpy_function(
            inference_fn,
            inp=[items],
            Tout=tf.int64,
            name="inference",
        ),
        # GPU stage.
        num_parallel_calls=4 if mem_limit > 10 else 1,
        name="inference_map",
    )

    ret = 0
    for row in ds:
        ret += row.numpy()
        print(f"ret: {ret}/{NUM_FRAMES_TOTAL}")
    run_time = time.perf_counter() - start
    print(f"Sum: {ret:,}")
    print(f"Run time: {run_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=20
    )
    args = parser.parse_args()
    if not os.path.exists(TF_PROFILER_LOGS):
        os.makedirs(TF_PROFILER_LOGS)

    # import multiprocessing
    # Start memory usage logging in a separate process
    # logging_process = multiprocessing.Process(target=log_memory_usage_process, args=(2, args.mem_limit))  # Log every 2 seconds
    # logging_process.start()
    limit_cpu_memory(args.mem_limit)

    bench(args.mem_limit)
    # logging_process.terminate()
