import os
import time
from typing import Any

import ray

ROW_SIZE = 1000 * 1000
# Overridable so a reviewer can run a shorter sweep; the defaults are the
# published configuration. See ../../../patches/README.md.
NUM_ROWS = int(os.environ.get("PARTITION_NUM_ROWS", 8192))
TIME_PER_ROW = 0.01


def busy(duration: float):
    start_time = time.time()
    while time.time() - start_time < duration:
        pass


def process(batch: dict[str, Any]):
    num_rows = len(batch["data"])
    # if num_rows >= 100:
    #     print(f"Process {num_rows} rows")
    busy(num_rows * TIME_PER_ROW)
    return batch


def drop(batch: dict[str, Any]):
    num_rows = len(batch["data"])
    # if num_rows >= 100:
    #     print(f"Drop {num_rows} rows")
    busy(num_rows * TIME_PER_ROW)
    return {"data": [0]}


def bench(num_rows_in_block: int):
    start_time = time.time()

    ds = ray.data.range_tensor(
        NUM_ROWS,
        shape=(1000, 125),  # np.uint64 is 8 bytes
        override_num_blocks=NUM_ROWS // num_rows_in_block,
    )
    ds = ds.map_batches(
        process,
        zero_copy_batch=True,
        concurrency=8,
    )
    ds = ds.map_batches(
        drop,
        num_cpus=0.99,
        zero_copy_batch=True,
        concurrency=8,
    )
    ds.take_all()

    ray.timeline(f"timeline_{num_rows_in_block}.json")

    duration = time.time() - start_time
    print(f"{num_rows_in_block=}, {duration=}")
    return duration


DEFAULT_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def main():
    sizes = [int(x) for x in
             os.environ.get("PARTITION_SIZES",
                            ",".join(map(str, DEFAULT_SIZES))).split(",")]
    bench(1000)
    print("Warmup done")
    for n in sizes:
        bench(n)


if __name__ == "__main__":
    # address="local": this benchmark fixes its own object store size, so it
    # must not attach to a cluster. See ../../../patches/README.md.
    ray.init(address="local",
             object_store_memory=float(os.environ.get("PARTITION_STORE_GB", 15)) * 1e9)

    data_context = ray.data.DataContext.get_current()
    # Per-operator memory reservation stays at Ray's default (0.5); at 0 a stage
    # buffers ahead without bound.
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = 1024 * 1024 * 1024
    data_context.is_budget_policy = False
    main()
