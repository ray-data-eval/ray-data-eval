import argparse
import os
import time

import numpy as np
import ray

from ray_data_eval.microbenchmarks.memory_pipelining.setting import (
    GB,
    log_memory_usage_process,
    TIME_UNIT,
    NUM_CPUS,
    NUM_GPUS,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    NUM_FRAMES_TOTAL,
    FRAME_SIZE_B,
)
from ray_data_eval.microbenchmarks.memory_pipelining.raydata import timeline_utils


VARIANTS = ("default", "no_adapt", "no_part")


def cluster_config(variant, mem_limit):
    # (num_cpus, num_gpus, object store GB) of the published runs.
    if variant == "default":
        return (NUM_CPUS if mem_limit >= 8 else 6,
                NUM_GPUS,
                min(12, mem_limit if mem_limit >= 8 else 4))
    if variant == "no_adapt":
        return (NUM_CPUS if mem_limit >= 8 else 4,
                NUM_GPUS if mem_limit >= 8 else 4,
                min(12, mem_limit if mem_limit >= 8 else 2.5))
    if mem_limit <= 8:
        store = 2
    elif mem_limit <= 10:
        store = 2.5
    elif mem_limit <= 12:
        store = 4
    else:
        store = min(12, mem_limit)
    return (NUM_CPUS if mem_limit > 10 else (4 if mem_limit > 8 else 2), NUM_GPUS, store)


def bench(mem_limit, variant):
    os.environ["RAY_DATA_OP_RESERVATION_RATIO"] = "0"

    def produce(batch):
        time.sleep(TIME_UNIT * 10)
        for id in batch["id"]:
            yield {
                "id": [id],
                "image": [np.zeros(FRAME_SIZE_B, dtype=np.uint8)],
            }

    def consume(batch):
        time.sleep(TIME_UNIT)
        return {"id": batch["id"], "image": [np.ones(FRAME_SIZE_B, dtype=np.uint8)]}

    def inference(batch):
        time.sleep(TIME_UNIT)
        return {"id": batch["id"]}

    data_context = ray.data.DataContext.get_current()
    data_context.execution_options.verbose_progress = True
    data_context.target_max_block_size = FRAME_SIZE_B * (10 if variant == "no_part" else 1)
    data_context.is_budget_policy = variant != "no_adapt"

    num_cpus, num_gpus, store_gb = cluster_config(variant, mem_limit)
    ray.init(address="local", num_cpus=num_cpus, num_gpus=num_gpus,
             object_store_memory=store_gb * GB)

    fixed = {"concurrency": 4} if variant == "no_adapt" else {}
    ds = ray.data.range(NUM_FRAMES_TOTAL, override_num_blocks=NUM_VIDEOS)
    ds = ds.map_batches(produce, batch_size=FRAMES_PER_VIDEO, **fixed)
    ds = ds.map_batches(consume, batch_size=1, num_cpus=0.99, **fixed)
    ds = ds.map_batches(inference, batch_size=1, num_cpus=0, num_gpus=1)

    start_time = time.time()
    for _ in ds.iter_batches(batch_size=FRAMES_PER_VIDEO):
        pass
    end_time = time.time()
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    print(f"Total time: {end_time - start_time:.4f}s")
    timeline_utils.save_timeline_with_cpus_gpus(
        f"timeline_ray_data_{variant}_{mem_limit}.json", NUM_CPUS, NUM_GPUS
    )
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=20
    )
    parser.add_argument(
        "--variant", choices=VARIANTS, default="default",
        help="Ray Data row of Figure 9: default, no_adapt (-Adapt.), no_part (-Part.)",
    )
    args = parser.parse_args()

    import multiprocessing

    logger = multiprocessing.Process(
        target=log_memory_usage_process, args=(2, args.mem_limit)
    )
    logger.start()
    try:
        bench(args.mem_limit, args.variant)
    finally:
        logger.terminate()
