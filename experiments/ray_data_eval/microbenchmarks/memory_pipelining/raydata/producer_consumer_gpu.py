import argparse
import os
import time

import numpy as np
import ray

from ray_data_eval.microbenchmarks.memory_pipelining.setting import (
    GB,
    TIME_UNIT,
    NUM_CPUS,
    NUM_GPUS,
    FRAMES_PER_VIDEO,
    NUM_VIDEOS,
    NUM_FRAMES_TOTAL,
    FRAME_SIZE_B,
)
from ray_data_eval.microbenchmarks.memory_pipelining.raydata import timeline_utils


def bench(mem_limit):
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
    data_context.target_max_block_size = FRAME_SIZE_B
    data_context.is_budget_policy = True
    # data_context.is_conservative_policy = True

    ray.init(address="local",
             num_cpus=NUM_CPUS if mem_limit >= 8 else 6,
             num_gpus=NUM_GPUS,
             object_store_memory=min(12, mem_limit if mem_limit >= 8 else 4) * GB)

    ds = ray.data.range(NUM_FRAMES_TOTAL, override_num_blocks=NUM_VIDEOS)
    ds = ds.map_batches(produce, batch_size=FRAMES_PER_VIDEO)
    ds = ds.map_batches(consume, batch_size=1, num_cpus=0.99)
    ds = ds.map_batches(inference, batch_size=1, num_cpus=0, num_gpus=1)

    start_time = time.time()
    for _ in ds.iter_batches(batch_size=FRAMES_PER_VIDEO):
        pass
    end_time = time.time()
    print(ds.stats())
    print(ray._private.internal_api.memory_summary(stats_only=True))
    print(f"Total time: {end_time - start_time:.4f}s")
    timeline_utils.save_timeline_with_cpus_gpus(
        f"timeline_ray_data_{mem_limit}.json", NUM_CPUS, NUM_GPUS
    )
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mem-limit", type=int, required=False, help="Memory limit in GB", default=20
    )
    args = parser.parse_args()
    bench(args.mem_limit)
