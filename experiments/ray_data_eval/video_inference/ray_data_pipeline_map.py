import argparse
import functools
import io
import os
import sys
import time

import numpy as np
import ray
from ray.data.block import DataBatch

import humanize
from ray_data_pipeline_helpers import postprocess
import timeline_utils
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s",
    "--source",
    default="s3",
    help="local or S3",
)

args = parser.parse_args()

DEVICE = "cuda"
MODEL_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
IMAGE_SIZE = 224
NUM_FRAMES = 16
MODEL_INPUT_SHAPE = (NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)
BATCH_SIZE = 32

PREPROCESSOR_CACHE = "/tmp/videomae_processor"
MODEL_CACHE = "/tmp/videomae_model"

if args.source == "local":
    print("Using local data.")
    train_data_path = "/home/ubuntu/kinetics/kinetics/k700-2020/train"
    labels = [os.path.join(train_data_path, label) for label in sorted(os.listdir(train_data_path))]
    INPUT_PATH = labels
else:
    print("Using S3 data.")
    INPUT_PATH = "s3://ray-data-eval-us-west-2/kinetics/k700-2020/test"


def timeit(name=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print(f"{name or func.__name__}: {time.time() - start:.2f} seconds")
            return result

        return wrapper

    if callable(name):
        return decorator(name)
    else:
        return decorator


def tensor_size(t: torch.Tensor) -> str:
    return humanize.naturalsize(t.element_size() * t.nelement())


def print_gpu_memory_usage():
    print(
        f"Total GPU memory: {humanize.naturalsize(torch.cuda.get_device_properties(0).total_memory)}"
    )
    print(f"Reserved GPU memory: {humanize.naturalsize(torch.cuda.memory_reserved(0))}")
    print(f"Allocated GPU memory: {humanize.naturalsize(torch.cuda.memory_allocated(0))}")


class Classifier:
    def __init__(self):
        start_time = time.time()
        self.model = (
            VideoMAEForVideoClassification.from_pretrained(MODEL_CACHE, local_files_only=True)
            .eval()
            .to(DEVICE)
        )
        print(f"Time to initialize model: {time.time() - start_time}")

    @timeit("Inference")
    @torch.no_grad
    def __call__(self, batch: DataBatch) -> DataBatch:
        inference_start_time = time.time()
        batch = collate_video_frames(batch)
        model_input = torch.from_numpy(batch["video"]).to(DEVICE)
        print(f"Input tensor size: {tensor_size(model_input)}")
        model_output = self.model(model_input)
        logits = model_output.logits
        preds = logits.argmax(-1)
        result = [self.model.config.id2label[pred.item()] for pred in preds]
        print_gpu_memory_usage()

        inference_end_time = time.time()
        print(
            "[Completed Batch]",
            inference_end_time,
            len(batch["video"]),
            "[Inference Tput]",
            len(batch["video"]) / (inference_end_time - inference_start_time),
        )
        return {"result": result}


def preprocess_video(row: DataBatch) -> DataBatch:
    from decord import VideoReader, DECORDError

    video_bytes = row["bytes"]
    try:
        vr = VideoReader(
            io.BytesIO(video_bytes),
            num_threads=1,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
        )
        frames = vr.get_batch(range(min(NUM_FRAMES, len(vr)))).asnumpy()
        if frames.shape[0] < NUM_FRAMES:
            last_frame = frames[-1:]
            last_frame_repeated = np.repeat(last_frame, NUM_FRAMES - len(frames), axis=0)
            frames = np.concatenate([frames, last_frame_repeated], axis=0)
    except DECORDError as e:
        print(f"Failed to process video: {e}")
        return {"video": np.zeros((1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)}

    frames = list(frames)
    processor = VideoMAEImageProcessor.from_pretrained(PREPROCESSOR_CACHE, local_files_only=True)
    ret = processor(frames, return_tensors="np")
    arr = ret.data["pixel_values"]

    return {"video": arr}


def collate_video_frames(batch: DataBatch) -> DataBatch:
    try:
        return {"video": np.concatenate(batch["video"], axis=0)}
    except ValueError as e:
        print(f"Failed to collate video frames: {e}", flush=True)
        for i in range(len(batch["video"])):
            if batch["video"][i].shape[0] != NUM_FRAMES:
                last_frame = batch["video"][i][-1:]
                last_frame_repeated = np.repeat(
                    last_frame, NUM_FRAMES - batch["video"][i].shape[0], axis=0
                )
                batch["video"][i] = np.concatenate([batch["video"][i], last_frame_repeated], axis=0)
        return {"video": np.concatenate(batch["video"], axis=0)}


@timeit
def main():
    ray.init("auto")

    print("Starting warmup.", flush=True)
    # cluster_resources(), not available_resources(): the latter reports
    # what is currently free, which is 0 if another job holds the cluster.
    NUM_CPUS_IN_CLUSTER = int(ray.cluster_resources()["CPU"])
    NUM_GPUS_IN_CLUSTER = int(ray.cluster_resources().get("GPU", 0))
    if NUM_GPUS_IN_CLUSTER == 0:
        raise RuntimeError("no GPUs in the Ray cluster; this benchmark needs at least one")

    data_context = ray.data.DataContext.get_current()
    data_context.is_budget_policy = True
    data_context.op_resource_reservation_ratio = 0
    data_context.execution_options.verbose_progress = True

    # RAY_DATA_CTX_<knob> environment variables select which system is being
    # emulated. With none set, behaviour is the committed default below.
    # See ../../../../patches/README.md.
    # With none set, the DataContext keeps its own defaults, which is Ray Data.
    # The file as committed hardcoded the microbatch baseline here instead.
    _overrides = {k[len("RAY_DATA_CTX_"):].lower(): v for k, v in os.environ.items()
                  if k.startswith("RAY_DATA_CTX_")}
    for _attr, _v in _overrides.items():
        if not hasattr(data_context, _attr):
            raise RuntimeError(
                f"DataContext has no attribute {_attr!r}; the Ray Data overlay is "
                f"probably not installed (see scripts/install_ray_data.sh)")
        if isinstance(_v, str):
            _v = {"True": True, "False": False}.get(_v, _v)
            if isinstance(_v, str) and _v.isdigit():
                _v = int(_v)
        setattr(data_context, _attr, _v)
        print(f"[ray_data] DataContext.{_attr} = {_v!r}", flush=True)
    print(f"[ray_data] effective scheduling_policy = "
          f"{data_context.scheduling_policy!r}", flush=True)


    print("NUM_GPUS_IN_CLUSTER", NUM_GPUS_IN_CLUSTER, "NUM_CPUS_IN_CLUSTER: ", NUM_CPUS_IN_CLUSTER)
    ds = (
        ray.data.read_binary_files(
            [
                "s3://ray-data-eval-us-west-2/kinetics/k700-2020/train/abseiling/__NrybzYzUg_000415_000425.mp4"
            ]
            * NUM_CPUS_IN_CLUSTER
            * 2
        )
        .map(preprocess_video)
        .map_batches(
            Classifier,
            batch_size=BATCH_SIZE,
            num_gpus=1,
            concurrency=NUM_GPUS_IN_CLUSTER,
            zero_copy_batch=True,
            max_concurrency=NUM_GPUS_IN_CLUSTER * 2,
        )
    )
    ds.take_all()

    print("Finished warmup. Starting the pipeline.", flush=True)
    time.sleep(10)

    INSTANCE = "g5_xlarge"
    TIMELINE_FILENAME = f"video_inference_{args.source}_{INSTANCE}_batch_{BATCH_SIZE}.json"
    OUTPUT_FILENAME = f"video_inference_{args.source}_{INSTANCE}_batch_{BATCH_SIZE}.out"

    start_time = time.time()
    print("[Start Time]", start_time, flush=True)

    ds = ray.data.read_binary_files(
        INPUT_PATH,
    )

    # RAY_DATA_LIMIT caps the number of videos, so a reviewer can run a fraction
    # of the split. Unset means the whole thing, as before.
    _limit = os.environ.get("RAY_DATA_LIMIT")
    if _limit:
        ds = ds.limit(int(_limit))
        print(f"[ray_data] RAY_DATA_LIMIT={_limit}: reading {_limit} videos", flush=True)

    ds = ds.map(
        preprocess_video,
    )

    ds = ds.map_batches(
        Classifier,
        batch_size=BATCH_SIZE,
        num_gpus=1,
        concurrency=NUM_GPUS_IN_CLUSTER,
        zero_copy_batch=True,
        max_concurrency=NUM_GPUS_IN_CLUSTER * 2,
    )

    try:
        ds.take_all()
        print(ds.stats())
    except Exception as e:
        print(f"Pipeline execution failed: {type(e).__name__}: {e}", flush=True)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            ray.timeline(
                f"video_inference_{args.source}_{INSTANCE}_batch_{BATCH_SIZE}_original.json"
            )
            timeline_utils.save_timeline_with_cpus_gpus(
                TIMELINE_FILENAME, NUM_CPUS_IN_CLUSTER, NUM_GPUS_IN_CLUSTER
            )
        except Exception as e:
            print(f"Failed to save timeline: {e}", flush=True)

        sys.stdout.flush()
        postprocess(OUTPUT_FILENAME)
        ray.shutdown()


if __name__ == "__main__":
    main()
