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
_PROCESSOR = None  # per-worker cache, see preprocess_video
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
    # Build the processor once per worker. Constructing it per row costs real
    # CPU, and if it ever resolves to the Hub rather than a local cache it draws
    # HTTP 429s that look exactly like a throughput collapse.
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = VideoMAEImageProcessor.from_pretrained(
            PREPROCESSOR_CACHE, local_files_only=True)
    processor = _PROCESSOR
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
    # No-op on stock DataContext; kept to match the original runs.
    data_context.is_budget_policy = True
    # Per-operator memory reservation. At 0 there is no backpressure and decode
    # buffers without bound, which stalls recovery in Figure 7c.
    data_context.op_resource_reservation_ratio = float(
        os.environ.get("RAY_DATA_RESV_RATIO", "0.5"))
    data_context.execution_options.verbose_progress = True
    # Prefer running a task where its output will be consumed. Fewer decoded
    # blocks then sit on a remote node, so less is lost if that node fails.
    if os.environ.get("RAY_DATA_LOCALITY_WITH_OUTPUT"):
        data_context.execution_options.locality_with_output = (
            os.environ["RAY_DATA_LOCALITY_WITH_OUTPUT"] == "1")

    # RAY_DATA_CTX_<setting> selects the system being emulated. With none set the
    # DataContext keeps its own defaults, which is Ray Data.
    _overrides = {k[len("RAY_DATA_CTX_"):].lower(): v for k, v in os.environ.items()
                  if k.startswith("RAY_DATA_CTX_")}
    for _attr, _v in _overrides.items():
        if not hasattr(data_context, _attr):
            raise RuntimeError(
                f"DataContext has no attribute {_attr!r}; Ray Data is probably "
                f"not installed (see scripts/setup/install_ray_data.sh)")
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

    # Section 5.1.2 variants. staged: each stage is materialized before the
    # next starts. static: a fixed number of replicas per operator, partitions
    # handed out in order, instead of the dynamic scheduler.
    staged = bool(os.environ.get("RAY_DATA_STAGED"))
    static = bool(os.environ.get("RAY_DATA_STATIC"))
    preprocess_fn, preprocess_args = preprocess_video, {}
    if static:
        class PreprocessReplica:
            def __call__(self, row):
                return preprocess_video(row)

        preprocess_fn = PreprocessReplica
        # Half the CPUs to read tasks, half to preprocess replicas.
        preprocess_args = dict(
            compute=ray.data.ActorPoolStrategy(
                size=max(1, NUM_CPUS_IN_CLUSTER // 2), max_tasks_in_flight_per_actor=1),
        )

    # Restart from a checkpoint (Figure 7c, dashed curves): skip the inputs
    # already processed at the checkpoint, in the order Ray lists them.
    input_path = INPUT_PATH
    skip = int(os.environ.get("RAY_DATA_SKIP_FILES", "0"))
    if skip:
        import pyarrow.fs as pafs

        fs, root = pafs.FileSystem.from_uri(INPUT_PATH)
        infos = fs.get_file_info(pafs.FileSelector(root, recursive=True))
        paths = sorted(i.path for i in infos if i.type == pafs.FileType.File)
        input_path = ["s3://" + q for q in paths[skip:]]
        print(f"[ray_data] skipping the first {skip} of {len(paths)} inputs", flush=True)

    read_blocks = int(os.environ.get("RAY_DATA_READ_BLOCKS", "0"))
    if static and not read_blocks:
        read_blocks = NUM_CPUS_IN_CLUSTER * 4
    # Retry the read tasks too: on Ray 2.40 the broken pipe from a dead node
    # surfaces from ReadBinary, and unretried it aborts the dataset.
    ds = ray.data.read_binary_files(
        input_path,
        ray_remote_args={"retry_exceptions": [OSError], "max_retries": 3},
        **({"override_num_blocks": read_blocks} if read_blocks else {}),
    )
    if staged:
        ds = ds.materialize()

    # A dead node raises OSError(Broken pipe) inside the task; Ray treats that as
    # an application error and will not retry it by default. Needed for Figure 7c.
    ds = ds.map(
        preprocess_fn,
        **({} if static else {"retry_exceptions": [OSError], "max_retries": 3}),
        **preprocess_args,
    )
    if staged:
        ds = ds.materialize()

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
