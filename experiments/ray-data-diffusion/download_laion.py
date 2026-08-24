import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

import ray
import requests

# Input schema:
# Dataset(
#    schema={
#       URL: string,
#       TEXT: string,
#       WIDTH: int32,
#       HEIGHT: int32,
#       similarity: double,
#       LANGUAGE: string,
#       hash: int64,
#       pwatermark: float,
#       punsafe: float,
#       aesthetic: float
#    }
# )
# Input sizes:
#  laion-art-8M:
#    num_rows=8070941,
#  aesthetic-120M:
#    num_rows=52068913,


def get_input_ds(args):
    # See https://huggingface.co/laion?search_datasets=laion for other LAION subsets
    dataset_name = args.dataset_name
    if dataset_name == "laion-aesthetic-120M":
        input_uri = []
        # https://huggingface.co/datasets/laion/laion1B-nolang-aesthetic
        input_uri.extend(
            [
                f"https://huggingface.co/datasets/laion/laion1B-nolang-aesthetic/resolve/main/part-{i:05}-604e83c4-a4f2-460a-8aae-1c0fa1d4f6d5-c000.snappy.parquet"
                for i in range(128)
            ]
        )
        # https://huggingface.co/datasets/laion/laion2B-en-aesthetic
        input_uri.extend(
            [
                f"https://huggingface.co/datasets/laion/laion2B-en-aesthetic/resolve/main/part-{i:05}-9230b837-b1e0-4254-8b88-ed2976e9cee9-c000.snappy.parquet"
                for i in range(128)
            ]
        )
        # https://huggingface.co/datasets/laion/laion2B-multi-aesthetic
        input_uri.extend(
            [
                f"https://huggingface.co/datasets/laion/laion2B-multi-aesthetic/resolve/main/part-{i:05}-41ee6475-31c6-4d39-960e-7dbbe96bc95b-c000.snappy.parquet"
                for i in range(128)
            ]
        )
        input_parallelism = 120_000_000 // args.num_images_per_task
    elif dataset_name == "laion-art-8M":
        input_uri = "https://huggingface.co/datasets/laion/laion-art/resolve/main/laion-art.parquet"
        input_parallelism = 8_000_000 // args.num_images_per_task
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print("Input parallelism", input_parallelism)
    ds = ray.data.read_parquet(input_uri, parallelism=input_parallelism)
    return ds


MAX_DOWNLOAD_RETRIES = 1
RETRY_DELAY = 0.1
DOWNLOAD_THREADS_PER_WORKER = 32
DOWNLOAD_TIMEOUT = 3.0


def download_images(batch: Dict[str, Any]) -> Dict[str, Any]:
    def download_image(url):
        jpg = b""
        error_msg = ""
        max_retries = MAX_DOWNLOAD_RETRIES
        while max_retries > 0:
            try:
                resp = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
                resp.raise_for_status()
                jpg = resp.content
                error_msg = ""
                break
            except Exception as e:
                max_retries -= 1
                if max_retries < 0:
                    jpg = b""
                    error_msg = str(e)
                    break
                time.sleep(RETRY_DELAY)
        return jpg, error_msg

    start = time.time()
    # Rename these columns
    col_mapping = [
        ("URL", "url"),
        ("TEXT", "caption"),
        ("HEIGHT", "height"),
        ("WIDTH", "width"),
    ]
    for c1, c2 in col_mapping:
        batch[c2] = batch[c1]
        del batch[c1]

    # Using ThreadPoolExecutor to download images in parallel
    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS_PER_WORKER) as executor:
        # List of futures of the binary content of images
        futures = executor.map(download_image, batch["url"])

    # Wait for all futures to complete
    download_res = list(futures)

    batch["jpg"] = [r[0] for r in download_res]
    batch["error_msg"] = [r[1] for r in download_res]

    num_downloaded = sum((jpg != b"") for jpg in batch["jpg"])
    print(
        f"Task finished in {time.time() - start:.2f} seconds. {num_downloaded}/{len(batch['url'])} images downloaded."
    )

    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        default="laion-aesthetic-120M",
        type=str,
        help="Name of the input dataset.",
    )
    parser.add_argument(
        "--dry_run",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--num_images_per_task",
        default=2000,
        type=int,
    )
    parser.add_argument(
        "--output_uri",
        required=True,
        type=str,
        help="The output S3 bucket URI for the dataset.",
    )
    # NOTE, as of Ray 2.10, write_parquet needs to accumulate all rows before writing.
    # Setting this argument to a too large value may hurt perf and stability.
    parser.add_argument(
        "--num_rows_per_output_file",
        default=10_000,
        type=int,
        help="Number of rows per output file.",
    )

    args = parser.parse_args()

    env_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ]

    if all(env_var in os.environ for env_var in env_vars):
        print("Using AWS credentials from environment")
        ray.init(runtime_env={"env_vars": {k: os.environ[k] for k in env_vars}})

    data_context = ray.data.DataContext.get_current()
    # Continue execution even if some tasks fail.
    data_context.max_errored_blocks = 1_000_000_000

    ds = get_input_ds(args)
    ds = ds.materialize()
    print("Input dataset", ds)

    ds = ds.map_batches(download_images, batch_size=None)

    print(f"Writing to {args.output_uri}.")
    start = time.time()
    if not args.dry_run:
        ds.write_parquet(
            args.output_uri,
            try_create_dir=True,
            num_rows_per_file=args.num_rows_per_output_file,
        )
    else:
        print("Dry run, not writing to S3.")

        ds.show(2)

        def drop_large_columns(batch):
            del batch["jpg"]
            return batch

        ds = ds.map_batches(drop_large_columns, batch_size=None)
        num_rows = 0
        for batch in ds.iter_batches():
            num_rows += len(batch["url"])
        print(f"Number of rows: {num_rows}")

    print(f"Finished in {time.time() - start:.2f} seconds.")
    with open("./download_laion.log", "w") as f:
        print(f"Finished in {time.time() - start:.2f} seconds.", file=f)
        print(ds.stats(), file=f)
