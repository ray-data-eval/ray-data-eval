import time
from argparse import ArgumentParser

from ray_diffusion.data.datasets import get_laion_streaming_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sdxl", action="store_true", help="Whether to preprocess data for SDXL."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only process data, not upload results to S3.",
        default=False,
    )
    parser.add_argument(
        "--num_gpu", type=int, default=16, help="Number of GPUs for preprocessing"
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="Image resolution, 256 or 512."
    )
    parser.add_argument(
        "--raw_data_uri",
        type=str,
        default="s3://yunxuanx-test/data/laion-raw-debug/val/",
    )
    parser.add_argument("--output_uri", type=str, default="s3://yunxuanx-test/tmp/tmp")
    # NOTE, as of Ray 2.10, write_parquet needs to accumulate all rows before writing.
    # Setting this argument to a too large value may hurt perf and stability.
    parser.add_argument(
        "--num_rows_per_output_file",
        default=5_000,
        type=int,
        help="Number of rows per output file.",
    )
    args = parser.parse_args()

    start_t = time.time()
    ds = get_laion_streaming_dataset(
        args.raw_data_uri,
        num_encoders=args.num_gpu,
        sdxl=args.sdxl,
        resolution=args.resolution,
        shuffle=None,
    )

    if not args.dry_run:
        ds.write_parquet(
            args.output_uri,
            num_rows_per_file=args.num_rows_per_output_file,
        )
        # No API to get the actual number of written rows.
        # See `ds.stats()`.
        num_rows = 0
    else:

        def drop_large_columns(batch):
            return {"url": batch["url"]}

        ds = ds.map_batches(drop_large_columns, batch_size=1000)
        num_rows = 0
        for batch in ds.iter_batches(batch_size=None):
            num_rows += len(batch["url"])

    total_t = time.time() - start_t

    print(f"{num_rows} finished in {total_t} seconds.")
    print(f"Throughput = {num_rows / total_t} images/s")
    print(ds.stats())
