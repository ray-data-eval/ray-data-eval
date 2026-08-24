import io
import math
from functools import partial
from typing import Any, Dict, List, Union

import numpy as np
import pyarrow as pa
import ray
import ray.air.util.tensor_extensions.arrow
from PIL import Image
from ray.util.accelerators import NVIDIA_TESLA_A10G

from ray_diffusion.data.encoders import SDLatentEncoder, SDXLLatentEncoder
from ray_diffusion.data.preprocessors import SDPreprocessor, SDXLPreprocessor
from ray_diffusion.utils.constants import *


def transform_and_tokenize(
    row: Dict[str, Any], preprocessor: Union[SDPreprocessor, SDXLPreprocessor]
) -> List[Dict[str, Any]]:
    """Transform the images and filter out the invalid ones."""
    try:
        if row["jpg"] == b"":
            return []

        if math.isnan(row["width"]):
            row["width"] = 0.0
        if math.isnan(row["height"]):
            row["height"] = 0.0

        image = Image.open(io.BytesIO(row["jpg"]))
        if image.mode != "RGB":
            image = image.convert("RGB")

        caption = row["caption"]

        row.update(preprocessor.transform(image))
        row.update(preprocessor.tokenize(caption))
    except Exception as e:
        return []

    for k, v in row.items():
        if v is None:
            print("Dropping bad row with None value", k)
            return []

    return [row]


def _get_training_columns(sdxl, resolution):
    if sdxl:
        key_list = [
            SDXL_CAPTION_LATENTS_1_KEY,
            SDXL_CAPTION_LATENTS_2_KEY,
            SDXL_ADD_TEXT_EMBEDS_KEY,
            CAPTION_KEY,
        ]
        if resolution == 256:
            key_list.append(SDXL_IMAGE_LATENTS_256_KEY)
        elif resolution == 512:
            key_list.append(SDXL_IMAGE_LATENTS_512_KEY)
    else:
        key_list = [CAPTION_LATENTS_KEY, CAPTION_KEY]
        if resolution == 256:
            key_list.append(IMAGE_LATENTS_256_KEY)
        elif resolution == 512:
            key_list.append(IMAGE_LATENTS_512_KEY)
    return key_list


def get_laion_streaming_dataset(
    input_uri,
    num_encoders,
    sdxl=False,
    resolution=256,
    shuffle=None,
    training_batch_size=None,
):
    """Get Ray Data stream with online preprocessing."""
    data_context = ray.data.DataContext.get_current()
    # Continue execution even if some tasks fail.
    data_context.max_errored_blocks = 1_000_000_000
    # Concurrency limits the number of concurrent tasks.
    # It's optional. But setting this will slightly improve performance.
    concurrency = 4 * num_encoders
    # Input files have inconsistent schemas,
    # The `width` and `height` columns have int or float types depending on the file.
    # So we need to specify the schema manually.
    schema = pa.schema(
        [
            pa.field("similarity", pa.float64()),
            pa.field("hash", pa.int64()),
            pa.field("punsafe", pa.float32()),
            pa.field("pwatermark", pa.float32()),
            pa.field("aesthetic", pa.float32()),
            pa.field("url", pa.string()),
            pa.field("caption", pa.string()),
            pa.field("height", pa.float64()),
            pa.field("width", pa.float64()),
            pa.field("jpg", pa.binary()),
            pa.field("error_msg", pa.string()),
        ]
    )

    ds = ray.data.read_parquet(
        input_uri,
        schema=schema,
        shuffle=shuffle,
        concurrency=concurrency,
    )
    print("Input dataset:", ds)
    print("Number of input files:", len(ds.input_files()))

    # 1. Clean and transform images
    preprocessor = (
        SDXLPreprocessor(resolution) if sdxl else SDPreprocessor(resolution=resolution)
    )
    transform_func = partial(transform_and_tokenize, preprocessor=preprocessor)

    ds = ds.flat_map(
        transform_func,
        concurrency=concurrency,
    )

    # 2 Encode images and captions for SDXL
    if sdxl:
        if resolution == 256:
            batch_size = 120
        else:
            assert resolution == 512
            batch_size = 40
        ds = ds.map_batches(
            SDXLLatentEncoder,
            fn_constructor_kwargs={"resolution": resolution},
            compute=ray.data.ActorPoolStrategy(size=num_encoders),
            num_gpus=1,  # Specify 1 GPU per model replica.
            batch_size=batch_size,  # Use the largest batch size that can fit on our GPUs
            max_concurrency=2,
            accelerator_type=NVIDIA_TESLA_A10G,
        )
    else:
        if resolution == 256:
            batch_size = 160
        else:
            assert resolution == 512
            batch_size = 40
        ds = ds.map_batches(
            SDLatentEncoder,
            fn_constructor_kwargs={"resolution": resolution},
            compute=ray.data.ActorPoolStrategy(size=num_encoders),
            num_gpus=1,  # Specify 1 GPU per model replica.
            batch_size=batch_size,  # Use the largest batch size that can fit on our GPUs
            max_concurrency=2,
            accelerator_type=NVIDIA_TESLA_A10G,
        )

    if training_batch_size is not None:
        cols = set(_get_training_columns(sdxl, resolution))

        def select_training_columns(batch):
            return {k: v for k, v in batch.items() if k in cols}

        ds = ds.map_batches(
            select_training_columns,
            zero_copy_batch=True,
            batch_size=training_batch_size,
        )

    return ds


def load_precomputed_dataset(
    train_data_uri, num_workers, sdxl=False, resolution=256, shuffle=None
):
    """Load from offline precomputed datasets."""
    data_context = ray.data.DataContext.get_current()
    # Continue execution even if some tasks fail.
    data_context.max_errored_blocks = 1_000_000_000
    # Concurrency limits the number of concurrent tasks.
    # It's optional. But setting this will slightly improve performance.
    concurrency = 4 * num_workers

    key_list = _get_training_columns(sdxl, resolution)
    ds = ray.data.read_parquet(
        train_data_uri,
        columns=key_list,
        shuffle=shuffle,
        concurrency=concurrency,
    )
    print(ds)

    def convert_precision(batch):
        for k, v in batch.items():
            if k in TENSOR_KEY_LIST:
                batch[k] = v.astype(np.float16)
        return batch

    return ds.map_batches(
        convert_precision,
        batch_size=None,
        concurrency=concurrency,
    )
