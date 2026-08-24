import os
import re
import tempfile
import torch
import wandb
import lightning.pytorch as pl
import logging
import numpy as np

from functools import partial
from lightning.pytorch.loggers.wandb import WandbLogger

from ray_diffusion.models import StableDiffusion, StableDiffusionXL
from ray_diffusion.utils.constants import *
from ray_diffusion.utils.callbacks import (
    RayTrainReportCallback,
    BenchmarkCallback,
)
from ray_diffusion.utils.utils import (
    parse_arguments,
    RayFSDPStrategy,
)
from ray_diffusion.data import (
    load_precomputed_dataset,
    get_laion_streaming_dataset,
    default_collate_fn,
)
from torch.distributed.fsdp import BackwardPrefetch

import ray.train
from ray.train import FailureConfig, RunConfig, ScalingConfig, Checkpoint
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import seed_everything

logger = logging.getLogger(__name__)

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
import time

def train_func(config):
    args = config["args"]
    seed_everything(args.seed)
    world_rank = ray.train.get_context().get_world_rank()
    trial_name = ray.train.get_context().get_trial_name()

    # Prepare Ray datasets
    collate_fn = partial(default_collate_fn, device=ray.train.torch.get_device())

    train_ds = ray.train.get_dataset_shard("train")
    train_dataloader = train_ds.iter_torch_batches(
        batch_size=args.batch_size_per_worker,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_batches=16,
    )

    # Initialize WandbLogger if `WANDB_API_KEY` is set.
    wandb_logger = None
    if WANDB_API_KEY is not None:
        wandb.login(key=WANDB_API_KEY)
        wandb_logger = WandbLogger(
            project=args.project_name, name=args.experiment_name, id=trial_name
        )
        if world_rank == 0:
            wandb_logger.log_hyperparams(args)


    init_t = t = time.time()
    for i, batch in enumerate(train_dataloader):
        time.sleep(0.9227)
        cur_t = time.time()
        iter_s = cur_t - t 
        t = cur_t

        if world_rank == 0:
            wandb_logger.log_metrics(
                {
                    "Global Throughput (img/s)": 4096.0 / iter_s,
                    "iter (s)": iter_s,
                    "iter_per_s": 1.0 / iter_s,
                    "t": t - init_t
                }, 
                step=i
            )


if __name__ == "__main__":
    args = parse_arguments()

    train_ds = get_laion_streaming_dataset(
        args.train_data_uri,
        num_encoders=args.num_encoders,
        sdxl=args.sdxl,
        resolution=args.resolution,
        shuffle="file",
        training_batch_size=args.batch_size_per_worker,
    )

    ray_datasets = {"train": train_ds}

    # Configure Storage Path
    artifact_storage = os.environ["ANYSCALE_ARTIFACT_STORAGE"]
    user_name = re.sub(r"\s+", "__", os.environ.get("ANYSCALE_USERNAME", "user"))
    storage_path = f"{artifact_storage}/{user_name}"

    trainer = TorchTrainer(
        train_func,
        train_loop_config={"args": args},
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=True),
        run_config=RunConfig(
            name=args.experiment_name,
            storage_path=storage_path,
            failure_config=FailureConfig(max_failures=args.max_failures),
        ),
        datasets=ray_datasets,
    )
    trainer.fit()
