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

    validation_ds = ray.train.get_dataset_shard("validation")
    validation_dataloader = validation_ds.iter_torch_batches(
        batch_size=args.batch_size_per_worker,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_batches=16,
    )

    # Initialize Stable Diffusion Model
    torch.set_float32_matmul_precision("high")

    if args.fsdp:
        import functools
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

        if args.auto_wrap_min_num_param:
            auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=int(args.auto_wrap_min_num_param),
            )
        else:
            auto_wrap_policy = None

        if args.activation_checkpoint_min_num_param:
            activation_checkpointing_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=int(args.activation_checkpoint_min_num_param),
            )
        else:
            activation_checkpointing_policy = None

        strategy = RayFSDPStrategy(
            sharding_strategy=args.sharding_policy,
            state_dict_type=args.checkpoint_sharding_strategy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy=activation_checkpointing_policy,
            use_orig_params=True,
        )
    else:
        strategy = RayDDPStrategy()

    # Initialize WandbLogger if `WANDB_API_KEY` is set.
    wandb_logger = None
    if WANDB_API_KEY is not None:
        wandb.login(key=WANDB_API_KEY)
        wandb_logger = WandbLogger(
            project=args.project_name, name=args.experiment_name, id=trial_name
        )
        if world_rank == 0:
            wandb_logger.log_hyperparams(args)
    else:
        logger.warning("WANDB_API_KEY is not set. Wandb Logging is disabled.")

    # Initialize Lightning Callbacks
    ray_train_reporter = RayTrainReportCallback(
        args.checkpoint_every_n_steps, args.checkpoint_sharding_strategy
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [ray_train_reporter, lr_monitor]
    callbacks.append(BenchmarkCallback(args))

    # Initialize Lightning Trainer
    lightning_log_dir = os.path.join(
        tempfile.gettempdir(), "lightning_logs", trial_name
    )
    os.makedirs(lightning_log_dir, exist_ok=True)
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        check_val_every_n_epoch=None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator="gpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=strategy,
        plugins=[RayLightningEnvironment()],
        callbacks=callbacks,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        default_root_dir=lightning_log_dir,
    )

    if args.sdxl:
        model_cls = StableDiffusionXL
    else:
        model_cls = StableDiffusion

    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        # Continue training from a previous checkpoint
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")

            if args.resume_from_checkpoint:
                # Case 1: Start a new run
                # Only restore the model weights, training starts from step 0
                model = model_cls.load_from_checkpoint(
                    ckpt_path, map_location=torch.device("cpu"), args=args
                )

                trainer.fit(
                    model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=validation_dataloader,
                )
            else:
                # Case 2: Restore from an interrupted/crashed run
                # Restore both the model weights and the trainer states (optimizer, steps, callbacks)
                model = model_cls(args=args)

                trainer.fit(
                    model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=validation_dataloader,
                    ckpt_path=ckpt_path,
                )
    else:
        # Start a new run from scratch
        model = model_cls(args=args)

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=validation_dataloader,
        )


if __name__ == "__main__":
    args = parse_arguments()

    # Configure Ray Datasets
    if args.online_preprocessing:
        train_ds = get_laion_streaming_dataset(
            args.train_data_uri,
            num_encoders=args.num_encoders,
            sdxl=args.sdxl,
            resolution=args.resolution,
            shuffle="files",
            training_batch_size=args.batch_size_per_worker,
        )
        validation_ds = get_laion_streaming_dataset(
            args.validation_data_uri,
            num_encoders=args.num_encoders,
            sdxl=args.sdxl,
            resolution=args.resolution,
            training_batch_size=args.batch_size_per_worker,
        )
    else:
        train_ds = load_precomputed_dataset(
            args.train_data_uri,
            num_workers=args.num_workers,
            sdxl=args.sdxl,
            resolution=args.resolution,
            shuffle="files",
        )

        validation_ds = load_precomputed_dataset(
            args.validation_data_uri,
            num_workers=args.num_workers,
            sdxl=args.sdxl,
            resolution=args.resolution,
        )

    ray_datasets = {"train": train_ds, "validation": validation_ds}

    # Configure Storage Path
    artifact_storage = os.environ["ANYSCALE_ARTIFACT_STORAGE"]
    user_name = re.sub(r"\s+", "__", os.environ.get("ANYSCALE_USERNAME", "user"))
    storage_path = f"{artifact_storage}/{user_name}"

    if args.restore_from_uri:
        print(f"Restore experiment from {args.restore_from_uri}...")
        assert TorchTrainer.can_restore(args.restore_from_uri)
        # If you want to overwite the old arguments, please specify 
        # `TorchTrainer.restore(train_loop_config=)` with the new arguments
        trainer = TorchTrainer.restore(
            args.restore_from_uri,
            datasets=ray_datasets,
            train_loop_per_worker=train_func,
        )
    else:
        checkpoint = None
        if args.resume_from_checkpoint:
            checkpoint = Checkpoint(args.resume_from_checkpoint)

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
            resume_from_checkpoint=checkpoint,
        )
    trainer.fit()
