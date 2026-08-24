import argparse
import json
import ray
import torch

from .constants import *
from contextlib import nullcontext
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class RayFSDPStrategy(FSDPStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def root_device(self) -> torch.device:
        return ray.train.torch.get_device()

    @property
    def distributed_sampler_kwargs(self):
        return dict(
            num_replicas=self.world_size,
            rank=self.global_rank,
        )


def strategy_context(fsdp=False, model=None):
    if fsdp:
        return FSDP.summon_full_params(model, writeback=False, recurse=False)
    else:
        return nullcontext()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=None, type=str, help="The configuration file."
    )
    parser.add_argument(
        "--project_name",
        default="ray-diffusion",
        type=str,
        help="Project name for Wandb.",
    )
    parser.add_argument(
        "--experiment_name",
        default="exp-base",
        type=str,
        help="Experiment name for both Ray Train and Wandb.",
    )

    # Model
    parser.add_argument(
        "--model_name",
        default="stabilityai/stable-diffusion-2-base",
        type=str,
    )
    parser.add_argument(
        "--init_from_pretrained",
        default=False,
        action="store_true",
        help="Turn it on for finetuning, otherwise pretraining from scrach.",
    )
    parser.add_argument("--sdxl", default=False, action="store_true")

    # Data
    parser.add_argument(
        "--online_preprocessing",
        default=False,
        action="store_true",
        help="Whether to enable online preprocessing. If this is enabled, "
        "you should provide the raw data path to the `train_data_uri` and `validation_data_uri`.",
    )
    parser.add_argument(
        "--num_encoders",
        default=16,
        type=int,
        help="The quantity of encoders designated for online preprocessing"
        "It should match the number of GPUs allocated for data processing.",
    )
    parser.add_argument(
        "--train_data_uri",
        type=str,
        help="Raw data uri here if enabled online_preprocessing, otherwise use the processed data uri.",
    )
    parser.add_argument(
        "--validation_data_uri",
        type=str,
        help="Raw data uri here if enabled online_preprocessing, otherwise use the processed data uri.",
    )
    parser.add_argument("--resolution", default=256, type=int)

    # Hyperparams
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--max_epochs", default=400, type=int)
    parser.add_argument("--num_warmup_steps", default=10000, type=int)
    parser.add_argument("--batch_size_per_worker", default=32, type=int)
    parser.add_argument("--global_batch_size", default=2048, type=int)
    parser.add_argument("--accumulate_grad_batches", default=4, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--use_xformers", default=False, action="store_true")

    # FSDP
    parser.add_argument(
        "--fsdp",
        default=False,
        action="store_true",
        help="whether to enable fsdp training",
    )
    parser.add_argument(
        "--sharding_policy",
        default="HYBRID_SHARD",
        type=str,
        help="FSDP sharding policy. one of [FULL_SHARD, HYBRID_SHARD, or SHARD_GRAD_OP]",
    )
    parser.add_argument(
        "--checkpoint_sharding_strategy",
        default="full",
        type=str,
        help="one of ['full', 'sharded']",
    )
    parser.add_argument("--auto_wrap_min_num_param", default=0, type=float)
    parser.add_argument("--activation_checkpoint_min_num_param", default=0, type=float)

    # Trainer
    parser.add_argument(
        "--seed",
        default=420,
        type=int,
    )
    parser.add_argument(
        "--checkpoint_every_n_steps",
        default=20000,
        type=int,
        help="The frequency(steps) of checkpointing.",
    )

    parser.add_argument(
        "--online_evaluation",
        default=False,
        action="store_true",
        help="Whether to do online evaluation. If enabled, it will increase GPU memory consumption.",
    )
    # The arguments below are effective only when online evaluation is enabled.
    parser.add_argument(
        "--val_check_interval",
        default=10000,
        type=int,
        help="Specifies the frequency (in steps) for checking the validation dataset.",
    )
    parser.add_argument(
        "--num_inference_steps",
        default=50,
        type=int,
        help="The number of denoising steps for image generation.",
    )
    parser.add_argument(
        "--num_prompt_samples",
        default=10,
        type=int,
        help="The number of samples to generate in validation epoch end.",
    )

    # Fault-tolerant Training
    parser.add_argument(
        "--restore_from_uri",
        type=str,
        help="The path to a Ray Train experiment folder for training restoration.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Create a new trainer and resume from a previously saved checkpoint.",
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        default=0,
        help="Tries to automatically recover the experiment at least this many times. "
        "Setting to -1 means infinite recovery retries. Setting to 0 will disable retries. "
        "Defaults to 0.",
    )

    args = parser.parse_args()

    # Priority: cmd line args > config file > default values
    if args.config:
        with open(args.config, "r") as f:
            parser.set_defaults(**json.load(f))
    args = parser.parse_args()

    print("Training configuration: ")
    for k, v in args.__dict__.items():
        print(k, v)

    assert (
        args.global_batch_size
        == args.accumulate_grad_batches * args.num_workers * args.batch_size_per_worker
    )
    return args
