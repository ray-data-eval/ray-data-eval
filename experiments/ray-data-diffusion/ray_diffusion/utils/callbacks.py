import lightning.pytorch as pl
import os
import ray
import ray.train
import shutil
import tempfile
import time
import torch

from contextlib import nullcontext
from datetime import datetime
from ray.train import Checkpoint
from ray_diffusion.utils.utils import strategy_context
from collections import defaultdict

class BenchmarkCallback(pl.callbacks.Callback):
    """
    According to Lightning's hooks: https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = self.args.batch_size_per_worker * self.args.num_workers
        self.frequency = 20
        self.batch_start_time = 0.0
        self.last_batch_end_time = 0.0
        self.forward_start_time = 0.0
        self.zero_grad_start_time = 0.0
        self.backward_start_time = 0.0
        self.backward_end_time = 0.0
        self.fit_start_time = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self.metrics = {}
        self.metrics_window = defaultdict(list)
        self.batch_idx = 0

    def update_metrics(self, key, val, moving_average=True):
        if moving_average:
            if self.batch_idx < self.frequency:
                return
            
            records = self.metrics_window[key]
            records.append(val)

            if len(records) < self.frequency:
                return
            elif len(records) > self.frequency:
                records.pop(0)
            
            self.metrics[key] = sum(records) / len(records)
        else:
            self.metrics[key] = val

    def on_fit_start(self, trainer, pl_module):
        self.fit_start_time = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_idx = batch_idx
        self.batch_start_time = time.time()
        self.update_metrics(
            "benchmark/avg_dataloading(s)",
            self.batch_start_time - self.last_batch_end_time,
        )

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        self.zero_grad_start_time = time.time()
        self.update_metrics(
            "benchmark/avg_forward(s)",
            self.zero_grad_start_time - self.batch_start_time,
        )

    def on_before_backward(self, trainer, pl_module, loss):
        self.backward_start_time = time.time()
        self.update_metrics(
            "benchmark/avg_zero_grad(s)",
            self.backward_start_time - self.zero_grad_start_time,
        )

    def on_after_backward(self, trainer, pl_module):
        self.backward_end_time = time.time()
        self.update_metrics(
            "benchmark/avg_backward(s)",
            self.backward_end_time - self.backward_start_time,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_end_time = time.time()
        self.update_metrics(
            "benchmark/avg_iter(s)", batch_end_time - self.last_batch_end_time
        )
        self.update_metrics(
            "benchmark/avg_optimizer(s)", batch_end_time - self.backward_end_time
        )
        self.last_batch_end_time = batch_end_time

        if (pl_module.current_training_steps + 1) % self.frequency == 0:
            self.metrics["benchmark/training_time"] = (
                batch_end_time - self.fit_start_time
            )
            if "benchmark/avg_iter(s)" in self.metrics:
                self.metrics["benchmark/device_throughput(iter_per_s)"] = (
                    1.0 / self.metrics["benchmark/avg_iter(s)"]
                )
                self.metrics["benchmark/global_throughput(images_per_s)"] = (
                    1.0 / self.metrics["benchmark/avg_iter(s)"] * self.batch_size
                )
            print(self.metrics)
            trainer.logger.log_metrics(
                self.metrics, step=pl_module.current_training_steps
            )


class RayTrainReportCallback(pl.callbacks.Callback):
    def __init__(
        self,
        every_n_train_steps: int = 5000,
        checkpoint_sharding_strategy: str = "full",
        fsdp: bool = False,
    ) -> None:
        super().__init__()
        self.every_n_train_steps = every_n_train_steps
        self.checkpoint_sharding_strategy = checkpoint_sharding_strategy
        self.fsdp = fsdp

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = pl_module.current_training_steps
        if (step + 1) % self.every_n_train_steps != 0:
            return

        temp_checkpoint_dir = os.path.join(
            tempfile.gettempdir(),
            ray.train.get_context().get_trial_name(),
            f"step={step}",
        )
        os.makedirs(temp_checkpoint_dir, exist_ok=True)

        # Fetch metrics
        metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in metrics.items()}

        # (Optional) Add customized metrics
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        # Save checkpoint to local
        ckpt_path = os.path.join(temp_checkpoint_dir, "checkpoint.ckpt")

        with strategy_context(fsdp=self.fsdp, model=trainer.model):
            trainer.save_checkpoint(ckpt_path, weights_only=False)

        if self.checkpoint_sharding_strategy == "full":
            checkpoint = (
                Checkpoint.from_directory(temp_checkpoint_dir)
                if ray.train.get_context().get_world_rank() == 0
                else None
            )
        elif self.checkpoint_sharding_strategy == "sharded":
            checkpoint = (
                Checkpoint.from_directory(temp_checkpoint_dir)
                if ray.train.get_context().get_local_rank() == 0
                else None
            )

        # Report to train session
        ray.train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here
        torch.distributed.barrier()

        # Clean up the checkpoint, since it's already be copied to storage.
        if ray.train.get_context().get_local_rank() == 0:
            shutil.rmtree(temp_checkpoint_dir)

def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"dump_{timestamp}"

    prof.export_memory_timeline(
        f"/mnt/local_storage/{file_prefix}.json", device="cuda:0"
    )


class GPUMemoryProfileCallback(pl.callbacks.Callback):
    def __init__(self):
        self.prof = None
        self.rank = ray.train.get_context().get_world_rank()
        if self.rank == 0:
            self.context = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=trace_handler,
            )
        else:
            self.context = nullcontext()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.rank == 0:
            if batch_idx == 1:
                self.prof = self.context.__enter__()

            if self.prof:
                self.prof.step()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.rank == 0:
            if batch_idx == 3:
                exc_type, exc_value, exc_traceback = None, None, None
                self.context.__exit__(exc_type, exc_value, exc_traceback)
