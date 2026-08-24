import torch
import lightning.pytorch as pl
import torch.nn.functional as F

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from typing import List
from PIL.Image import Image
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    PretrainedConfig,
    get_linear_schedule_with_warmup,
)
from ray_diffusion.utils.constants import *
from ray_diffusion.utils.utils import strategy_context
from ray.train.torch import get_device


class StableDiffusion(pl.LightningModule):
    """
    model_name: stabilityai/stable-diffusion-2-base -> 4.6G
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()

        # Initialize U-net
        if args.init_from_pretrained:
            self.unet = UNet2DConditionModel.from_pretrained(
                args.model_name, subfolder="unet"
            )
        else:
            model_config = PretrainedConfig.get_config_dict(
                args.model_name, subfolder="unet"
            )
            self.unet = UNet2DConditionModel(**model_config[0])

        if args.use_xformers:
            print("enable xformers memeff attention")
            self.unet.enable_xformers_memory_efficient_attention()

        if self.args.fsdp:
            self.unet = torch.compile(self.unet)

        # Define the training noise schedulers
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.model_name, subfolder="scheduler"
        )

        if args.online_evaluation:
            # Initialize images and captions encoders in FP16
            self.vae = AutoencoderKL.from_pretrained(args.model_name, subfolder="vae")
            self.tokenizer = CLIPTokenizer.from_pretrained(
                args.model_name, subfolder="tokenizer"
            )
            self.text_encoder = CLIPTextModel.from_pretrained(
                args.model_name, subfolder="text_encoder"
            )

            # Freeze encoders during diffusion training
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)

            # Accelerate inference with DDIM (non-Markovian diffusion processes)
            self.inference_noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.noise_scheduler.config.num_train_timesteps,
                beta_start=self.noise_scheduler.config.beta_start,
                beta_end=self.noise_scheduler.config.beta_end,
                beta_schedule=self.noise_scheduler.config.beta_schedule,
                trained_betas=self.noise_scheduler.config.trained_betas,
                clip_sample=self.noise_scheduler.config.clip_sample,
                set_alpha_to_one=self.noise_scheduler.config.set_alpha_to_one,
                prediction_type="epsilon",
            )
            self.inference_noise_scheduler.set_timesteps(args.num_inference_steps)

            # Pipeline to generate Images
            self.pipeline = StableDiffusionPipeline(
                vae=self.vae,
                unet=self.unet,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                scheduler=self.inference_noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
            )
            self.pipeline.set_progress_bar_config(disable=True)

            self.prompt_samples = PROMPT_SAMPLES
            self.prompt_samples = self.prompt_samples[: args.num_prompt_samples]

        # Setup loss function
        self.loss_fn = F.mse_loss
        self.current_training_steps = 0

    def on_fit_start(self) -> None:
        # Move cumprod tensor to GPU in advance to avoid data movement on each step.
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            get_device()
        )

    def forward(self, batch):
        # Use latents if specified and available. When specified, they might not exist during eval
        if self.args.resolution == 256:
            latents = batch[IMAGE_LATENTS_256_KEY]
        elif self.args.resolution == 512:
            latents = batch[IMAGE_LATENTS_512_KEY]
        conditioning = batch[CAPTION_LATENTS_KEY]

        # Sample the diffusion timesteps
        timesteps = torch.randint(
            0, len(self.noise_scheduler), (latents.shape[0],), device=latents.device
        )
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # Forward through the model
        outputs = self.unet(noised_latents, timesteps, conditioning)["sample"]
        return outputs, noise

    def training_step(self, batch, batch_idx):
        outputs, targets = self.forward(batch)
        loss = self.loss_fn(outputs, targets)
        self.log(
            "train/loss_mse", loss.item(), prog_bar=False, on_step=True, sync_dist=False
        )
        self.current_training_steps += 1
        return loss

    def validation_step(self, batch, batch_idx):
        outputs, targets = self.forward(batch)
        loss = self.loss_fn(outputs, targets)
        self.log(
            "validation/loss_mse",
            loss.item(),
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        # Set a large training steps here to keep lr constant after warm-up
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=100000000000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    @torch.no_grad()
    def generate(self, prompts, guidance_scale=7.5) -> List[Image]:
        return self.pipeline(
            prompts,
            guidance_scale=guidance_scale,
            width=self.args.resolution,
            height=self.args.resolution,
        ).images

    def on_validation_epoch_end(self) -> None:
        # Generate sample images for the prompt samples
        if self.args.online_evaluation:
            for guidance_scale in [1, 3, 5, 7]:
                with strategy_context(fsdp=self.args.fsdp, model=self.trainer.model):
                    images = self.generate(
                        self.prompt_samples,
                        guidance_scale=guidance_scale,
                    )

                if self.global_rank == 0:
                    self.logger.log_image(
                        key=f"sampled_images-gs={guidance_scale}",
                        step=self.current_training_steps,
                        images=images,
                        caption=self.prompt_samples,
                    )
