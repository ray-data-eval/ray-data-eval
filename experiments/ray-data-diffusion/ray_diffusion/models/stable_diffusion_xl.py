import torch
import lightning.pytorch as pl
import torch.nn.functional as F

from diffusers import (
    EulerDiscreteScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import (
    PretrainedConfig,
    get_linear_schedule_with_warmup,
)

from ray_diffusion.utils.constants import *
from ray.train.torch import get_device


class StableDiffusionXL(pl.LightningModule):
    """
    model_name: stabilityai/stable-diffusion-xl-base-1.0 -> 13.5G
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        self.cuda_device = get_device()

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

        # Define the training noise schedulers
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            args.model_name, subfolder="scheduler"
        )

        # Accelerate inference with EulerDiscreteScheduler
        self.inference_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            args.model_name, subfolder="scheduler"
        )

        # Setup loss function
        self.loss_fn = F.mse_loss

        self.prompt_samples = PROMPT_SAMPLES
        self.prompt_samples = self.prompt_samples[: args.num_prompt_samples]

        self.current_training_steps = 0

    def forward(self, batch):
        # Use latents if specified and available. When specified, they might not exist during eval
        if self.args.resolution == 256:
            latents = batch[SDXL_IMAGE_LATENTS_256_KEY]
        else:
            latents = batch[SDXL_IMAGE_LATENTS_512_KEY]

        conditioning, conditioning_2, add_text_embeds = (
            batch[SDXL_CAPTION_LATENTS_1_KEY],
            batch[SDXL_CAPTION_LATENTS_2_KEY],
            batch[SDXL_ADD_TEXT_EMBEDS_KEY],
        )

        conditioning = torch.concat([conditioning, conditioning_2], dim=-1)

        # Sample the diffusion timesteps
        timesteps = torch.randint(
            0, len(self.noise_scheduler), (latents.shape[0],), device=latents.device
        )
        # Add noise to the inputs (forward diffusion)
        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Construct additional conditions
        # We already cropped the image to 512 x 512 in preprocessing, so no crop here.
        # format: [ori_width, ori_height, top_left_x, top_left_y, tgt_width, tgt_height]
        # TODO(yunxuanx): Implement SDXL random cropping algorithm
        add_time_ids = torch.tensor(
            [[512, 512, 0.0, 0.0, 512, 512]], device=self.cuda_device
        ).repeat(latents.shape[0], 1)

        additional_conditions = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }

        # Forward through the model
        outputs = self.unet(
            noised_latents,
            timesteps,
            conditioning,
            added_cond_kwargs=additional_conditions,
        )
        return outputs["sample"], noise

    def training_step(self, batch, batch_idx):
        outputs, targets = self.forward(batch)
        loss = self.loss_fn(outputs, targets)
        self.log(
            "train/loss_mse", loss.item(), prog_bar=True, on_step=True, sync_dist=False
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

    def configure_optimizers(self):
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
