import gc
import torch

from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from ray_diffusion.utils.constants import *
from ray_diffusion.data.utils import convert_tensor_to_array, delete_tensor


class SDLatentEncoder:
    """
    Latent encoder for SD-base data preprocessing.
    """

    def __init__(
        self, model_name="stabilityai/stable-diffusion-2-base", resolution=256
    ):
        # self.device_id = ray.get_gpu_ids()[0]
        self.device = "cuda"
        self.resolution = resolution

        # Image and text encoders
        self.vae = AutoencoderKL.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.float16
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.float16
        )

        # Move the encoders to GPU
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)

    def __call__(self, batch):
        with torch.no_grad():
            # Step 1: Encode images
            input_images = torch.tensor(
                batch[f"image_{self.resolution}"], device=self.device
            )
            del batch[f"image_{self.resolution}"]

            latent_dist = self.vae.encode(input_images.half())["latent_dist"]
            image_latents = latent_dist.sample() * 0.18215
            assert not torch.isnan(image_latents).any()

            if self.resolution == 256:
                # Shape = [batch_size, 4, 32, 32]
                batch[IMAGE_LATENTS_256_KEY] = convert_tensor_to_array(image_latents)
            elif self.resolution == 512:
                # Shape = [batch_size, 4, 64, 64]
                batch[IMAGE_LATENTS_512_KEY] = convert_tensor_to_array(image_latents)

            delete_tensor(input_images)
            delete_tensor(image_latents)
            gc.collect()
            # torch.cuda.empty_cache()

            # Step 2: Encode captions
            caption_ids_tensor = torch.tensor(batch["caption_ids"], device=self.device)
            del batch["caption_ids"]
            caption_latents_tensor = self.text_encoder(caption_ids_tensor)[0]

            # Shape = [batch_size, 77, 1024]
            batch[CAPTION_LATENTS_KEY] = convert_tensor_to_array(caption_latents_tensor)
        return batch


class SDXLLatentEncoder:
    """
    Latent encoder for SDXL data preprocessing.
    """

    def __init__(
        self, model_name="stabilityai/stable-diffusion-xl-base-1.0", resolution=512
    ):
        # self.device_id = ray.get_gpu_ids()[0]
        self.device = "cuda"
        self.resolution = resolution

        # Image and text encoders
        # Use a fixed version to avoid generating nan
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        self.text_encoder_1 = CLIPTextModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.float16
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            model_name, subfolder="text_encoder_2", torch_dtype=torch.float16
        )

        # Move the encoders to GPU
        self.vae = self.vae.to(self.device)
        self.text_encoder_1 = self.text_encoder_1.to(self.device)
        self.text_encoder_2 = self.text_encoder_2.to(self.device)

    def __call__(self, batch):
        # Construct input tensors for image and caption token ids
        with torch.no_grad():
            # Step 1: Encode images with VAE
            input_images = torch.tensor(batch[f"image_{self.resolution}"]).to(
                self.device
            )
            del batch[f"image_{self.resolution}"]

            latent_dist = self.vae.encode(input_images.half())["latent_dist"]
            image_latents = latent_dist.sample() * 0.13025
            assert not torch.isnan(image_latents).any()

            if self.resolution == 512:
                # Shape = [batch_size, 4, 64, 64]
                batch[SDXL_IMAGE_LATENTS_512_KEY] = convert_tensor_to_array(
                    image_latents
                )
            elif self.resolution == 256:
                # Shape = [batch_size, 4, 32, 32]
                batch[SDXL_IMAGE_LATENTS_256_KEY] = convert_tensor_to_array(
                    image_latents
                )

            delete_tensor(input_images)
            delete_tensor(image_latents)
            gc.collect()
            # torch.cuda.empty_cache()

            # Step 2: Encode captions with text encoder 1
            input_caption_1 = torch.tensor(
                batch["caption_ids_sdxl_1"], device=self.device
            )
            del batch["caption_ids_sdxl_1"]

            output_1 = self.text_encoder_1(input_caption_1, output_hidden_states=True)
            encoder_hidden_states_1 = output_1.hidden_states[-2]

            # Shape = [batch_size, 77, 768]
            batch[SDXL_CAPTION_LATENTS_1_KEY] = convert_tensor_to_array(
                encoder_hidden_states_1
            )

            # Step 3: Encode captions with text encoder 2
            input_caption_2 = torch.tensor(
                batch["caption_ids_sdxl_2"], device=self.device
            )
            del batch["caption_ids_sdxl_2"]

            output_2 = self.text_encoder_2(input_caption_2, output_hidden_states=True)
            encoder_hidden_states_2 = output_2.hidden_states[-2]
            add_text_embeds = output_2[0]

            # Shape = [batch_size, 77, 1280]
            batch[SDXL_CAPTION_LATENTS_2_KEY] = convert_tensor_to_array(
                encoder_hidden_states_2
            )
            # Shape = [batch_size, 1280]
            batch[SDXL_ADD_TEXT_EMBEDS_KEY] = convert_tensor_to_array(add_text_embeds)

        return batch
