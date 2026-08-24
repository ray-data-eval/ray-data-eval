from torchvision import transforms
from transformers import CLIPTokenizer

from ray_diffusion.utils.constants import *
from ray_diffusion.data.utils import LargestCenterSquare, RandomCropSquare


class SDPreprocessor:
    """Transform images and tokenize image captions for Stable Diffusion."""

    def __init__(self, resolution=256) -> None:
        self.resolution = resolution
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.crop = LargestCenterSquare(resolution)
        self.transforms = transforms.Compose([transforms.ToTensor(), normalize])

        self.text_tokenizer = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-2-base", subfolder="tokenizer"
        )

    def transform(self, image):
        image, size_conditioning = self.crop(image)
        return {
            f"image_{self.resolution}": self.transforms(image),
            CROP_AND_SIZE_CONDITIONING_KEY: size_conditioning,
        }

    def tokenize(self, text):
        caption_ids = self.text_tokenizer(
            text,
            padding="max_length",
            max_length=self.text_tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )["input_ids"][0]

        return {"caption_ids": caption_ids}


class SDXLPreprocessor:
    """Transform images and tokenize image captions for Stable Diffusion XL."""

    def __init__(self, resolution=256, crop_mode="static") -> None:
        self.resolution = resolution
        if crop_mode == "static":
            self.crop = LargestCenterSquare(resolution)
        elif crop_mode == "random":
            self.crop = RandomCropSquare(resolution)
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.transforms = transforms.Compose([transforms.ToTensor(), normalize])

        self.text_tokenizer_sdxl_1 = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer"
        )
        self.text_tokenizer_sdxl_2 = CLIPTokenizer.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2"
        )

    def transform(self, image):
        image, size_conditioning = self.crop(image)
        return {
            f"image_{self.resolution}": self.transforms(image),
            CROP_AND_SIZE_CONDITIONING_KEY: size_conditioning,
        }

    def tokenize(self, text):
        caption_ids_sdxl_1 = self.text_tokenizer_sdxl_1(
            text,
            padding="max_length",
            max_length=self.text_tokenizer_sdxl_1.model_max_length,
            truncation=True,
            return_tensors="np",
        )["input_ids"][0]

        caption_ids_sdxl_2 = self.text_tokenizer_sdxl_2(
            text,
            padding="max_length",
            max_length=self.text_tokenizer_sdxl_2.model_max_length,
            truncation=True,
            return_tensors="np",
        )["input_ids"][0]

        return {
            "caption_ids_sdxl_1": caption_ids_sdxl_1,
            "caption_ids_sdxl_2": caption_ids_sdxl_2,
        }
