import torch
import numpy as np

from torchvision import transforms
from torchvision.transforms.functional import crop

from ray_diffusion.utils.constants import *


def default_collate_fn(batch, device):
    for k, v in batch.items():
        if k in TENSOR_KEY_LIST:
            batch[k] = torch.tensor(v).to(device)
        else:
            batch[k] = v.tolist()
    return batch


def convert_tensor_to_array(tensor, dtype=np.float32):
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype)


def delete_tensor(tensor):
    tensor.detach()
    tensor.grad = None
    tensor.storage().resize_(0)
    del tensor


class LargestCenterSquare:
    """Center crop to the largest square of a PIL image."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        orig_w, orig_h = img.size
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = transforms.functional.resize(img, self.size, antialias=True)

        # Then take a center crop to a square.
        w, h = img.size
        c_top = (h - self.size) // 2
        c_left = (w - self.size) // 2
        img = crop(img, c_top, c_left, self.size, self.size)
        tgt_w, tgt_h = img.size
        return img, [orig_w, orig_h, c_top, c_left, tgt_w, tgt_h]


class RandomCropSquare:
    """Randomly crop square of a PIL image and return the crop parameters."""

    def __init__(self, size):
        self.size = size
        self.random_crop = transforms.RandomCrop(size)

    def __call__(self, img):
        orig_w, orig_h = img.size
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = transforms.functional.resize(img, self.size, antialias=True)
        # Then take a center crop to a square & return crop params.
        c_top, c_left, h, w = self.random_crop.get_params(img, (self.size, self.size))
        img = crop(img, c_top, c_left, h, w)
        tgt_w, tgt_h = img.size
        return img, [orig_w, orig_h, c_top, c_left, tgt_w, tgt_h]
