from typing import List

import numpy as np
import PIL

from .configuration_swin_ir import SwinIRConfig
from .modeling_flax_swin_ir import FlaxSwinIR


def prepare_inputs(images: List["PIL.Image"], window_size=8):
    """
    Prepare inputs for SwinIR model. Assumes that the input `images` are of same size.
    """
    images = [np.array(image, dtype=np.float32) for image in images]
    images = np.concatenate([image[np.newaxis, ...] for image in images], axis=0)
    images = images / 255.0

    _, h_old, w_old, _ = images.shape
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    images = np.concatenate([images, np.flip(images, [1])], 1)[:, : h_old + h_pad, :, :]
    images = np.concatenate([images, np.flip(images, [2])], 2)[:, :, : w_old + w_pad, :]
    return images


def numpy_to_pil(images: np.ndarray):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[None, ...]
    images = np.clip(images, 0, 1)
    images = (images * 255).round().astype("uint8")
    pil_images = [PIL.Image.fromarray(image) for image in images]
    return pil_images
