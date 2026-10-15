# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""

import math
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import _log_api_usage_once


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, img, target=None):
        if isinstance(img, torch.Tensor):
            return img, target
        return T.ToTensor()(img), target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Rescale(object):
    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, img, target=None):
        img = img / 65535.0
        if target is None:
            return img
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ChannelAttenuation:
    def __init__(self, min_gain=0.5, max_gain=1.0, p=0.0):
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() > self.p:
            return x

        gain = torch.empty(x.shape[0], 1, 1, device=x.device).uniform_(
            self.min_gain, self.max_gain
        )
        return torch.clamp(x * gain, 0.0, 1.0)


class StatefulChannelOscillation:
    def __init__(
        self,
        C=3,
        amp=1,
        freq=0.25,
        constant_mod=False,
    ):
        self.C = C
        self.amp = amp
        self.freq = freq
        self.constant_mod = constant_mod

        # phase per channel
        self._reset()
        # g = torch.Generator()
        # g.manual_seed(42)
        # self.phase = torch.rand(C, generator=g) * 2 * math.pi

    def __call__(self, x, target=None):
        C, H, W = x.shape
        assert C == self.C

        self.t += 1
        t = self.t

        # one shared oscillation for all channels
        base_phase = self.phase[0]
        sine = torch.sin(self.freq * t + base_phase)  # scalar

        gain = 1 + self.amp * sine

        x = x * gain

        return torch.clamp(x, 0.0, 1.0), target

    def _reset(self):
        # reset generator, phase, and time counter
        g = torch.Generator()
        g.manual_seed(42)
        self.phase = torch.rand(self.C, generator=g) * 2 * math.pi
        self.t = 0


class MinMaxNormalize(object):
    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, input, target=None):
        min_val = input.amin(dim=(1, 2), keepdim=True)
        max_val = input.amax(dim=(1, 2), keepdim=True)

        normalized = (input - min_val) / (max_val - min_val)

        if target is None:
            return normalized

        return normalized, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class MinMaxNormalizePercentile:
    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, input, target=None):
        normalized = input.clone()
        for c in range(input.shape[0]):
            channel_data = input[c]

            p1 = torch.quantile(channel_data.flatten(), 0.01)
            p99 = torch.quantile(channel_data.flatten(), 0.99)

            clipped_channel = torch.clamp(channel_data, p1.item(), p99.item())

            normalized[c] = (clipped_channel - p1) / (p99 - p1)

        if target is None:
            return normalized

        return normalized, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Grayscale(object):
    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, img, target):
        return T.Grayscale()(img), target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Resize(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, img, target):
        _, w, h = img.shape
        img = F.resize(img, [int(w / self.ratio), int(h / self.ratio)], antialias=True)
        target["boxes"] = target["boxes"] / self.ratio
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)

        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return vflip(img, target)

        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GaussianBlur(object):
    def __init__(self, kernel_size, p=0.2):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return T.GaussianBlur(kernel_size=self.kernel_size)(img), target

        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class GaussianNoise(object):
    def __init__(self, mean=0.1, std=0.05, p=0.2):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.std + self.mean
            return img + noise, target
        return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ColorJitter(object):
    def __init__(self, brightness=0.25, contrast=0.25, saturation=0.1, hue=0.05, p=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            t = T.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
            return t(img), target
        else:
            return img, target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def hflip(image, target):
    flipped_image = F.hflip(image)

    _, w, _ = image.shape

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes[:, 0] = 1 - boxes[:, 0]
        target["boxes"] = boxes

    return flipped_image, target


def vflip(image, target):
    flipped_image = F.vflip(image)

    _, w, _ = image.shape

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes[:, 1] = 1 - boxes[:, 1]
        target["boxes"] = boxes

    return flipped_image, target


def crop_image_variable(image, centroid, window_size):
    """Crops a patch from the input image centered around the specified centroid if possible. If the patch would exceed the image boundaries, the patch center is adjusted accordingly.

    Parameters
    ----------
    image : ndarray
        Input image.
    centroid : list
        List containing the x and y coordinates of the centroid.
    window_size : int
        Specifies half of the patch size.

    Returns
    -------
    ndarray
        Cropped patch.
    """

    # Calculate the cropping region boundaries
    x1 = max(0, centroid[0] - window_size)
    y1 = max(0, centroid[1] - window_size)
    x2 = min(image.shape[1], centroid[0] + window_size)
    y2 = min(image.shape[0], centroid[1] + window_size)

    # Calculate expansion needed in each direction
    left_expand = window_size - (centroid[0] - x1)
    right_expand = window_size - (x2 - centroid[0])
    top_expand = window_size - (centroid[1] - y1)
    bottom_expand = window_size - (y2 - centroid[1])

    # Adjust the patch size in all directions if necessary
    x1 = int(max(0, x1 - left_expand))
    y1 = int(max(0, y1 - top_expand))
    x2 = int(min(image.shape[1], x2 + right_expand))
    y2 = int(min(image.shape[0], y2 + bottom_expand))

    # Crop the patch, ensuring it has the same dtype as the input image
    patch = image[y1:y2, x1:x2].astype(np.float32)

    return np.array(patch)


def crop_image_centered(image, centroid, window_size):
    """Crops a patch from the input image centered around the specified centroid and pads it to the specified patch_size.

    Parameters
    ----------
    image : tensor
        Input image.
    centroid : list
        List containing the x and y coordinates of the centroid.
    window_size : int
        Specifies half of the patch size.

    Returns
    -------
    tensor
        Cropped and resized patch.
    """

    # Calculate the cropping region boundaries
    x1 = int(max(0, centroid[0] - window_size))
    y1 = int(max(0, centroid[1] - window_size))
    x2 = int(min(image.shape[2], centroid[0] + window_size))
    y2 = int(min(image.shape[1], centroid[1] + window_size))

    # Crop the patch, ensuring it has the same dtype as the input image
    # patch = image[y1:y2, x1:x2]  # .astype(np.float32)
    patch = F.crop(image, y1, x2, y2 - y1, x2 - x1)

    # Calculate the padding needed to reach the specified patch_size
    pad_bottom = int(window_size - (y2 - centroid[1]))
    pad_top = int(window_size - (centroid[1] - y1))
    pad_right = int(window_size - (x2 - centroid[0]))
    pad_left = int(window_size - (centroid[0] - x1))

    patch = T.Pad(padding=[pad_left, pad_top, pad_right, pad_bottom])(patch)

    return patch
