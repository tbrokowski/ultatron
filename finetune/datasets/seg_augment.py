"""
finetune/datasets/seg_augment.py  ·  Shared breast-US segmentation augmentations
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def augment_breast_us(
    img: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Spatial + photometric augmentations for breast ultrasound segmentation.
    Geometric transforms are applied identically to image and mask.
    """
    _, h, w = img.shape

    if random.random() < 0.5:
        img  = TF.hflip(img)
        mask = TF.hflip(mask)

    if random.random() < 0.3:
        img  = TF.vflip(img)
        mask = TF.vflip(mask)

    if random.random() < 0.5:
        angle = random.uniform(-20, 20)
        img  = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

    if random.random() < 0.5:
        scale = random.uniform(0.75, 1.25)
        new_h = max(h, int(h * scale))
        new_w = max(w, int(w * scale))
        img = F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0), size=(new_h, new_w),
            mode="nearest",
        ).squeeze(0)
        top  = random.randint(0, new_h - h)
        left = random.randint(0, new_w - w)
        img  = img[:,  top:top + h, left:left + w]
        mask = mask[:, top:top + h, left:left + w]

    if random.random() < 0.5:
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
    if random.random() < 0.3:
        img = img.clamp(0, 1) ** random.uniform(0.7, 1.4)
    if random.random() < 0.4:
        img = (img + torch.randn_like(img) * random.uniform(0.01, 0.05)).clamp(0, 1)

    return img, mask
