"""
finetune/datasets/native_batch.py  ·  Native-resolution finetune batching
=========================================================================

Finetune defaults to fixed ``input_size`` resize for all backbones (student
included).  Optional pretrain-style native crops are opt-in via
``native_resolution: true`` and ``input_size: 0``:

  image         : (B, C, H_max, W_max)
  mask          : (B, 1, H_max, W_max)
  padding_mask  : (B, ph, pw) bool — True = real patch (stride-4 grid)
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from finetune.base import FinetuneConfig

# Hiera student effective patch stride (see models/student/hiera_backbone.py)
STUDENT_PATCH_STRIDE = 4


def snap_to_stride(n: int, stride: int) -> int:
    """Floor to a multiple of *stride*, at least *stride*."""
    return max(stride, (int(n) // stride) * stride)


def native_crop_shape(
    height: int,
    width: int,
    *,
    patch_stride: int = STUDENT_PATCH_STRIDE,
    max_px: int = 512,
) -> tuple[int, int]:
    """
    Target (H, W) for a native crop: snapped to patch grid, capped like pretrain.
    """
    h = min(max_px, snap_to_stride(height, patch_stride))
    w = min(max_px, snap_to_stride(width, patch_stride))
    return h, w


def extract_native_region(
    tensor: torch.Tensor,
    target_h: int,
    target_w: int,
    *,
    mode: str = "center",
) -> torch.Tensor:
    """
    Crop or pad a CHW tensor to (target_h, target_w) without resizing.

    *mode* ``center``: center crop when larger; zero-pad bottom/right when smaller.
    """
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    _, h, w = tensor.shape
    if h >= target_h and w >= target_w:
        if mode == "center":
            top = (h - target_h) // 2
            left = (w - target_w) // 2
        else:
            top = left = 0
        return tensor[:, top : top + target_h, left : left + target_w].contiguous()

    out = tensor.new_zeros((tensor.shape[0], target_h, target_w))
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    out[:, :copy_h, :copy_w] = tensor[:, :copy_h, :copy_w]
    return out


def padding_mask_for_shape(
    height: int,
    width: int,
    *,
    valid_h: int,
    valid_w: int,
    patch_stride: int = STUDENT_PATCH_STRIDE,
) -> torch.Tensor:
    """Build (ph, pw) bool mask — True on patches that overlap valid content."""
    ph = height // patch_stride
    pw = width // patch_stride
    vph = max(0, min(ph, (valid_h + patch_stride - 1) // patch_stride))
    vpw = max(0, min(pw, (valid_w + patch_stride - 1) // patch_stride))
    mask = torch.zeros(ph, pw, dtype=torch.bool)
    if vph > 0 and vpw > 0:
        mask[:vph, :vpw] = True
    return mask


def uses_native_resolution(cfg: "FinetuneConfig", encoder=None) -> bool:
    """
    Return True only for explicit pretrain-style native crops.

    When ``input_size`` > 0 (the finetune default), every backbone — including
    student / Hiera — uses the same fixed resize as ViT, ResNet, and US FMs.
  """
    input_sz = int(getattr(cfg, "input_size", 224) or 0)
    if input_sz > 0:
        return False

    if not bool(getattr(cfg, "native_resolution", False)):
        return False

    from models.heads.finetune_seg import encoder_has_hierarchical_features

    if encoder is not None:
        return encoder_has_hierarchical_features(encoder)
    return True


def finetune_img_size(cfg: "FinetuneConfig", default: int = 224) -> int:
    """Fixed resize side length; falls back when input_size is 0 (native mode)."""
    sz = int(getattr(cfg, "input_size", None) or cfg.output_size or default)
    return sz if sz > 0 else default


def finetune_native_collate(
    batch: list[dict[str, Any]],
    *,
    patch_stride: int = STUDENT_PATCH_STRIDE,
) -> dict[str, Any]:
    """Pad variable-size native crops to per-batch max (images + masks + pmask)."""
    max_h = max(b["image"].shape[-2] for b in batch)
    max_w = max(b["image"].shape[-1] for b in batch)
    max_ph = max_h // patch_stride
    max_pw = max_w // patch_stride

    images, masks, pmasks = [], [], []
    label_maps = []
    has_labels = "label_map" in batch[0]
    out: dict[str, Any] = {
        k: []
        for k in batch[0]
        if k not in ("image", "mask", "padding_mask", "label_map")
    }

    for b in batch:
        img = b["image"]
        msk = b["mask"]
        _, h, w = img.shape
        _, mh, mw = msk.shape

        pad_img = F.pad(img, (0, max_w - w, 0, max_h - h), value=0.0)
        pad_msk = F.pad(msk, (0, max_w - mw, 0, max_h - mh), value=0.0)
        if has_labels:
            lm = b["label_map"]
            _, lh, lw = lm.shape
            pad_lm = F.pad(lm.float(), (0, max_w - lw, 0, max_h - lh), value=0.0).long()
            label_maps.append(pad_lm)

        pm = b.get("padding_mask")
        if pm is None:
            pm = padding_mask_for_shape(
                max_h, max_w, valid_h=h, valid_w=w, patch_stride=patch_stride,
            )
        else:
            pm = pm.clone()
        pad_pm = torch.zeros(max_ph, max_pw, dtype=torch.bool)
        ph, pw = pm.shape
        pad_pm[:ph, :pw] = pm

        images.append(pad_img)
        masks.append(pad_msk)
        pmasks.append(pad_pm)

        for k in out:
            out[k].append(b[k])

    collated: dict[str, Any] = {
        "image": torch.stack(images, dim=0),
        "mask": torch.stack(masks, dim=0),
        "padding_mask": torch.stack(pmasks, dim=0),
    }
    if label_maps:
        collated["label_map"] = torch.stack(label_maps, dim=0)
    for k, vals in out.items():
        if k in ("cls_label",):
            collated[k] = torch.stack(vals, dim=0)
        else:
            collated[k] = vals
    return collated
