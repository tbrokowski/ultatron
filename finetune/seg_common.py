"""
finetune/seg_common.py  ·  Shared helpers for segmentation finetune experiments
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from finetune.base import FinetuneConfig
from models.heads import build_finetune_seg_head
from models.heads.seg_losses import binary_segmentation_loss


def build_seg_finetune_head(
    encoder,
    cfg: FinetuneConfig,
    *,
    n_classes: int = 1,
) -> nn.Module:
    """Build a segmentation head with shared enhancement flags from FinetuneConfig."""
    return build_finetune_seg_head(
        encoder=encoder,
        n_classes=n_classes,
        head_type=cfg.head_type,
        patch_size=16,
        output_size=cfg.output_size,
        use_refine_up=getattr(cfg, "refine_up", False),
        use_adapters=getattr(cfg, "seg_use_adapters", True),
        use_aspp=getattr(cfg, "seg_use_aspp", True),
        use_attention_gates=getattr(cfg, "seg_use_attention_gates", True),
    )


def upsample_logits_to_mask(head_output: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Bilinear upsample head logits to GT mask resolution."""
    return F.interpolate(
        head_output,
        size=target_mask.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def compute_binary_seg_loss(
    head_output: torch.Tensor,
    target_mask: torch.Tensor,
    cfg: FinetuneConfig,
    *,
    use_pos_weight: bool = True,
) -> torch.Tensor:
    """Standard frozen-backbone binary seg loss (BCE + Dice + boundary)."""
    pred = upsample_logits_to_mask(head_output, target_mask)
    return binary_segmentation_loss(
        pred,
        target_mask,
        boundary_weight=getattr(cfg, "boundary_loss_weight", 0.5),
        use_pos_weight=use_pos_weight,
    )
