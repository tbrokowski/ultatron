"""
models/heads/seg_losses.py  ·  Shared segmentation loss helpers
================================================================

Used by finetune experiments (BUSI, TN3K, BUS-BRA, CAMUS binary, …) so all
frozen-backbone segmentation runs share the same BCE + Dice + boundary protocol.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def boundary_map(mask: Tensor, kernel_size: int = 5) -> Tensor:
    """Thin boundary ring via morphological dilation − erosion."""
    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded  = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def binary_segmentation_loss(
    pred_logits: Tensor,
    target_mask: Tensor,
    *,
    boundary_weight: float = 0.5,
    pos_weight_max: float = 50.0,
    use_pos_weight: bool = True,
) -> Tensor:
    """
    BCE (+ optional foreground reweighting) + soft Dice + optional boundary BCE.

    Parameters
    ----------
    pred_logits      : (B, 1, H, W) logits aligned to target resolution
    target_mask      : (B, 1, H, W) float in {0, 1}
    boundary_weight  : weight on boundary-ring BCE (0 disables)
    pos_weight_max   : cap for dynamic pos_weight on small foreground regions
    use_pos_weight   : apply per-sample pos_weight in BCE (BUSI-style)
    """
    if use_pos_weight:
        spatial = target_mask.shape[2] * target_mask.shape[3]
        fg = target_mask.sum(dim=(1, 2, 3))
        bg = spatial - fg
        pos_weight = (bg / (fg + 1.0)).clamp(min=1.0, max=pos_weight_max).view(-1, 1, 1, 1)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target_mask, pos_weight=pos_weight,
        )
    else:
        bce = F.binary_cross_entropy_with_logits(pred_logits, target_mask)

    p_s = torch.sigmoid(pred_logits)
    inter = (p_s * target_mask).sum(dim=(1, 2, 3))
    denom = p_s.sum(dim=(1, 2, 3)) + target_mask.sum(dim=(1, 2, 3))
    dice = 1.0 - (2 * inter + 1) / (denom + 1)

    loss = bce + dice.mean()
    if boundary_weight > 0:
        bnd = boundary_map(target_mask)
        loss = loss + boundary_weight * F.binary_cross_entropy_with_logits(pred_logits, bnd)
    return loss
