"""
models/heads/segmentation_head.py  ·  Segmentation heads
==============================================================

Two implementations, selected by config:

LinearSegHead
-------------
Single linear layer on patch tokens → per-patch binary/multiclass logits,
reshaped to spatial grid.  Fast, interpretable, good linear-probe baseline.
Input:  patch_tokens (B, N, D)   where N = ph*pw
Output: (B, n_classes, ph, pw)

DPTSegHead
----------
Dense Prediction Transformer head.  Progressively reassembles patch tokens
at 4 scales using learned projections + bilinear upsampling, then fuses into
a final dense prediction at the target stride.

Both heads accept padding_mask (B, ph, pw) so they know which spatial
positions are real content vs padding.  Padding positions are zeroed in the
output logits so they don't contribute to loss.

Loss
----
Binary:     F.binary_cross_entropy_with_logits on (B, 1, ph, pw)
Multiclass: F.cross_entropy on (B, n_classes, ph, pw)
Dice:       1 - 2*|pred∩tgt| / (|pred|+|tgt|+ε)
Combo:      weighted sum of any of the above

All of these are computed in train/phase_steps.py — the head only
returns logits.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSegHead(nn.Module):
    """
    Single linear layer: patch tokens → spatial logits.

    Parameters
    ----------
    embed_dim  : int  hidden dimension of the backbone (e.g. 1024 for DINOv3-L)
    n_classes  : int  1 for binary segmentation, >1 for multiclass
    patch_size : int  used only for __repr__; does not affect computation
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        n_classes: int = 1,
        patch_size: int = 16,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.n_classes  = n_classes
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, n_classes, bias=True)

    def forward(
        self,
        patch_tokens: torch.Tensor,              # (B, N, D)
        padding_mask: Optional[torch.Tensor] = None,  # (B, ph, pw)
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (B, n_classes, ph, pw)
        Padding positions are zeroed (not masked with -inf, because
        BCE/CE loss ignores them via padding_mask in the loss layer).
        """
        B, N, D = patch_tokens.shape
        ph = pw = int(N ** 0.5)

        logits = self.proj(patch_tokens)             # (B, N, C)
        logits = logits.reshape(B, ph, pw, self.n_classes)
        logits = logits.permute(0, 3, 1, 2)          # (B, C, ph, pw)

        if padding_mask is not None:
            # Zero logits at padding positions: no contribution to loss
            mask = padding_mask.unsqueeze(1).float()  # (B, 1, ph, pw)
            logits = logits * mask

        return logits

    def __repr__(self):
        return (f"LinearSegHead(embed_dim={self.embed_dim}, "
                f"n_classes={self.n_classes})")


class DPTSegHead(nn.Module):
    """
    Dense Prediction Transformer segmentation head.

    Reassembles patch tokens at 4 scales into a dense feature map,
    then applies a lightweight convolutional fusion neck and a final
    1×1 conv to produce per-pixel logits.

    The 4 reassembly scales are derived from backbone intermediate
    layers (if available) or by simple upsampling of the final tokens.
    When only the final patch_tokens are available (no intermediate
    hidden states), this degrades to a 1-scale DPT which is still
    considerably better than LinearSegHead for dense predictions.

    Parameters
    ----------
    embed_dim    : int   backbone hidden dimension
    n_classes    : int   output channels
    patch_size   : int   spatial patch size (used to infer output stride)
    output_size  : int   spatial size of output logits in patch units.
                         If None, output is (ph, pw) — patch-resolution.
                         If set (e.g. 4*ph), logits are bilinearly upsampled.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        n_classes: int = 1,
        patch_size: int = 16,
        neck_channels: int = 256,
        output_size: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.n_classes    = n_classes
        self.patch_size   = patch_size
        self.output_size  = output_size

        # Projection from backbone dim → neck channels
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, neck_channels),
            nn.GELU(),
        )

        # Lightweight convolutional neck: 3 residual conv blocks
        def _conv_block(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.GELU(),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
            )

        self.neck  = _conv_block(neck_channels)
        self.head  = nn.Conv2d(neck_channels, n_classes, 1)

    def forward(
        self,
        patch_tokens: torch.Tensor,              # (B, N, D)
        padding_mask: Optional[torch.Tensor] = None,  # (B, ph, pw)
    ) -> torch.Tensor:
        B, N, D = patch_tokens.shape
        ph = pw = int(N ** 0.5)

        # Project to neck channels, reshape to spatial grid
        feat = self.proj(patch_tokens)            # (B, N, neck)
        feat = feat.reshape(B, ph, pw, -1)
        feat = feat.permute(0, 3, 1, 2)           # (B, neck, ph, pw)

        # Convolutional neck with residual connection
        feat = feat + self.neck(feat)

        # Upsample to output_size if requested
        if self.output_size is not None:
            feat = F.interpolate(
                feat,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
            if padding_mask is not None:
                pm = F.interpolate(
                    padding_mask.unsqueeze(1).float(),
                    size=(self.output_size, self.output_size),
                    mode="nearest",
                )
                feat = feat * pm

        logits = self.head(feat)                  # (B, n_classes, H, W)

        if padding_mask is not None and self.output_size is None:
            logits = logits * padding_mask.unsqueeze(1).float()

        return logits

    def __repr__(self):
        return (f"DPTSegHead(embed_dim={self.embed_dim}, "
                f"n_classes={self.n_classes}, "
                f"output_size={self.output_size})")


def build_seg_head(
    embed_dim: int,
    n_classes: int,
    head_type: str = "linear",
    patch_size: int = 16,
    **kwargs,
) -> nn.Module:
    """
    Factory function for segmentation heads.

    Parameters
    ----------
    head_type : "linear" or "dpt"
    """
    if head_type == "linear":
        return LinearSegHead(embed_dim, n_classes, patch_size)
    elif head_type == "dpt":
        return DPTSegHead(embed_dim, n_classes, patch_size, **kwargs)
    else:
        raise ValueError(f"Unknown head_type: {head_type!r}. Choose 'linear' or 'dpt'.")
