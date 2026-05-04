"""
models/heads/regression_head.py  ·  Regression and measurement heads
==========================================================================

Covers continuous output tasks:

RegressionHead
--------------
Continuous scalar output from CLS token.
Used for: ejection fraction % (EchoNet-Dynamic), fractional shortening,
global longitudinal strain.

Loss: MSE or MAE depending on LossType in config.

MeasurementHead
---------------
Physical measurement prediction in mm from patch tokens.
Uses attentive pooling to locate the measurement axis, then
predicts a scalar distance.
Used for: HC18 head circumference (mm), fetal abdominal circumference.

Loss: MAE with an optional normalisation factor to account for the
pixel-to-mm conversion stored in the manifest entry metadata.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionHead(nn.Module):
    """
    Scalar regression from CLS token.

    Parameters
    ----------
    embed_dim  : int
    output_min : float   lower clamp for output (e.g. 0.0 for EF%)
    output_max : float   upper clamp for output (e.g. 100.0 for EF%)
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
        output_min: float = 0.0,
        output_max: float = 100.0,
    ):
        super().__init__()
        self.output_min = output_min
        self.output_max = output_max
        output_layer = nn.Linear(hidden_dim, 1)
        if output_min is not None and output_max is not None:
            nn.init.constant_(output_layer.bias, (output_min + output_max) / 2.0)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            output_layer,
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        # cls: (B, D) → (B, 1) → optionally clamped scalar
        out = self.net(cls)                      # (B, 1)
        # Only clamp at inference — during training the raw logit must stay
        # unclamped so gradients flow freely.  Hard-clamping to [10, 85] kills
        # all gradients when random-init outputs land below the lower bound.
        if not self.training:
            if self.output_min is not None or self.output_max is not None:
                out = torch.clamp(out, self.output_min, self.output_max)
        return out.squeeze(-1)                   # (B,)

    def __repr__(self):
        return (f"RegressionHead(D={self.net[1].in_features}, "
                f"range=[{self.output_min}, {self.output_max}])")


class MeasurementHead(nn.Module):
    """
    Physical measurement prediction from patch tokens via attentive pooling.

    Produces a raw pixel-distance prediction; the calling code applies
    the pixel-to-mm conversion from the manifest entry.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,              # (B, N, D)
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = patch_tokens.shape

        scores = self.attn_score(patch_tokens).squeeze(-1)   # (B, N)
        if padding_mask is not None:
            flat = padding_mask.flatten(1)
            scores = scores.masked_fill(~flat, float("-inf"))

        weights = F.softmax(scores, dim=-1)                  # (B, N)
        pooled  = (patch_tokens * weights.unsqueeze(-1)).sum(1)  # (B, D)

        out = self.regressor(pooled).squeeze(-1)             # (B,)
        return out.clamp(min=0.0)                            # measurements ≥ 0
