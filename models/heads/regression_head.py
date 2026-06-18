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

from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.heads.temporal_pool import TemporalAttentionPool


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


class VideoRegressionHead(nn.Module):
    """
    Video regression head for EchoNet-style EF and related cardiac measurements.

    Follows the EchoNet-Dynamic protocol (Ouyang et al.): a full cine clip is
    reduced to a scalar (or vector) target.  With frozen backbones we adapt the
    original CNN+LSTM design as:

      video_native=True  : deep MLP on ``clip_cls`` (backbone already fused T)
      video_native=False : trainable TemporalAttentionPool over per-frame CLS
                           embeddings, then deep MLP (LSTM → attention analogue)

    Parameters
    ----------
    embed_dim : int
        Backbone embedding dimension.
    hidden_dim : int
        MLP hidden width (512 matches OpenUS LVEF head and EchoNet follow-ups).
    n_layers : int
        Number of MLP hidden blocks (default 3).
    dropout : float
    n_outputs : int
        1 for EF (%), >1 for multi-target wall thickness, etc.
    video_native : bool
        True when ``encode_video`` already returns a temporally fused ``clip_cls``.
    output_min, output_max : float or None
        Inference clamp for scalar outputs (n_outputs == 1 only).
    """

    def __init__(
        self,
        embed_dim:    int   = 1024,
        hidden_dim:   int   = 512,
        n_layers:     int   = 3,
        dropout:      float = 0.2,
        n_outputs:    int   = 1,
        video_native: bool  = True,
        output_min:   Optional[float] = 10.0,
        output_max:   Optional[float] = 85.0,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.n_outputs    = n_outputs
        self.video_native = video_native
        self.output_min   = output_min
        self.output_max   = output_max

        self.frame_pool: Optional[TemporalAttentionPool] = None
        if not video_native:
            self.frame_pool = TemporalAttentionPool(embed_dim, dropout=dropout)

        layers: list[nn.Module] = [nn.LayerNorm(embed_dim)]
        in_dim = embed_dim
        for _ in range(max(n_layers - 1, 0)):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        self.out_proj = nn.Linear(in_dim, n_outputs)
        if n_outputs == 1 and output_min is not None and output_max is not None:
            nn.init.constant_(self.out_proj.bias, (output_min + output_max) / 2.0)
        self.mlp = nn.Sequential(*layers)

    def _pool(self, clip_cls: Optional[Tensor], frame_tokens: Optional[Tensor]) -> Tensor:
        if self.video_native:
            if clip_cls is None:
                raise ValueError("video_native=True requires clip_cls")
            return clip_cls
        if frame_tokens is None:
            raise ValueError("video_native=False requires frame_tokens (B, T, D)")
        assert self.frame_pool is not None
        return self.frame_pool(frame_tokens)

    def forward(
        self,
        clip_cls:      Optional[Tensor] = None,
        frame_tokens:  Optional[Tensor] = None,
    ) -> Tensor:
        x   = self._pool(clip_cls, frame_tokens)
        x   = self.mlp(x)
        out = self.out_proj(x)
        if self.n_outputs == 1:
            out = out.squeeze(-1)
            if not self.training and (self.output_min is not None or self.output_max is not None):
                out = torch.clamp(out, self.output_min, self.output_max)
            return out
        return out

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.mlp.parameters()
        yield from self.out_proj.parameters()
        if self.frame_pool is not None:
            yield from self.frame_pool.parameters()

    def __repr__(self) -> str:
        mode = "video_native" if self.video_native else "frame_attention"
        return (
            f"VideoRegressionHead(D={self.embed_dim}, hidden={self.out_proj.in_features}, "
            f"n_out={self.n_outputs}, mode={mode})"
        )


def build_video_regression_head(
    head_type:    str,
    embed_dim:    int,
    video_native: bool,
    *,
    n_outputs:    int   = 1,
    hidden_dim:   int   = 512,
    dropout:      float = 0.2,
    output_min:   Optional[float] = 10.0,
    output_max:   Optional[float] = 85.0,
) -> nn.Module:
    """
    Factory for video regression heads.

    head_type
    ---------
    ``linear`` : shallow RegressionHead (legacy baseline).
    ``mlp``    : VideoRegressionHead with frame attention for image-only FMs.
    """
    if head_type == "linear":
        if n_outputs == 1:
            return RegressionHead(
                embed_dim=embed_dim,
                hidden_dim=256,
                output_min=output_min,
                output_max=output_max,
            )
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_outputs),
        )

    return VideoRegressionHead(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=3,
        dropout=dropout,
        n_outputs=n_outputs,
        video_native=video_native,
        output_min=output_min if n_outputs == 1 else None,
        output_max=output_max if n_outputs == 1 else None,
    )


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
