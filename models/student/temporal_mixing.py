"""
models/student/temporal_mixing.py  ·  Temporal adapter modules
==============================================================

Two lightweight temporal mixing modules that can be injected between
Hiera stages.  Both are identity no-ops when T=1, so image batches
pass through without any overhead or change to the representation.

FactorizedTemporalAttention
----------------------------
Reshape (B, T, N, D) → (B*N, T, D), apply standard MHA across T,
reshape back.  Output projection zero-initialized so the module starts
as an identity residual.  Gradually opens as training proceeds.

TemporalDepthwiseConv
---------------------
Lightweight fallback: 1D depthwise convolution along the T dimension.
Zero-initialized output.  Cheaper than attention for long T.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FactorizedTemporalAttention(nn.Module):
    """
    Factorized temporal self-attention over T frames at each spatial position.

    For T=1 the forward short-circuits and returns x unchanged (no compute,
    no grad through the attention path).

    Parameters
    ----------
    dim       : token dimension D
    num_heads : attention heads (must divide D evenly)
    zero_init : if True, output projection is zero-initialized → identity
                residual at t=0; the module gradually "opens" during training.
    dropout   : attention dropout probability
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        zero_init: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim       = dim
        self.num_heads = num_heads

        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        if zero_init:
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(
        self,
        x: Tensor,                           # (B, T, N, D)
        padding_mask: Optional[Tensor] = None,  # (B, N) bool — True = real
    ) -> Tensor:
        B, T, N, D = x.shape

        if T == 1:
            return x

        # Reshape: treat each spatial position as an independent sequence of T
        # tokens.  (B, T, N, D) → (B*N, T, D)
        x_norm = self.norm(x)
        xr = x_norm.permute(0, 2, 1, 3).reshape(B * N, T, D)

        # key_padding_mask for MHA: True = IGNORE.  We have no per-temporal mask
        # so we pass None.  padding_mask here is spatial (which positions are
        # real), not temporal; it's used to zero out output at padded positions.
        out, _ = self.attn(xr, xr, xr)           # (B*N, T, D)
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)  # (B, T, N, D)

        result = x + out

        # Zero out padding positions so they don't pollute later stages
        if padding_mask is not None:
            # padding_mask: (B, N) bool True=real → expand to (B, T, N, 1)
            mask = padding_mask.unsqueeze(1).unsqueeze(-1).float()
            result = result * mask

        return result

    def __repr__(self) -> str:
        return f"FactorizedTemporalAttention(dim={self.dim}, heads={self.num_heads})"


class TemporalDepthwiseConv(nn.Module):
    """
    Lightweight temporal mixing via 1D depthwise convolution along T.

    Cheaper than attention for long clips.  Acts as a local smoothing
    over adjacent frames.  Zero-initialized output — starts as identity.

    Parameters
    ----------
    dim    : feature dimension D
    kernel : temporal kernel size (odd, e.g. 3 or 5)
    """

    def __init__(self, dim: int, kernel: int = 3, zero_init: bool = True):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd"
        self.dim    = dim
        self.kernel = kernel

        self.norm    = nn.LayerNorm(dim)
        self.dw_conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel,
            padding=kernel // 2,
            groups=dim,
            bias=False,
        )
        self.proj = nn.Linear(dim, dim, bias=True)

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: Tensor,                           # (B, T, N, D)
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, N, D = x.shape

        if T == 1:
            return x

        x_norm = self.norm(x)
        # (B, T, N, D) → (B*N, D, T) for Conv1d
        xr = x_norm.permute(0, 2, 3, 1).reshape(B * N, D, T)
        out = self.dw_conv(xr)               # (B*N, D, T)
        out = out.reshape(B, N, D, T).permute(0, 3, 1, 2)  # (B, T, N, D)
        out = self.proj(out)

        result = x + out

        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).unsqueeze(-1).float()
            result = result * mask

        return result

    def __repr__(self) -> str:
        return f"TemporalDepthwiseConv(dim={self.dim}, kernel={self.kernel})"


def build_temporal_mixer(
    mixing_type: str,
    dim: int,
    num_heads: int = 8,
    zero_init: bool = True,
) -> nn.Module:
    """
    Factory for temporal mixing modules.

    Parameters
    ----------
    mixing_type : "factorized_attn" | "depthwise_conv"
    """
    if mixing_type == "factorized_attn":
        return FactorizedTemporalAttention(dim, num_heads=num_heads, zero_init=zero_init)
    elif mixing_type == "depthwise_conv":
        return TemporalDepthwiseConv(dim, zero_init=zero_init)
    else:
        raise ValueError(
            f"Unknown temporal_mixing: {mixing_type!r}. "
            "Choose 'factorized_attn' or 'depthwise_conv'."
        )
