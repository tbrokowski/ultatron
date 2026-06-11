"""
models/heads/temporal_pool.py  ·  Temporal Attention Pooling
=============================================================

Aggregates per-frame CLS token embeddings from image-only encoders into a
single video-level representation.

Used as the ``encode_video`` bridge for any BackboneEncoder that lacks native
video support (ResNet50, ViT, BioMed-CLIP, USFM image-only variants, etc.).

Design
------
Implements scaled dot-product cross-attention with a single learnable query
vector (one-head cross-attention where Q is a trainable parameter and K/V come
from per-frame embeddings).

    frame_tokens : (B, T, D)   — per-frame CLS embeddings from frozen backbone
    query        : (1, 1, D)   — learned video-level query

    clip_cls     : (B, D)      — attended video representation

This module is trained **jointly with the downstream task head** while the
upstream backbone remains frozen.  Its parameters are exposed via the parent
BackboneEncoder.trainable_parameters() and added to the task optimiser.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TemporalAttentionPool(nn.Module):
    """
    Aggregate (B, T, D) frame embeddings → (B, D) via learned cross-attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the frame embeddings (= backbone embed_dim).
    num_heads : int
        Number of attention heads in the MHA layer. Defaults to 8; silently
        clamped to the largest divisor of embed_dim that is ≤ 8.
    dropout : float
        Dropout applied inside MultiheadAttention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Clamp num_heads to a valid divisor of embed_dim
        while num_heads > 1 and embed_dim % num_heads != 0:
            num_heads -= 1

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Learnable video-level query (B-independent)
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        frame_tokens: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        frame_tokens : (B, T, D)
            Per-frame CLS embeddings.
        key_padding_mask : (B, T) bool or None
            True where frames should be *ignored* (padding).

        Returns
        -------
        clip_cls : (B, D)
        """
        B = frame_tokens.size(0)
        q = self.query.expand(B, -1, -1)          # (B, 1, D)
        out, _ = self.attn(
            q,
            frame_tokens,
            frame_tokens,
            key_padding_mask=key_padding_mask,
        )                                          # (B, 1, D)
        return self.norm(out.squeeze(1))           # (B, D)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"
