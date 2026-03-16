"""
models/heads/classification_head.py  ·  Classification heads
==================================================================

Three variants:

LinearClsHead
-------------
Single linear layer on CLS token → logits.
The standard linear probe for SSL evaluation.
Input:  cls (B, D)
Output: (B, n_classes)

MLPClsHead
----------
Two-layer MLP with GELU, LayerNorm, and optional dropout.
Used for Phase 4 supervised fine-tuning when a linear head underfits.
Input:  cls (B, D)
Output: (B, n_classes)

AttentivePoolClsHead
--------------------
Attentive pooling over patch tokens followed by a linear head.
Learns *which* patches to attend to for classification — useful for
anatomy detection where the relevant structure is localised.
Input:  patch_tokens (B, N, D)
Output: (B, n_classes)

All heads are compatible with multiclass (softmax + CE) and multilabel
(sigmoid + BCE) tasks — the task type is determined by the LossType in
the calling code, not by the head itself.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClsHead(nn.Module):
    """Single linear classifier on CLS token."""

    def __init__(self, embed_dim: int = 1024, n_classes: int = 256):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        # cls: (B, D) → (B, n_classes)
        return self.fc(cls)

    def __repr__(self):
        return f"LinearClsHead(D={self.fc.in_features}, n={self.fc.out_features})"


class MLPClsHead(nn.Module):
    """
    Two-layer MLP: D → hidden → n_classes.
    LayerNorm before projection, GELU activation, optional dropout.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        n_classes: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm  = nn.LayerNorm(embed_dim)
        self.fc1   = nn.Linear(embed_dim, hidden_dim)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.fc2   = nn.Linear(hidden_dim, n_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        x = self.norm(cls)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)

    def __repr__(self):
        return (f"MLPClsHead(D={self.fc1.in_features}, "
                f"hidden={self.fc1.out_features}, n={self.fc2.out_features})")


class AttentivePoolClsHead(nn.Module):
    """
    Attentive pooling over patch tokens.

    A small attention network produces per-patch weights; the weighted
    sum of patch tokens is passed to a linear classifier.

    Useful when the relevant anatomy is localised (e.g. thyroid nodule,
    breast mass) and the CLS token may not capture it sufficiently.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        n_classes: int = 256,
        attn_hidden: int = 128,
    ):
        super().__init__()
        # Attention scoring network: D → attn_hidden → 1
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1, bias=False),
        )
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(
        self,
        patch_tokens: torch.Tensor,              # (B, N, D)
        padding_mask: Optional[torch.Tensor] = None,  # (B, ph, pw)
    ) -> torch.Tensor:
        B, N, D = patch_tokens.shape

        # Compute attention logits
        attn_logits = self.attn(patch_tokens).squeeze(-1)  # (B, N)

        # Mask out padding positions before softmax
        if padding_mask is not None:
            flat_mask = padding_mask.flatten(1)            # (B, N)
            attn_logits = attn_logits.masked_fill(~flat_mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)      # (B, N)

        # Weighted sum
        pooled = (patch_tokens * attn_weights.unsqueeze(-1)).sum(1)  # (B, D)

        return self.fc(pooled)

    def __repr__(self):
        return (f"AttentivePoolClsHead(D={self.fc.in_features}, "
                f"n={self.fc.out_features})")


def build_cls_head(
    embed_dim: int,
    n_classes: int,
    head_type: str = "linear",
    **kwargs,
) -> nn.Module:
    """
    Factory for classification heads.

    Parameters
    ----------
    head_type : "linear" | "mlp" | "attentive_pool"
    """
    if head_type == "linear":
        return LinearClsHead(embed_dim, n_classes)
    elif head_type == "mlp":
        return MLPClsHead(embed_dim, n_classes, **kwargs)
    elif head_type == "attentive_pool":
        return AttentivePoolClsHead(embed_dim, n_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown head_type: {head_type!r}. "
            f"Choose 'linear', 'mlp', or 'attentive_pool'."
        )
