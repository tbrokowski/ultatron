"""
models/heads/mil_head.py  ·  Gated Attention MIL pooling + patient classifier
===============================================================================

GatedAttentionMILPool implements Ilse et al. (2018) attention-based MIL aggregation.
PatientMILClsHead chains the pool with an MLPClsHead for patient-level binary CLS.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.heads.classification_head import MLPClsHead

MIL_HIDDEN_DIM = 256


class GatedAttentionMILPool(nn.Module):
    """
    Gated Attention Multiple Instance Learning (Ilse et al., 2018).

    For a bag of N instance embeddings H:

        e_i = tanh(W_V · h_i)
        g_i = sigmoid(W_U · h_i)
        a_i = softmax(w^T (e_i ⊙ g_i))
        z   = Σ a_i · h_i

    Supports:
      - Single bag:  H (N, D)  → z (D,)
      - Batched:     H (B, N, D), mask (B, N)  → z (B, D)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = MIL_HIDDEN_DIM):
        super().__init__()
        self.attn_V = nn.Linear(embed_dim, hidden_dim)
        self.attn_U = nn.Linear(embed_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1, bias=False)

    def _attention_weights(self, H: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Return unnormalised attention logits with shape (..., N, 1)."""
        gated = torch.tanh(self.attn_V(H)) * torch.sigmoid(self.attn_U(H))
        logits = self.attn_w(gated)
        if mask is not None:
            while mask.dim() < logits.dim():
                mask = mask.unsqueeze(-1)
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def forward(self, H: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if H.dim() == 2:
            A = self._attention_weights(H.unsqueeze(0), mask).squeeze(0)  # (N, 1)
            A = torch.softmax(A, dim=0)
            return (A * H).sum(dim=0)

        A = self._attention_weights(H, mask)          # (B, N, 1)
        A = torch.softmax(A, dim=1)
        return (A * H).sum(dim=1)                       # (B, D)

    def attention_weights(self, H: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Return normalised attention weights over instances."""
        if H.dim() == 2:
            A = self._attention_weights(H.unsqueeze(0), mask).squeeze(0)
            return torch.softmax(A, dim=0).squeeze(-1)
        A = self._attention_weights(H, mask)
        return torch.softmax(A, dim=1).squeeze(-1)


class PatientMILClsHead(nn.Module):
    """GatedAttentionMILPool → MLPClsHead for patient-level binary classification."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = MIL_HIDDEN_DIM,
        mlp_hidden_dim: Optional[int] = None,
        n_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = GatedAttentionMILPool(embed_dim, hidden_dim=hidden_dim)
        mlp_hidden = mlp_hidden_dim or embed_dim // 2
        self.classifier = MLPClsHead(
            embed_dim=embed_dim,
            n_classes=n_classes,
            hidden_dim=mlp_hidden,
            dropout=dropout,
        )

    def forward(self, H: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        z = self.pool(H, mask=mask)
        if z.dim() == 1:
            return self.classifier(z.unsqueeze(0)).squeeze(0)
        return self.classifier(z)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        if targets.dim() == 1:
            targets = targets.unsqueeze(-1)
        return F.binary_cross_entropy_with_logits(logits, targets.float())

    def attention_weights(self, H: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.pool.attention_weights(H, mask=mask)

    def __repr__(self) -> str:
        return (
            f"PatientMILClsHead(embed_dim={self.pool.attn_V.in_features}, "
            f"mil_hidden={self.pool.attn_V.out_features}, "
            f"n={self.classifier.fc2.out_features})"
        )
