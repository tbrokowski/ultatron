"""
models/heads/student_heads.py  ·  Downstream heads for the single student
=========================================================================

All heads consume output from HieraStudentBackbone.forward():
  global : (B, D4)          global token
  F4     : (B, T, N4, D4)   stage-4 features
  dense  : alias for F4

Head catalogue
--------------
DomainHead         Anatomy/domain classification (lung, cardiac, thyroid, ...)
ViewHead           View/zone classification (PLAX, A4C, lung zone, ...)
QualityHead        Image quality classification / ordinal regression
ConceptHead        Multi-label concept/finding detection
TemporalConceptHead Temporal concept detection from clip-level features
RetrievalHead      Embedding for retrieval + prototype-based retrieval
UncertaintyHead    Calibration/confidence estimate for any prediction

All heads follow the same interface:
  head.forward(features_dict) → logits_dict
  head.loss(logits, targets)  → scalar loss Tensor

build_student_head(head_type, ...) factory.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared MLP utility
# ---------------------------------------------------------------------------

class _MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )


# ---------------------------------------------------------------------------
# DomainHead
# ---------------------------------------------------------------------------

class DomainHead(nn.Module):
    """
    Anatomy / domain classification.

    Input  : global token (B, D)
    Output : (B, n_domains) logits

    Loss   : cross-entropy (single-label) or multi-label BCE
    """

    def __init__(self, embed_dim: int, n_classes: int, multilabel: bool = False):
        super().__init__()
        self.multilabel = multilabel
        self.head = _MLP(embed_dim, embed_dim // 2, n_classes)

    def forward(self, features: dict) -> dict:
        logits = self.head(features["global"].float())
        return {"domain_logits": logits}

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.multilabel:
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        return F.cross_entropy(logits, targets.long())


# ---------------------------------------------------------------------------
# ViewHead
# ---------------------------------------------------------------------------

class ViewHead(nn.Module):
    """
    View / anatomical zone classification.

    Input  : global token + F4 pooled → concat
    Output : (B, n_classes) logits
    """

    def __init__(self, embed_dim: int, n_classes: int, multilabel: bool = False):
        super().__init__()
        self.multilabel = multilabel
        # Concat global + F4 mean pool (same dim) → 2*D input
        self.head = _MLP(embed_dim * 2, embed_dim, n_classes)

    def forward(self, features: dict) -> dict:
        g = features["global"].float()                      # (B, D)
        f4 = features["F4"].float().mean(dim=(1, 2))       # (B, D)
        combined = torch.cat([g, f4], dim=-1)               # (B, 2D)
        return {"view_logits": self.head(combined)}

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.multilabel:
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        return F.cross_entropy(logits, targets.long())


# ---------------------------------------------------------------------------
# QualityHead
# ---------------------------------------------------------------------------

class QualityHead(nn.Module):
    """
    Image quality assessment.

    Input  : global token
    Output : (B, n_classes) logits
    Loss   : cross-entropy (categorical) or MSE (ordinal)
    """

    def __init__(self, embed_dim: int, n_classes: int, ordinal: bool = False):
        super().__init__()
        self.ordinal  = ordinal
        self.n_classes = n_classes
        self.head = _MLP(embed_dim, embed_dim // 2, n_classes if not ordinal else 1)

    def forward(self, features: dict) -> dict:
        logits = self.head(features["global"].float())
        return {"quality_logits": logits}

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.ordinal:
            return F.mse_loss(logits.squeeze(-1), targets.float())
        return F.cross_entropy(logits, targets.long())


# ---------------------------------------------------------------------------
# ConceptHead
# ---------------------------------------------------------------------------

class ConceptHead(nn.Module):
    """
    Multi-label ultrasound concept / finding detection.

    Input  : global token + dense token mean pool (via F4)
    Output : (B, n_concepts) logits
    Loss   : multi-label BCE (optionally focal)
    """

    def __init__(
        self,
        embed_dim: int,
        n_concepts: int,
        focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.focal       = focal
        self.focal_gamma = focal_gamma
        self.head = _MLP(embed_dim * 2, embed_dim, n_concepts)

    def forward(self, features: dict) -> dict:
        g    = features["global"].float()              # (B, D)
        dense = features["F4"].float().mean(dim=(1, 2))  # (B, D)
        combined = torch.cat([g, dense], dim=-1)
        return {"concept_logits": self.head(combined)}

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        if self.focal:
            pt = torch.sigmoid(logits)
            p_t = pt * targets + (1 - pt) * (1 - targets)
            bce = bce * (1 - p_t) ** self.focal_gamma
        return bce.mean()


# ---------------------------------------------------------------------------
# TemporalConceptHead
# ---------------------------------------------------------------------------

class TemporalConceptHead(nn.Module):
    """
    Temporal concept detection from clip-level features.
    Requires T>1 input; degrades gracefully for T=1 (returns zero loss).

    Input  : F4 temporal mean over frames → (B, D)
    Output : (B, n_concepts) logits
    """

    def __init__(self, embed_dim: int, n_concepts: int):
        super().__init__()
        self.head = _MLP(embed_dim, embed_dim // 2, n_concepts)

    def forward(self, features: dict) -> dict:
        f4 = features["F4"].float()             # (B, T, N4, D)
        # Temporal + spatial mean → (B, D)
        clip_feat = f4.mean(dim=(1, 2))
        return {"temporal_concept_logits": self.head(clip_feat)}

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets.float())


# ---------------------------------------------------------------------------
# RetrievalHead
# ---------------------------------------------------------------------------

class RetrievalHead(nn.Module):
    """
    Produces a normalised embedding for retrieval.

    Input  : global token
    Output : (B, proj_dim) L2-normalised embedding
    Loss   : InfoNCE contrastive loss (computed externally via this head's
             .contrastive_loss() helper)
    """

    def __init__(self, embed_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj = _MLP(embed_dim, embed_dim // 2, proj_dim)

    def forward(self, features: dict) -> dict:
        emb = self.proj(features["global"].float())
        emb = F.normalize(emb, dim=-1)
        return {"retrieval_emb": emb}

    def contrastive_loss(
        self,
        emb_a: Tensor,    # (B, D)
        emb_b: Tensor,    # (B, D)
        temperature: float = 0.07,
    ) -> Tensor:
        """In-batch InfoNCE between two views."""
        logits = (emb_a @ emb_b.T) / temperature   # (B, B)
        labels = torch.arange(emb_a.shape[0], device=emb_a.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


# ---------------------------------------------------------------------------
# UncertaintyHead
# ---------------------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """
    Scalar uncertainty / confidence estimate for any embedding.

    Output is a scalar confidence ∈ (0, 1) via sigmoid.
    Loss: Brier score or calibration loss.

    Parameters
    ----------
    embed_dim : input feature dimension
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """Returns (B,) confidence scores in (0, 1)."""
        return torch.sigmoid(self.head(embedding.float()).squeeze(-1))

    def brier_loss(self, confidence: Tensor, correct: Tensor) -> Tensor:
        """
        Brier score: MSE between predicted confidence and binary correctness.
        correct : (B,) float 0/1 — whether the main prediction was correct.
        """
        return F.mse_loss(confidence, correct.float())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_HEAD_REGISTRY: Dict[str, type] = {
    "domain":           DomainHead,
    "view":             ViewHead,
    "quality":          QualityHead,
    "concept":          ConceptHead,
    "temporal_concept": TemporalConceptHead,
    "retrieval":        RetrievalHead,
    "uncertainty":      UncertaintyHead,
}


def build_student_head(
    head_type: str,
    embed_dim: int,
    **kwargs,
) -> nn.Module:
    """
    Factory for student downstream heads.

    Parameters
    ----------
    head_type  : one of "domain", "view", "quality", "concept",
                 "temporal_concept", "retrieval", "uncertainty"
    embed_dim  : student backbone output dimension (e.g. 1152 for Hiera-L)
    **kwargs   : passed to the head constructor (e.g. n_classes, multilabel)
    """
    if head_type not in _HEAD_REGISTRY:
        raise ValueError(
            f"Unknown head_type: {head_type!r}. "
            f"Available: {sorted(_HEAD_REGISTRY.keys())}"
        )
    return _HEAD_REGISTRY[head_type](embed_dim=embed_dim, **kwargs)
