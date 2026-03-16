"""
models/heads/concept_detection_head.py  ·  Concept detection head
=======================================================================

Anatomy concept detection: given a set of K named concepts
(e.g. "pleural effusion", "mitral valve leaflet", "thyroid nodule"),
predict a binary presence score for each concept.

This is NOT standard object detection (no bounding boxes).  It answers:
"Does this image/frame contain this anatomical concept?" — a multilabel
binary classification over a fixed vocabulary.

Architecture
------------
ConceptDetectionHead uses attentive pooling independently for each concept,
implemented efficiently as a single multi-head attention where each head
corresponds to one concept.

Input:  patch_tokens (B, N, D)
Output: (B, K) logits  — K = number of concepts

Each concept has its own learnable query vector that attends over all
patch tokens.  The attended features are projected to a scalar logit.

Loss: binary_cross_entropy_with_logits per concept (multilabel).

Usage in Phase 4
----------------
This head is used as a lightweight probe to verify the backbone has
encoded interpretable anatomy-level features before running full
segmentation.  It can also be used in the agent loop to decide which
downstream segmentation heads to invoke.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptDetectionHead(nn.Module):
    """
    Multi-concept binary detection via per-concept cross-attention.

    Parameters
    ----------
    embed_dim    : int   backbone hidden dimension
    n_concepts   : int   number of binary concept outputs
    concept_names: list  optional list of concept name strings (for logging only)
    attn_heads   : int   number of parallel attention heads per concept
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        n_concepts: int = 64,
        concept_names: Optional[List[str]] = None,
        attn_heads: int = 1,
    ):
        super().__init__()
        self.embed_dim    = embed_dim
        self.n_concepts   = n_concepts
        self.concept_names = concept_names or [f"concept_{i}" for i in range(n_concepts)]

        # One learnable query per concept
        # Shape: (n_concepts, embed_dim)
        self.concept_queries = nn.Parameter(
            torch.randn(n_concepts, embed_dim) * (embed_dim ** -0.5)
        )

        # Per-concept linear projection for keys and values
        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.val_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final per-concept logit projection: embed_dim → 1
        self.score_proj = nn.Linear(embed_dim, 1, bias=True)

        self.scale = embed_dim ** -0.5

    def forward(
        self,
        patch_tokens: torch.Tensor,              # (B, N, D)
        padding_mask: Optional[torch.Tensor] = None,  # (B, ph, pw)
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (B, K)  unnormalised concept detection scores
        """
        B, N, D = patch_tokens.shape

        # Keys and values from patch tokens
        K = self.key_proj(patch_tokens)          # (B, N, D)
        V = self.val_proj(patch_tokens)          # (B, N, D)

        # Queries: (n_concepts, D) → (1, n_concepts, D) → (B, n_concepts, D)
        Q = self.concept_queries.unsqueeze(0).expand(B, -1, -1)  # (B, K_c, D)

        # Scaled dot-product attention: (B, K_c, N)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale   # (B, K_c, N)

        if padding_mask is not None:
            flat_mask = padding_mask.flatten(1)                # (B, N)
            # Broadcast to (B, K_c, N)
            attn = attn.masked_fill(
                ~flat_mask.unsqueeze(1).expand_as(attn), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)                         # (B, K_c, N)

        # Attended features: (B, K_c, D)
        attended = torch.bmm(attn, V)

        # Per-concept logit: (B, K_c, D) → (B, K_c, 1) → (B, K_c)
        logits = self.score_proj(attended).squeeze(-1)

        return logits                                          # (B, K_c)

    def concept_name(self, idx: int) -> str:
        return self.concept_names[idx]

    def __repr__(self):
        return (f"ConceptDetectionHead(D={self.embed_dim}, "
                f"K={self.n_concepts})")
