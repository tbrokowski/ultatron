"""
models/branches/shared.py  ·  Shared branch utilities
======================================================
EMA update, cross-branch distillation, prototype head.
These are independent of which backbone is used.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _iter_params(model: nn.Module):
    """
    Yield trainable parameters for EMA.
    Uses .parameters_for_ema() if available (our BaseModel subclasses can
    exclude frozen adapter layers), otherwise falls back to .parameters().
    """
    if hasattr(model, "parameters_for_ema"):
        yield from model.parameters_for_ema()
    else:
        yield from model.parameters()


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float):
    """
    In-place EMA: teacher_p ← momentum·teacher_p + (1−momentum)·student_p
    Operates on .data directly — no gradient graph, no optimizer.
    Only updates parameters (not buffers like BN running stats).
    """
    for s_p, t_p in zip(_iter_params(student), _iter_params(teacher)):
        t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)


class CrossBranchDistillation(nn.Module):
    """
    Aligns image teacher patch tokens with video student tube tokens.

    Two linear projections map each modality into a shared alignment space.
    A BYOL-style asymmetric predictor is applied to the video side only —
    this allows the video student to "translate" toward the image teacher's
    representation without requiring the image side to regress toward noisy
    video features, preventing collapse without explicit negatives.

    Projections are used by both the FILIP per-token loss and the SwAV
    prototype loss in phase3_step (train/phase_steps.py).
    """

    def __init__(self, img_dim: int = 1024, vid_dim: int = 1024,
                 align_dim: int = 512):
        super().__init__()
        self.align_dim = align_dim
        self.proj_img  = nn.Linear(img_dim, align_dim, bias=False)
        self.proj_vid  = nn.Linear(vid_dim, align_dim, bias=False)
        # BYOL-style predictor on video side only (asymmetric).
        # Applied after proj_vid before computing cross-modal loss.
        # The image teacher side receives no gradient from this head.
        self.predictor_vid = nn.Sequential(
            nn.Linear(align_dim, align_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(align_dim * 4, align_dim, bias=False),
        )

    def forward(
        self,
        img_teacher_patches: torch.Tensor,   # (B_img, N, D_img)
        vid_student_tubes: torch.Tensor,     # (B_vid, M, D_vid)
    ) -> torch.Tensor:
        """Global mean-pool cosine loss (legacy path; per-pair FILIP preferred)."""
        img_feat = F.normalize(self.proj_img(img_teacher_patches.mean(1)), dim=-1)
        vid_proj = self.proj_vid(vid_student_tubes.mean(1))
        vid_feat = F.normalize(self.predictor_vid(vid_proj), dim=-1)
        B = min(img_feat.shape[0], vid_feat.shape[0])
        return (1 - (img_feat[:B] * vid_feat[:B]).sum(-1)).mean()


class PrototypeHead(nn.Module):
    """
    DINO-style prototype consistency loss.
    K learnable prototypes; soft assignment via cosine similarity.
    Loss: symmetric cross-entropy between image and video prototype distributions.
    """

    def __init__(self, embed_dim: int = 1024, n_prototypes: int = 256,
                 temperature: float = 0.1):
        super().__init__()
        self.prototypes  = nn.Parameter(
            F.normalize(torch.randn(n_prototypes, embed_dim), dim=-1)
        )
        self.temperature = temperature

    def _assign(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D) → soft assignment (B, K)
        feat  = F.normalize(tokens.mean(1), dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        return F.softmax((feat @ proto.T) / self.temperature, dim=-1)

    def consistency_loss(
        self,
        img_tokens: torch.Tensor,
        vid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        p_img = self._assign(img_tokens)
        p_vid = self._assign(vid_tokens)
        B = min(p_img.shape[0], p_vid.shape[0])
        loss = (
            -(p_img[:B] * (p_vid[:B] + 1e-8).log()).sum(-1).mean()
            - (p_vid[:B] * (p_img[:B] + 1e-8).log()).sum(-1).mean()
        ) / 2.0
        return loss
