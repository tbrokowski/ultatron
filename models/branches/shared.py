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


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, momentum: float):
    """
    In-place EMA: teacher_p ← momentum·teacher_p + (1−momentum)·student_p
    Operates on .data directly — no gradient graph, no optimizer.
    Only updates parameters (not buffers like BN running stats).
    """
    for s_p, t_p in zip(student.parameters_for_ema(),
                        teacher.parameters_for_ema()):
        t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)


class CrossBranchDistillation(nn.Module):
    """
    Aligns image teacher patch tokens with video student tube tokens.
    Linear projection from each space into a shared alignment space,
    followed by cosine distance loss.
    """

    def __init__(self, img_dim: int = 1024, vid_dim: int = 1024,
                 align_dim: int = 512):
        super().__init__()
        self.proj_img = nn.Linear(img_dim, align_dim, bias=False)
        self.proj_vid = nn.Linear(vid_dim, align_dim, bias=False)

    def forward(
        self,
        img_teacher_patches: torch.Tensor,   # (B_img, N, D_img)
        vid_student_tubes: torch.Tensor,     # (B_vid, M, D_vid)
    ) -> torch.Tensor:
        img_feat = F.normalize(self.proj_img(img_teacher_patches.mean(1)), dim=-1)
        vid_feat = F.normalize(self.proj_vid(vid_student_tubes.mean(1)),   dim=-1)
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
