"""
models/branches/image_branch.py  · Image branch
================================================================

ImageBranch holds a student + EMA teacher pair and two optional frozen
large-scale distillation teachers:

  teacher_d  : DINOv3-7B or similar — CLS-level distillation (lam_7b)
  teacher_sam: SAM2/SAM3 Perception Encoder — patch-level RADIO-style
               distillation so student features are compatible with
               SAM's prompt encoder / mask decoder (lam_sam)

A pixel reconstruction head (recon_head) regresses the original pixel
intensities at masked patch positions, implementing the paper's
global-view mask reconstruction pre-training task (lam_recon).

The EMA update uses student.parameters_for_ema() and
teacher.parameters_for_ema() so backbones can exclude frozen adapter
layers from the EMA sweep if needed.
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

from models.base import ImageBackboneBase, FrozenTeacherBase
from models.branches.shared import ema_update

log = logging.getLogger(__name__)


class ImageBranch(nn.Module):
    """
    Image branch.

    Attributes
    ----------
    student        : ImageBackboneBase  (trainable)
    teacher        : ImageBackboneBase  (EMA copy, frozen grad)
    teacher_d      : FrozenTeacherBase  (optional DINOv3-7B distillation teacher)
    proj_d         : nn.Linear          (teacher_d dim → student dim, CLS)
    teacher_sam    : FrozenTeacherBase  (optional SAM2/SAM3 distillation teacher)
    proj_sam_cls   : nn.Linear          (SAM dim → student dim, CLS)
    proj_sam_patch : nn.Linear          (SAM dim → student dim, patch tokens)
    recon_head     : nn.Linear          (student dim → patch_size²×C, pixel recon)
    embed_dim      : int                (student hidden_size)
    patch_size     : int
    """

    def __init__(
        self,
        student: ImageBackboneBase,
        teacher: ImageBackboneBase,
        teacher_d: Optional[FrozenTeacherBase] = None,
        teacher_sam: Optional[FrozenTeacherBase] = None,
        patch_size: int = 16,
        recon_channels: int = 3,
    ):
        super().__init__()
        self.student    = student
        self.teacher    = teacher
        self.teacher_d  = teacher_d
        self.teacher_sam = teacher_sam
        self.embed_dim  = student.hidden_size
        self.patch_size = patch_size

        # DINOv3-7B CLS distillation projection
        if teacher_d is not None:
            self.proj_d = nn.Linear(teacher_d.hidden_size, student.hidden_size,
                                    bias=False)
        else:
            self.proj_d = None

        # SAM2/SAM3 patch-level RADIO distillation projections
        if teacher_sam is not None:
            self.proj_sam_cls   = nn.Linear(teacher_sam.hidden_size,
                                             student.hidden_size, bias=False)
            self.proj_sam_patch = nn.Linear(teacher_sam.hidden_size,
                                             student.hidden_size, bias=False)
        else:
            self.proj_sam_cls   = None
            self.proj_sam_patch = None

        # Pixel reconstruction head: student patch tokens → pixel values
        # Predicts patch_size² × C values per masked patch position.
        self.recon_head = nn.Linear(
            student.hidden_size,
            patch_size * patch_size * recon_channels,
            bias=True,
        )

        # Freeze EMA teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def update_teacher(self, momentum: float = 0.9995):
        """EMA-update teacher weights from student. No gradients."""
        ema_update(self.student, self.teacher, momentum)

    def forward_student(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        return self.student(pixel_values, padding_mask=padding_mask, **kwargs)

    @torch.no_grad()
    def forward_teacher(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        self.teacher.eval()
        return self.teacher(pixel_values, padding_mask=padding_mask, **kwargs)

    @torch.no_grad()
    def forward_teacher_d(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> Optional[dict]:
        """Run the DINOv3-7B distillation teacher. Returns None if absent."""
        if self.teacher_d is None:
            return None
        return self.teacher_d(pixel_values, **kwargs)

    @torch.no_grad()
    def forward_teacher_sam(
        self,
        pixel_values: torch.Tensor,
        target_ph: Optional[int] = None,
        target_pw: Optional[int] = None,
        **kwargs,
    ) -> Optional[dict]:
        """
        Run the SAM2/SAM3 frozen teacher.

        If target_ph/target_pw are provided, SAM's patch features are
        bilinearly interpolated to match the student's spatial grid,
        enabling direct per-position patch distillation.

        Returns None if teacher_sam is not configured.
        """
        if self.teacher_sam is None:
            return None
        return self.teacher_sam(
            pixel_values,
            target_ph=target_ph,
            target_pw=target_pw,
            **kwargs,
        )
