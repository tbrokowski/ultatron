"""
models/branches/image_branch.py  · Image branch
================================================================

ImageBranch holds a student + EMA teacher pair and an optional frozen
large-scale distillation teacher.  

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
    student   : ImageBackboneBase  (trainable)
    teacher   : ImageBackboneBase  (EMA copy, frozen grad)
    teacher_d : FrozenTeacherBase  (optional large-scale distillation teacher)
    proj_d    : nn.Linear          (projects teacher_d output → student dim)
    embed_dim : int                (student hidden_size)
    """

    def __init__(
        self,
        student: ImageBackboneBase,
        teacher: ImageBackboneBase,
        teacher_d: Optional[FrozenTeacherBase] = None,
    ):
        super().__init__()
        self.student   = student
        self.teacher   = teacher
        self.teacher_d = teacher_d
        self.embed_dim = student.hidden_size

        # Projection head: distillation teacher dim → student dim
        if teacher_d is not None:
            self.proj_d = nn.Linear(teacher_d.hidden_size, student.hidden_size,
                                    bias=False)
        else:
            self.proj_d = None

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
        """Run the distillation teacher (e.g. DINOv3-7B). Returns None if absent."""
        if self.teacher_d is None:
            return None
        return self.teacher_d(pixel_values, **kwargs)
