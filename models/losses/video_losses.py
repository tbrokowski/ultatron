"""
models/losses/video_losses.py  ·  Video-branch SSL losses
==============================================================

Two loss functions for the video branch:

jepa_tube_loss
--------------
V-JEPA tube prediction loss.
L_tube = mean(1 - cos(predicted_tokens, teacher_tokens))
Computed only at positions that are (a) real frames, (b) real patches
(non-padding), and (c) masked in the tube.

clip_cls_loss
-------------
Clip-level temporal representation alignment.
L_clip = mean(1 - cos(student_clip_cls, teacher_clip_cls))
The student clip representation must match the teacher's on the
full unmasked clip.  This prevents the student from ignoring temporal
context and simply predicting from the visible frames.

Both are pure functions — no nn.Module state.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def jepa_tube_loss(
    predicted: Tensor,      # (B, T*ph*pw, D) — predictor output
    teacher_tokens: Tensor, # (B, T*ph*pw, D) — teacher tube tokens
    active_mask: Tensor,    # (B, T*ph*pw) bool — real AND tube-masked
) -> Tensor:
    """
    V-JEPA tube prediction loss.

    Computes cosine distance only at active (real + masked) positions.
    If no active positions, returns 0.

    Parameters
    ----------
    predicted     : student predictor output at all spatiotemporal positions
    teacher_tokens: teacher features at corresponding positions
    active_mask   : True = compute loss here
    """
    if not active_mask.any():
        return predicted.new_tensor(0.0)

    flat_p = predicted.reshape(-1, predicted.shape[-1])
    flat_t = teacher_tokens.reshape(-1, teacher_tokens.shape[-1])
    flat_m = active_mask.flatten()

    p_sel = F.normalize(flat_p[flat_m].float(), dim=-1)
    t_sel = F.normalize(flat_t[flat_m].float(), dim=-1)

    return (1.0 - (p_sel * t_sel).sum(dim=-1)).mean()


def clip_cls_loss(
    student_cls: Tensor,    # (B, D) — student mean-pooled clip rep
    teacher_cls: Tensor,    # (B, D) — teacher mean-pooled clip rep
) -> Tensor:
    """
    Clip-level temporal alignment loss.
    Cosine distance between student and teacher clip representations.
    """
    s = F.normalize(student_cls.float(), dim=-1)
    t = F.normalize(teacher_cls.float(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


def video_ssl_loss(
    student_clip_cls: Tensor,
    teacher_clip_cls: Tensor,
    predicted: Tensor,
    teacher_tokens: Tensor,
    active_mask: Tensor,
    lam_cls: float = 1.0,
    lam_tube: float = 1.0,
) -> dict:
    """
    Combined video SSL loss.

    Returns a dict with individual loss components and the weighted total,
    for easy logging and ablation.
    """
    l_cls  = clip_cls_loss(student_clip_cls, teacher_clip_cls)
    l_tube = jepa_tube_loss(predicted, teacher_tokens, active_mask)
    total  = lam_cls * l_cls + lam_tube * l_tube
    return {
        "loss_vid_total": total,
        "loss_cls":        l_cls,
        "loss_tube":       l_tube,
    }
