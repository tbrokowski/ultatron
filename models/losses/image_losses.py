"""
models/losses/image_losses.py  ·  Image-branch SSL losses
==============================================================

Three loss functions for the image branch:

dino_cls_loss
-------------
DINO CLS token cosine distance loss.
L_cls = mean(1 - cos(student_cls, teacher_cls))
Applied to global crops and local crops (with lower weight for local).

ibot_patch_loss
---------------
iBOT-style masked patch prediction loss.
L_patch = mean(1 - cos(student_patches, teacher_patches))
          summed only over positions that are (a) real content AND
          (b) frequency-masked (the masked student patch prediction task).

koleo_loss
----------
KoLeo uniformity regulariser (DINOv3).
Encourages features to be spread uniformly over the unit hypersphere.
L_KoLeo = -log(min nearest-neighbour distance across batch)
Prevents representation collapse without needing negative pairs.

All functions operate on float32 L2-normalised features and return
scalar tensors.  They are pure functions with no nn.Module state.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def dino_cls_loss(
    student_cls: Tensor,    # (B, D)
    teacher_cls: Tensor,    # (B, D)
) -> Tensor:
    """
    Symmetric cosine distance between student and teacher CLS tokens.
    Both inputs are L2-normalised internally.

    Returns a scalar loss tensor.
    """
    s = F.normalize(student_cls.float(), dim=-1)
    t = F.normalize(teacher_cls.float(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


def dino_cls_loss_multicrop(
    student_cls: Tensor,    # (B, D) — student on masked global crop
    teacher_cls: Tensor,    # (B, D) — teacher on clean global crop
    local_cls_list: list[Tensor],   # list of (B, D) from local crops
    local_weight: float = 0.5,
) -> Tensor:
    """
    Full DINO multi-crop CLS loss.

    All local crop student embeddings are pulled toward the teacher's
    global crop embedding.  This encourages global-local consistency.
    """
    loss_global = dino_cls_loss(student_cls, teacher_cls)
    if not local_cls_list:
        return loss_global
    loss_local = sum(
        dino_cls_loss(loc, teacher_cls) for loc in local_cls_list
    ) / len(local_cls_list)
    return loss_global + local_weight * loss_local


def ibot_patch_loss(
    student_patches: Tensor,    # (B, N, D)
    teacher_patches: Tensor,    # (B, N, D)
    active_mask: Tensor,        # (B, N) bool — real AND freq-masked positions
) -> Tensor:
    """
    iBOT masked patch prediction loss.

    Only computes the loss at positions marked True in active_mask.
    If no positions are active, returns 0.

    Parameters
    ----------
    student_patches : features at ALL patch positions (model predicts masked ones)
    teacher_patches : clean teacher features at same positions
    active_mask     : True = compute loss here (real content AND was masked)
    """
    if not active_mask.any():
        return student_patches.new_tensor(0.0)

    flat_s = student_patches.reshape(-1, student_patches.shape[-1])
    flat_t = teacher_patches.reshape(-1, teacher_patches.shape[-1])
    flat_m = active_mask.flatten()

    s_sel = F.normalize(flat_s[flat_m].float(), dim=-1)
    t_sel = F.normalize(flat_t[flat_m].float(), dim=-1)

    return (1.0 - (s_sel * t_sel).sum(dim=-1)).mean()


def koleo_loss(
    features: Tensor,            # (B, D) L2-normalised student features
    eps: float = 1e-8,
) -> Tensor:
    """
    KoLeo uniformity loss (DINOv3 eq. 4).

    L_KoLeo = -(1/B) Σ_i log(min_{j≠i} ||f_i - f_j||_2)

    Encourages features to spread uniformly over the unit sphere.
    Applied to student CLS tokens (no teacher involved).

    Parameters
    ----------
    features : (B, D)  L2-normalised features
    eps      : small value to avoid log(0)
    """
    B = features.shape[0]
    if B < 2:
        return features.new_tensor(0.0)

    f = F.normalize(features.float(), dim=-1)

    # Pairwise squared distances: ||f_i - f_j||^2 = 2 - 2*dot(f_i, f_j)
    dots = f @ f.T                              # (B, B)
    sq_dist = (2.0 - 2.0 * dots).clamp(min=0)  # (B, B), diagonal = 0

    # Mask diagonal (self-distance) with large value before taking min
    sq_dist.fill_diagonal_(float("inf"))

    # Nearest-neighbour distances
    nn_dist = sq_dist.min(dim=-1).values        # (B,)  ≥ 0
    nn_dist = nn_dist.clamp(min=eps)

    return -nn_dist.log().mean()
