"""
models/losses/cross_branch.py  ·  Cross-branch distillation loss
======================================================================

cross_branch_loss
-----------------
Aligns image teacher patch representations with video student tube
representations in a shared low-dimensional projection space.

The loss encourages the video student to learn representations that are
consistent with the image teacher's semantic structure — even though
the video student never sees static images during Phase 3.

Architecture
------------
Both image patch tokens and video tube tokens are projected into a
shared 256-D alignment space via independent linear projections,
L2-normalised, then aligned with cosine distance.

Mean-pooling over spatial/temporal positions reduces variable-length
sequences to a single (B, 256) vector before computing the loss.

This is the formalisation of CrossBranchDistillation from
models/branches/shared.py — that class holds the learnable projection
parameters; this module holds the loss computation logic.

Loss weight
-----------
λ6 in the full loss:
    L_total = L_img + L_vid + λ6·L_cross + λ7·L_proto + λ_gram·L_gram

λ6 is ramped from 0.5 (Phase 3, Stage 2) to 1.0 (Phase 3, Stage 3).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def cross_branch_loss(
    img_feat: Tensor,   # (B_img, align_dim) — projected + L2-normed image features
    vid_feat: Tensor,   # (B_vid, align_dim) — projected + L2-normed video features
) -> Tensor:
    """
    Cosine distance between image teacher and video student features
    in the shared projection space.

    Parameters
    ----------
    img_feat : already L2-normalised, shape (B_img, D_align)
    vid_feat : already L2-normalised, shape (B_vid, D_align)

    Both inputs are normalised externally (by CrossBranchDistillation.forward)
    so this function is a pure distance computation.

    When batch sizes differ (B_img ≠ B_vid), aligns on min(B_img, B_vid).
    This can happen when image and video batch sizes differ in config.
    """
    B = min(img_feat.shape[0], vid_feat.shape[0])
    if B == 0:
        return img_feat.new_tensor(0.0)
    return (1.0 - (img_feat[:B] * vid_feat[:B]).sum(dim=-1)).mean()


def cross_branch_loss_from_tokens(
    img_patch_tokens: Tensor,    # (B_img, N_img, D_img)
    vid_tube_tokens: Tensor,     # (B_vid, N_vid, D_vid)
    proj_img: torch.nn.Module,   # Linear(D_img, align_dim)
    proj_vid: torch.nn.Module,   # Linear(D_vid, align_dim)
) -> Tensor:
    """
    Convenience wrapper: projects raw tokens and computes the loss.

    This is called when you have raw backbone outputs rather than
    pre-projected features.  The projections are the ones stored in
    CrossBranchDistillation (models/branches/shared.py).
    """
    img_feat = F.normalize(proj_img(img_patch_tokens.mean(1).float()), dim=-1)
    vid_feat = F.normalize(proj_vid(vid_tube_tokens.mean(1).float()), dim=-1)
    return cross_branch_loss(img_feat, vid_feat)
