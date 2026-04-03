"""
models/losses/cross_branch.py  ·  Cross-branch alignment losses
======================================================================

Three losses are provided, applied in Phase 3:

cross_branch_loss / cross_branch_loss_from_tokens
-------------------------------------------------
Legacy global mean-pool cosine distance.  Still used as a fallback when
no alignment pairs exist (e.g. Stage 1, or disjoint-study batches).

filip_cross_loss_for_pair
--------------------------
FILIP-style token-level maximum similarity (Yao et al. 2021).
Operates on a single (img, vid) pair indexed by an AlignmentPair.

  - Projects image patch tokens and video tube tokens at frame_offset
    into align_dim, L2-normalises per token.
  - i2v: each image patch finds its nearest video tube token → mean
  - v2i: each video tube token finds its nearest image patch → mean
  - Loss = (1 - i2v + 1 - v2i) / 2, weighted by pair.weight.

This preserves spatial correspondence: the hepatic patch aligns with
the hepatic tube token rather than averaging everything together.

infonce_cross_loss / infonce_cross_loss_from_pairs
--------------------------------------------------
CLIP-style InfoNCE contrastive loss (Radford et al. 2021) over a batch
of pre-projected, L2-normalised (B_pair, align_dim) image/video features.

  sim_matrix = img_feats @ vid_feats.T / temperature   (B_global, B_global)
  diagonal = positive pairs; off-diagonal = in-batch negatives
  L = (CE(sim, labels) + CE(sim.T, labels)) / 2

In distributed training, features are all-gathered across ranks before
forming the similarity matrix (effective batch = B_local × world_size).
Gradients flow only through each rank's own local slice — gathered copies
from other ranks are treated as constants (MoCo v3 / OpenCLIP pattern).

Requires at least 2 global pairs; returns 0 otherwise.
Applied in Stage 3 only.

Loss weights (in phase3_step lam dict)
---------------------------------------
lam6        FILIP cross-branch weight (Stage 2: ×0.5, Stage 3: ×1.0)
lam6_nce    InfoNCE weight (Stage 3 only)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from data.pipeline.collators_extended import AlignmentPair


# ── Distributed helpers ────────────────────────────────────────────────────────

def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_gather_nograd(t: Tensor) -> Tensor:
    """
    Gather a tensor from all ranks along dim 0; no gradient through the result.

    Returns the input unchanged when not in a distributed context (single-GPU,
    unit tests, etc.), so all callers degrade gracefully.
    """
    if not _is_dist() or dist.get_world_size() == 1:
        return t.detach()
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0)


def _gather_with_local_grad(t: Tensor) -> Tensor:
    """
    Gather tensor from all ranks, but keep the gradient path live through
    this rank's own slice.

    Pattern (MoCo v3 / OpenCLIP):
      1. All-gather with no grad → global tensor of detached copies.
      2. Splice this rank's live (grad-attached) slice back in-place.

    Result: a (world_size * B_local, D) tensor where only the local slice
    carries gradients.  The (B_global, B_global) similarity matrix computed
    from it will propagate gradients only to this rank's features — correct
    behaviour for DDP because each rank updates its own parameters.
    """
    if not _is_dist() or dist.get_world_size() == 1:
        return t

    world = dist.get_world_size()
    rank  = dist.get_rank()

    gathered = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.contiguous())

    # Replace the local rank's entry with the live tensor so grads flow through
    gathered[rank] = t
    return torch.cat(gathered, dim=0)


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
    proj_img: nn.Module,         # Linear(D_img, align_dim)
    proj_vid: nn.Module,         # Linear(D_vid, align_dim)
    predictor_vid: Optional[nn.Module] = None,  # BYOL predictor on video side
) -> Tensor:
    """
    Legacy convenience wrapper: global mean-pool + cosine distance.

    predictor_vid: if provided (CrossBranchDistillation.predictor_vid),
    applied to video features after projection (BYOL-style asymmetric head).
    """
    img_feat = F.normalize(proj_img(img_patch_tokens.mean(1).float()), dim=-1)
    vid_proj = proj_vid(vid_tube_tokens.mean(1).float())
    if predictor_vid is not None:
        vid_proj = predictor_vid(vid_proj)
    vid_feat = F.normalize(vid_proj, dim=-1)
    return cross_branch_loss(img_feat, vid_feat)


# ── FILIP token-level alignment ────────────────────────────────────────────────

def filip_cross_loss_for_pair(
    img_tokens: Tensor,          # (N, D_img)  image teacher patch tokens for one sample
    vid_tube_tokens: Tensor,     # (T*ph*pw, D_vid)  video student ALL tube tokens for one sample
    proj_img: nn.Module,         # Linear(D_img, align_dim)
    proj_vid: nn.Module,         # Linear(D_vid, align_dim)
    predictor_vid: Optional[nn.Module],  # BYOL predictor (video side only)
    frame_offset: int,           # temporal frame index in the video clip (0-indexed)
    ph: int,                     # spatial patch grid height
    pw: int,                     # spatial patch grid width
) -> Tensor:
    """
    FILIP-style per-pair token-level maximum cosine similarity loss.

    Slices vid_tube_tokens at [frame_offset*ph*pw : (frame_offset+1)*ph*pw]
    to extract the spatial tokens at the temporally aligned frame, then:
      - Projects both to align_dim, L2-normalises per token (not per sequence).
      - i2v: each image patch finds its closest video tube token.
      - v2i: each tube token finds its closest image patch.
      - Loss = (1 - mean(i2v) + 1 - mean(v2i)) / 2

    This preserves spatial grounding: each anatomical region in the image
    is pulled toward the corresponding spatial-temporal region in the video.
    """
    N_vid = vid_tube_tokens.shape[0]
    n_spatial = ph * pw
    t_start   = frame_offset * n_spatial
    t_end     = min(t_start + n_spatial, N_vid)

    if t_start >= N_vid or t_start >= t_end:
        # Frame offset out of range — fall back to all tokens
        vid_frame_tokens = vid_tube_tokens
    else:
        vid_frame_tokens = vid_tube_tokens[t_start:t_end]   # (ph*pw, D_vid)

    # Project to align_dim, normalise per token
    img_proj = F.normalize(proj_img(img_tokens.float()), dim=-1)    # (N, align_dim)
    vid_proj_raw = proj_vid(vid_frame_tokens.float())               # (ph*pw, align_dim)
    if predictor_vid is not None:
        vid_proj_raw = predictor_vid(vid_proj_raw)
    vid_proj = F.normalize(vid_proj_raw, dim=-1)                    # (ph*pw, align_dim)

    sim = img_proj @ vid_proj.T     # (N, ph*pw)

    i2v = sim.max(dim=1).values.mean()   # each image patch → best video token
    v2i = sim.max(dim=0).values.mean()   # each video token → best image patch

    return (1.0 - i2v + 1.0 - v2i) / 2.0


def filip_cross_loss_from_pairs(
    img_patch_tokens: Tensor,    # (B_img, N, D_img) — all image teacher tokens in batch
    vid_tube_tokens: Tensor,     # (B_vid, T*ph*pw, D_vid) — all video student tokens in batch
    proj_img: nn.Module,
    proj_vid: nn.Module,
    predictor_vid: Optional[nn.Module],
    alignment_pairs: "List[AlignmentPair]",
    ph: int,
    pw: int,
) -> Tensor:
    """
    Iterates over alignment_pairs and averages FILIP loss across all pairs,
    weighting each pair by pair.weight (1.0 = exact frame, decays with distance).

    Returns 0 if alignment_pairs is empty.
    """
    if not alignment_pairs:
        return img_patch_tokens.new_tensor(0.0)

    total_loss   = img_patch_tokens.new_tensor(0.0)
    total_weight = 0.0

    for pair in alignment_pairs:
        img_tok = img_patch_tokens[pair.img_batch_idx]   # (N, D_img)
        vid_tok = vid_tube_tokens[pair.vid_batch_idx]    # (T*ph*pw, D_vid)
        pair_loss = filip_cross_loss_for_pair(
            img_tok, vid_tok, proj_img, proj_vid, predictor_vid,
            pair.frame_offset, ph, pw,
        )
        total_loss   = total_loss + pair.weight * pair_loss
        total_weight += pair.weight

    if total_weight > 0:
        return total_loss / total_weight
    return total_loss


# ── InfoNCE contrastive loss ───────────────────────────────────────────────────

def infonce_cross_loss(
    img_feats: Tensor,      # (B_global, align_dim) — projected + L2-normed
    vid_feats: Tensor,      # (B_global, align_dim) — projected + L2-normed (after predictor)
    temperature: float = 0.07,
    rank_offset: int = 0,   # this rank's start index in the global batch (for label offset)
) -> Tensor:
    """
    CLIP-style InfoNCE loss over a global batch of aligned (image, video) pairs.

    The (B_global, B_global) similarity matrix has positive pairs on the diagonal
    and all other positions as in-batch negatives.  In distributed training,
    img_feats / vid_feats are the globally-gathered tensors; rank_offset shifts
    the CE labels so each rank's positives sit on the correct diagonal entries.

    Requires at least 2 pairs; returns 0 otherwise.

    Parameters
    ----------
    img_feats    : (B_global, align_dim)  already L2-normalised image features
    vid_feats    : (B_global, align_dim)  already L2-normalised video features
    temperature  : float  default 0.07 (CLIP standard)
    rank_offset  : int    this rank's start row/col in the global sim matrix
    """
    B = img_feats.shape[0]
    if B < 2:
        return img_feats.new_tensor(0.0)

    sim = img_feats @ vid_feats.T / temperature   # (B_global, B_global)
    labels = torch.arange(B, device=img_feats.device)

    loss_i2v = F.cross_entropy(sim,   labels)
    loss_v2i = F.cross_entropy(sim.T, labels)
    return (loss_i2v + loss_v2i) / 2.0


def infonce_cross_loss_from_pairs(
    img_patch_tokens: Tensor,    # (B_img, N, D_img)
    vid_tube_tokens: Tensor,     # (B_vid, T*ph*pw, D_vid)
    proj_img: nn.Module,
    proj_vid: nn.Module,
    predictor_vid: Optional[nn.Module],
    alignment_pairs: "List[AlignmentPair]",
    ph: int,
    pw: int,
    temperature: float = 0.07,
) -> Tensor:
    """
    Builds per-pair mean-pooled projections for each alignment pair on this rank,
    all-gathers features from all DDP ranks, then computes InfoNCE over the full
    global batch (effective size = B_local × world_size).

    Gradient flow:
        Only this rank's local slice of img_feats / vid_feats carries gradients
        (_gather_with_local_grad pattern).  Remote slices are detached constants,
        serving only as hard negatives.

    Returns 0 if the global batch has fewer than 2 pairs.
    """
    if len(alignment_pairs) < 1:
        return img_patch_tokens.new_tensor(0.0)

    img_feat_list = []
    vid_feat_list = []

    for pair in alignment_pairs:
        img_tok = img_patch_tokens[pair.img_batch_idx]   # (N, D_img)
        vid_tok = vid_tube_tokens[pair.vid_batch_idx]    # (T*ph*pw, D_vid)

        n_spatial = ph * pw
        t_start   = pair.frame_offset * n_spatial
        t_end     = min(t_start + n_spatial, vid_tok.shape[0])
        vid_frame = vid_tok[t_start:t_end] if t_start < t_end else vid_tok

        img_proj = F.normalize(proj_img(img_tok.float().mean(0)), dim=-1)   # (align_dim,)
        vid_proj_raw = proj_vid(vid_frame.float().mean(0))                   # (align_dim,)
        if predictor_vid is not None:
            vid_proj_raw = predictor_vid(vid_proj_raw)
        vid_proj = F.normalize(vid_proj_raw, dim=-1)

        img_feat_list.append(img_proj)
        vid_feat_list.append(vid_proj)

    # Local features — (B_local, align_dim), grads live here
    img_local = torch.stack(img_feat_list)
    vid_local = torch.stack(vid_feat_list)

    # All-gather to (B_global, align_dim); only the local slice retains grads
    img_global = _gather_with_local_grad(img_local)
    vid_global = _gather_with_local_grad(vid_local)

    if img_global.shape[0] < 2:
        return img_patch_tokens.new_tensor(0.0)

    return infonce_cross_loss(img_global, vid_global, temperature)
