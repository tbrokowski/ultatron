"""
models/losses/image_losses.py  ·  Image-branch SSL losses
==============================================================

Cosine-distance losses (original DINOv2/iBOT style):

dino_cls_loss / dino_cls_loss_multicrop
  L_cls = mean(1 - cos(student_cls, teacher_cls))

ibot_patch_loss
  L_patch = mean(1 - cos(student_patches, teacher_patches))
  at real-content AND frequency-masked positions.

Cross-entropy losses (paper formulation — softmax + CE with temperature):

dino_cls_loss_ce
  p = softmax(f / τ); L = CE(p_teacher, p_student)
  Applied to global CLS tokens and local crop CLS tokens.

ibot_patch_loss_ce
  Same CE formulation at masked patch positions.

V-JEPA 2.1 dense context loss:

dense_context_loss
  Applied to VISIBLE (unmasked) tokens, weighted by distance to
  nearest masked patch.  Enforces local spatial continuity.
  λ_i = λ / sqrt(d_min(i, masked)), L1 in feature space.

Pixel reconstruction loss:

pixel_reconstruction_loss
  L1 between a linear reconstruction head's output and the original
  (pre-masking) pixel intensities at masked patch positions.

KoLeo uniformity regulariser:

koleo_loss
  L_KoLeo = -log(min nearest-neighbour distance across batch)

Grouped / unified SPC losses:

spectral_predictive_coding_loss  (grouped / unified modes)
  Single energy-weighted CE objective that subsumes CLS + patch +
  intermediate + local-crop alignment losses.  Token-level loss is
  scaled by the continuous spectral-energy map supplied by the
  frequency-masking pipeline.

grouped_decode_loss  (grouped mode)
  Combines pixel reconstruction + external-teacher distillation at
  high-energy positions into one L_decode term.

All functions operate on float32 features and return scalar tensors.
They are pure functions with no nn.Module state.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
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


def dino_cls_loss_ce(
    student_cls: Tensor,          # (B, D)
    teacher_cls: Tensor,          # (B, D)
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
) -> Tensor:
    """
    Cross-entropy loss between softmax distributions over the feature dimension.

    Teacher acts as soft target (sharp, low temperature).
    Student acts as prediction (higher temperature, softer).
    L = -sum_d( softmax(t/τ_t) · log_softmax(s/τ_s) )

    This is the original DINO formulation, extended to cover both global→global
    and global→local view pairs via dino_cls_loss_ce_multicrop.
    """
    t     = F.softmax(teacher_cls.float() / teacher_temp, dim=-1)
    log_s = F.log_softmax(student_cls.float() / student_temp, dim=-1)
    return -(t * log_s).sum(dim=-1).mean()


def dino_cls_loss_ce_multicrop(
    student_cls: Tensor,           # (B, D) — student on masked global crop
    teacher_cls: Tensor,           # (B, D) — teacher on clean global crop
    local_cls_list: list[Tensor],  # list of (B, D) from local crops
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    local_weight: float = 0.5,
) -> Tensor:
    """Full DINO multi-crop CE loss. Local student crops pulled toward teacher global."""
    loss_global = dino_cls_loss_ce(student_cls, teacher_cls, student_temp, teacher_temp)
    if not local_cls_list:
        return loss_global
    loss_local = sum(
        dino_cls_loss_ce(loc, teacher_cls, student_temp, teacher_temp)
        for loc in local_cls_list
    ) / len(local_cls_list)
    return loss_global + local_weight * loss_local


def ibot_patch_loss_ce(
    student_patches: Tensor,  # (B, N, D)
    teacher_patches: Tensor,  # (B, N, D)
    active_mask: Tensor,      # (B, N) bool — real AND freq-masked positions
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
) -> Tensor:
    """
    iBOT patch prediction loss using cross-entropy (paper formulation).

    Only computes loss at active_mask=True positions (real content that was masked).
    Teacher provides soft targets; student must recover the distribution.
    """
    if not active_mask.any():
        return student_patches.new_tensor(0.0)

    flat_s = student_patches.reshape(-1, student_patches.shape[-1])
    flat_t = teacher_patches.reshape(-1, teacher_patches.shape[-1])
    flat_m = active_mask.flatten()

    t_sel     = F.softmax(flat_t[flat_m].float() / teacher_temp, dim=-1)
    log_s_sel = F.log_softmax(flat_s[flat_m].float() / student_temp, dim=-1)
    return -(t_sel * log_s_sel).sum(dim=-1).mean()


def dense_context_loss(
    student_patches: Tensor,   # (B, N, D)
    teacher_patches: Tensor,   # (B, N, D)
    patch_mask: Tensor,        # (B, ph, pw) bool — True = masked
    lam_ctx: float = 1.0,
) -> Tensor:
    """
    V-JEPA 2.1 distance-weighted context loss on VISIBLE (unmasked) tokens.

    For each visible token i:
        λ_i = lam_ctx / sqrt(d_min(i, M))

    where d_min(i, M) is the minimum Manhattan distance from token i to the
    nearest masked patch in the (ph, pw) grid.  Loss is L1 in feature space.

    Tokens close to masked regions are penalised more, enforcing spatial
    continuity in the student's representations.
    """
    B, ph, pw = patch_mask.shape
    N = ph * pw

    if not patch_mask.any():
        return student_patches.new_tensor(0.0)

    device = patch_mask.device
    iy = torch.arange(ph, device=device, dtype=torch.float32)
    ix = torch.arange(pw, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(iy, ix, indexing="ij")  # (ph, pw)
    all_y = gy.flatten()  # (N,)
    all_x = gx.flatten()  # (N,)

    weights_list = []
    for b in range(B):
        m = patch_mask[b]   # (ph, pw)
        if not m.any():
            weights_list.append(patch_mask.new_zeros(N, dtype=torch.float32))
            continue
        my = gy[m]  # (K,)
        mx = gx[m]  # (K,)
        # (N, K) Manhattan distances
        dist = (all_y.unsqueeze(1) - my.unsqueeze(0)).abs() + \
               (all_x.unsqueeze(1) - mx.unsqueeze(0)).abs()
        d_min = dist.min(dim=1).values.clamp(min=1.0)   # (N,)
        w = lam_ctx / d_min.sqrt()
        w = w * (~m.flatten()).float()  # zero out masked positions
        weights_list.append(w)

    weights = torch.stack(weights_list, dim=0)  # (B, N)

    flat_s = student_patches.float().reshape(B * N, -1)
    flat_t = teacher_patches.float().reshape(B * N, -1)
    flat_w = weights.reshape(B * N)

    l1   = (flat_s - flat_t).abs().mean(dim=-1)   # (B*N,)
    denom = flat_w.sum().clamp(min=1e-8)
    return (flat_w * l1).sum() / denom


def pixel_reconstruction_loss(
    pred_pixels: Tensor,    # (B, N, ps*ps*C)  student reconstruction head output
    target_img: Tensor,     # (B, C, H, W)     clean unmasked image (pre-masking crop)
    active_mask: Tensor,    # (B, N) bool       True at MASKED patch positions
    patch_size: int = 16,
) -> Tensor:
    """
    L1 pixel reconstruction loss at masked patch positions.

    The student's reconstruction head predicts the pixel intensities of the
    masked patches using context from unmasked patch tokens.  The target is
    the original (pre-masking) pixel values at those positions.

    Uses F.unfold to extract non-overlapping patch pixel windows from
    target_img, then computes L1 at active_mask=True positions only.
    """
    if not active_mask.any():
        return pred_pixels.new_tensor(0.0)

    # Extract non-overlapping patch pixels: (B, C*ps*ps, N)
    target_patches = F.unfold(
        target_img.float(),
        kernel_size=patch_size,
        stride=patch_size,
    )   # (B, C*ps*ps, N)
    target_patches = target_patches.permute(0, 2, 1)  # (B, N, C*ps*ps)

    flat_pred = pred_pixels.float().reshape(-1, pred_pixels.shape[-1])
    flat_tgt  = target_patches.reshape(-1, target_patches.shape[-1])
    flat_m    = active_mask.flatten()

    return F.l1_loss(flat_pred[flat_m], flat_tgt[flat_m].clamp(0.0, 1.0))


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


# ---------------------------------------------------------------------------
# Grouped / Unified SPC losses
# ---------------------------------------------------------------------------

def _ce_per_token(
    student: Tensor,   # (..., D)
    teacher: Tensor,   # (..., D)
    tau_s: float,
    tau_t: float,
) -> Tensor:
    """Cross-entropy loss per token (scalar per position in leading dims)."""
    t     = F.softmax(teacher.float() / tau_t, dim=-1)
    log_s = F.log_softmax(student.float() / tau_s, dim=-1)
    return -(t * log_s).sum(dim=-1)  # (...,)


def spectral_predictive_coding_loss(
    student_cls: Tensor,                        # (B, D)
    student_patches: Tensor,                    # (B, N, D)
    teacher_cls: Tensor,                        # (B, D)
    teacher_patches: Tensor,                    # (B, N, D)
    energy_map: Tensor,                         # (B, N) float [0,1]
    local_cls_list: List[Tensor],               # list of (B, D)
    tau_s: float,
    tau_t: float,
    padding_mask: Optional[Tensor] = None,      # (B, N) bool — True = padding
    intermediate_pairs: Optional[List[Tuple[Tensor, Tensor]]] = None,
    ctx_base: float = 0.0,
    local_weight: float = 0.5,
) -> Tensor:
    """
    Spectral Predictive Coding loss — single energy-weighted CE objective.

    Subsumes CLS alignment, patch alignment, intermediate layer alignment,
    and local crop alignment into one formula weighted by the continuous
    spectral-energy map from the FFT masking pipeline.

    Token-level weight:
        w_patch_i = energy_map_i + ctx_base   (adds a floor for visible tokens)
        w_cls     = mean(energy_map, dim=1)   (one scalar per sample)

    Intermediate layer tokens (if provided) are averaged over layers with
    the same per-token weights as the final-layer patches.

    Local crop CLS tokens are pulled toward the teacher's global CLS using
    w_cls as the per-sample weight.

    Parameters
    ----------
    student_cls, teacher_cls     : (B, D)
    student_patches, teacher_patches : (B, N, D)
    energy_map                   : (B, N) float — spectral energy per patch
    local_cls_list               : list of (B, D) from additional local crops
    tau_s, tau_t                 : student / teacher temperatures
    padding_mask                 : (B, N) bool True = padding token (excluded)
    intermediate_pairs           : list of (student_layer, teacher_layer) tuples,
                                   each (B, N, D)
    ctx_base                     : small floor added to all patch weights so
                                   visible patches still contribute to the loss
    local_weight                 : relative weight of local crop CLS loss
    """
    energy = energy_map.float()  # (B, N)

    # Per-patch weight: continuous energy + optional floor for visible tokens
    w_patch = (energy + ctx_base).clamp(min=0.0)          # (B, N)
    if padding_mask is not None:
        w_patch = w_patch * (~padding_mask).float()

    # Per-sample CLS weight: mean energy in the spatial map
    w_cls = energy.mean(dim=1)                             # (B,)

    total_w = w_patch.new_tensor(0.0)
    weighted_sum = w_patch.new_tensor(0.0)

    # ── Final-layer patch alignment ─────────────────────────────────────────
    ce_patch = _ce_per_token(student_patches, teacher_patches, tau_s, tau_t)  # (B, N)
    weighted_sum = weighted_sum + (w_patch * ce_patch).sum()
    total_w      = total_w      + w_patch.sum()

    # ── CLS alignment ───────────────────────────────────────────────────────
    ce_cls = _ce_per_token(student_cls, teacher_cls, tau_s, tau_t)   # (B,)
    weighted_sum = weighted_sum + (w_cls * ce_cls).sum()
    total_w      = total_w      + w_cls.sum()

    # ── Local crop CLS alignment ─────────────────────────────────────────────
    if local_cls_list:
        ce_local_sum = sum(
            _ce_per_token(loc, teacher_cls, tau_s, tau_t) for loc in local_cls_list
        )  # (B,) — sum over local views
        ce_local_mean = ce_local_sum / len(local_cls_list)
        weighted_sum = weighted_sum + local_weight * (w_cls * ce_local_mean).sum()
        total_w      = total_w      + local_weight * w_cls.sum()

    # ── Intermediate layer patch alignment ──────────────────────────────────
    if intermediate_pairs:
        n_inter = len(intermediate_pairs)
        for s_layer, t_layer in intermediate_pairs:
            ce_inter = _ce_per_token(s_layer, t_layer, tau_s, tau_t)   # (B, N)
            weighted_sum = weighted_sum + (w_patch * ce_inter).sum() / n_inter
        total_w = total_w + w_patch.sum()   # one extra N-patch weight block

    return weighted_sum / total_w.clamp(min=1e-8)


def grouped_decode_loss(
    student_patches: Tensor,                         # (B, N, D)
    active_mask: Tensor,                             # (B, N) bool
    recon_head: Optional[nn.Module],                 # projects D → ps*ps*C
    raw_crops: Optional[Tensor],                     # (B, C, H, W)
    patch_size: int,
    external_teachers: Optional[List[Tuple[Tensor, nn.Module]]] = None,
    alpha_recon: float = 0.5,
    alpha_ext: float = 0.5,
) -> Tensor:
    """
    Grouped L_decode term: pixel reconstruction + external teacher distillation.

    Combines ``pixel_reconstruction_loss`` at masked patches with an L1
    feature-space distillation loss toward each external teacher (e.g. SAM,
    DINOv3-7B), weighted by ``alpha_recon`` and ``alpha_ext`` respectively.

    Parameters
    ----------
    student_patches  : (B, N, D) — student patch features
    active_mask      : (B, N) bool — positions contributing to decode loss
    recon_head       : nn.Module(D → ps*ps*C) or None
    raw_crops        : (B, C, H, W) original unmasked crop or None
    patch_size       : patch size (pixels)
    external_teachers: list of (teacher_patches (B,N,D), proj_head nn.Module)
                       where proj_head maps student D → teacher D′.
    alpha_recon      : weight for pixel reconstruction term
    alpha_ext        : weight for each external teacher term
    """
    if not active_mask.any():
        return student_patches.new_tensor(0.0)

    terms: list[Tensor] = []

    # ── Pixel reconstruction ─────────────────────────────────────────────────
    if recon_head is not None and raw_crops is not None:
        pred_pixels = recon_head(student_patches)   # (B, N, ps*ps*C)
        terms.append(
            alpha_recon * pixel_reconstruction_loss(
                pred_pixels, raw_crops, active_mask, patch_size
            )
        )

    # ── External teacher distillation at masked positions ────────────────────
    if external_teachers:
        flat_s = student_patches.reshape(-1, student_patches.shape[-1])
        flat_m = active_mask.flatten()
        for teacher_patches, proj_head in external_teachers:
            proj_s = proj_head(flat_s[flat_m])        # (K, D_t)
            flat_t = teacher_patches.reshape(-1, teacher_patches.shape[-1])
            terms.append(
                alpha_ext * F.l1_loss(
                    F.normalize(proj_s.float(), dim=-1),
                    F.normalize(flat_t[flat_m].float(), dim=-1),
                )
            )

    if not terms:
        return student_patches.new_tensor(0.0)

    return sum(terms) / len(terms)
