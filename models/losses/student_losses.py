"""
models/losses/student_losses.py  ·  SSL losses for the single-student pipeline
===============================================================================

All functions:
  - Are pure functions (no nn.Module state)
  - Accept optional padding_mask (B, ph, pw) bool — losses only at True positions
  - Return scalar Tensor
  - Operate on float32 (cast internally if needed)

Loss catalogue
--------------
Image-side (T=1 batches)
  img_global_distill_loss   cosine: student global vs frozen DINO global CLS
  img_patch_distill_loss    token-level cosine at unmasked positions
  img_masked_semantic_loss  cosine at masked positions (iBOT-style)
  img_proto_loss            SwAV asymmetric prototype CE (teacher Sinkhorn → student softmax)
  img_ema_global_loss       cosine: student global vs EMA teacher (native hidden)
  img_ema_patch_loss        cosine: student F1 vs EMA F1 at masked-valid positions

Video-side (T>1 batches)
  vid_global_distill_loss   cosine: student clip global vs V-JEPA clip global
  vid_tube_prediction_loss  cosine at masked tube positions (V-JEPA latent pred)
  vid_temporal_consistency  smoothness of per-frame embeddings across T
  vid_proto_loss            V-JEPA-aligned prototype CE on pooled video tubes
  prototype_assignment_stats  entropy / max-prob diagnostics for logging

Paired-batch (Stage 3 paired frame + clip)
  fused_distill_loss        cosine(s_vid, stopgrad(z_fused))
  preservation_loss         cosine(z_fused, stopgrad(z_vid)) — anchor fused to V-JEPA
  frame_clip_consistency    cosine: student(frame as T=1) vs student(clip)[frame slot]

Shared utility
  cosine_loss               1 - cos(a, b), optionally masked
  prototype_assign          soft prototype assignment (Sinkhorn or softmax)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def cosine_loss(
    a: Tensor,                          # (..., D)
    b: Tensor,                          # (..., D)
    mask: Optional[Tensor] = None,      # (...) bool, True = include
) -> Tensor:
    """
    1 - cosine_similarity(a, b), averaged over valid positions.

    Works for any leading dimensions (scalar, (B,), (B,N), etc.).
    """
    a = F.normalize(a.float(), dim=-1, eps=1e-6)
    b = F.normalize(b.float(), dim=-1, eps=1e-6)
    sim = (a * b).sum(dim=-1)           # (...) in [-1, 1]
    loss = 1.0 - sim

    if mask is not None:
        mask = mask.bool()
        if mask.any():
            return loss[mask].mean()
        return loss.new_tensor(0.0)
    return loss.mean()


def prototype_assign(
    tokens: Tensor,          # (B, N, D) or (B, D)
    prototypes: Tensor,      # (K, D)
    temperature: float = 0.1,
    sinkhorn_iters: int = 3,
    use_sinkhorn: bool = True,
) -> Tensor:
    """
    Soft prototype assignment.

    For teacher tokens use use_sinkhorn=True (Sinkhorn for stable clusters).
    For student tokens use use_sinkhorn=False (plain softmax).

    Returns (B, K) or (B, N, K) probability vectors.
    """
    proto = F.normalize(prototypes.float(), dim=-1)
    t = F.normalize(tokens.float(), dim=-1)

    if t.dim() == 2:
        logits = (t @ proto.T) / temperature         # (B, K)
    else:
        logits = torch.einsum("bnd,kd->bnk", t, proto) / temperature  # (B, N, K)

    if use_sinkhorn:
        return _sinkhorn(logits, n_iters=sinkhorn_iters)
    return F.softmax(logits, dim=-1)


def _sinkhorn(logits: Tensor, n_iters: int = 3) -> Tensor:
    """
    Sinkhorn-Knopp normalisation for balanced prototype assignments.
    Operates on the last two dimensions (B, K).

    logits are expected to already be scaled by the caller's temperature.
    Applies max-shift per row for numerical stability before exp.

    Alternates col→row normalisation for n_iters, then final row-norm so
    rows sum to 1 (probability distributions over prototypes).
    Used only for logging stats (prototype_assignment_stats).
    """
    x = logits.float() - logits.float().amax(dim=-1, keepdim=True)   # per-row stability
    Q = torch.exp(x.clamp(-50.0, 50.0))
    for _ in range(n_iters):
        Q = Q / Q.sum(dim=0, keepdim=True).clamp(min=1e-8)   # col: equal prototype use
        Q = Q / Q.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # row: valid distributions
    return Q / Q.sum(dim=-1, keepdim=True).clamp(min=1e-8)   # final row-norm


# ---------------------------------------------------------------------------
# Image-side losses
# ---------------------------------------------------------------------------

def img_global_distill_loss(
    student_global: Tensor,    # (B, D_student)
    teacher_global: Tensor,    # (B, D_teacher_proj)  already projected
) -> Tensor:
    """Cosine distance: student global token vs frozen DINO global CLS."""
    return cosine_loss(student_global, teacher_global)


def img_patch_distill_loss(
    student_patches: Tensor,           # (B, N_s, D_s)
    teacher_patches: Tensor,           # (B, N_t, D_t_proj)  already projected
    mask: Optional[Tensor] = None,     # (B, N) bool — unmasked positions only
) -> Tensor:
    """
    Token-level cosine distillation at unmasked patch positions.

    When student and teacher grids differ in spatial resolution (N_s ≠ N_t),
    the higher-resolution grid is downsampled to match the coarser teacher
    grid (preferred — avoids 16× upsample memory spikes on the student grid).
    """
    student_patches, teacher_patches = _align_patch_grids(student_patches, teacher_patches)

    if mask is not None:
        flat_mask = _align_token_mask(mask, student_patches.shape[1])
        return cosine_loss(student_patches, teacher_patches, mask=flat_mask)
    return cosine_loss(student_patches, teacher_patches)


def img_masked_semantic_loss(
    student_patches: Tensor,     # (B, N, D_s)
    teacher_patches: Tensor,     # (B, N, D_t_proj)
    masked_positions: Tensor,    # (B, N) bool — True = masked (predict here)
    padding_mask: Optional[Tensor] = None,  # (B, ph, pw) bool
) -> Tensor:
    """
    iBOT-style masked patch prediction: student must reconstruct teacher
    tokens at positions that were masked (the hard prediction task).
    """
    student_patches, teacher_patches = _align_patch_grids(student_patches, teacher_patches)
    n_tokens = student_patches.shape[1]
    mask = _align_token_mask(masked_positions, n_tokens)
    if padding_mask is not None:
        pm_flat = _align_token_mask(padding_mask, n_tokens)
        mask = mask & pm_flat

    return cosine_loss(student_patches, teacher_patches, mask=mask)


def img_ema_global_loss(
    student_global: Tensor,    # (B, D) native hidden
    ema_global: Tensor,        # (B, D) native hidden, stopgrad applied by caller
) -> Tensor:
    """Auxiliary cosine: live student global vs EMA teacher global (native D4)."""
    return cosine_loss(student_global, ema_global)


def img_ema_patch_loss(
    student_patches: Tensor,           # (B, N_s, D_s) native F1
    ema_patches: Tensor,               # (B, N_t, D_s) native F1, stopgrad
    masked_positions: Optional[Tensor] = None,  # (B, ph, pw) or (B, N); True = predict
    padding_mask: Optional[Tensor] = None,      # (B, ph, pw) bool, True = real patch
) -> Tensor:
    """
    Patch-level EMA self-distillation in native hidden space.

    When masked_positions is given, loss is at masked AND valid (padding) tokens.
    Otherwise all valid non-padding tokens are used.
    """
    n_tokens = student_patches.shape[1]
    if student_patches.shape[1] != ema_patches.shape[1]:
        ema_patches = _interpolate_tokens(ema_patches, n_tokens)

    if masked_positions is not None:
        mask = _align_token_mask(masked_positions, n_tokens)
        if padding_mask is not None:
            mask = mask & _align_token_mask(padding_mask, n_tokens)
    elif padding_mask is not None:
        mask = _align_token_mask(padding_mask, n_tokens)
    else:
        mask = None

    return cosine_loss(student_patches, ema_patches, mask=mask)


def img_proto_loss(
    student_tokens: Tensor,    # (B, N, D) or (B, D)
    teacher_tokens: Tensor,    # (B, N, D) or (B, D)  — stopgrad applied by caller
    prototypes: Tensor,        # (K, D)
    temperature: float = 0.1,  # student softmax temperature τ
    sinkhorn_eps: float = 0.05, # teacher Sinkhorn temperature ε  (< τ → sharper target)
) -> Tensor:
    """
    Asymmetric SwAV prototype loss: balanced Sinkhorn target from the teacher,
    log-softmax prediction from the student.

    Teacher  : Q = Sinkhorn( exp(cos_sim / sinkhorn_eps) )   [stop-gradient]
    Student  : p = softmax(  cos_sim / temperature       )
    Loss     : L = -Σ Q · log(p)

    Delegates to swav_proto_loss_from_tokens for DDP-safe global Sinkhorn.
    Returns 0.0 when B_global < K (video micro-batch too small for balanced SK).
    """
    from .proto_loss import swav_proto_loss_from_tokens

    return swav_proto_loss_from_tokens(
        teacher_tokens,
        student_tokens,
        prototypes,
        temperature=temperature,
        sinkhorn_eps=sinkhorn_eps,
    )


# ---------------------------------------------------------------------------
# Video-side losses
# ---------------------------------------------------------------------------

def vid_global_distill_loss(
    student_clip_global: Tensor,   # (B, D_s)
    teacher_clip_global: Tensor,   # (B, D_t_proj)
) -> Tensor:
    """Cosine distance: student clip global vs frozen V-JEPA clip global."""
    return cosine_loss(student_clip_global, teacher_clip_global)


def vid_tube_prediction_loss(
    student_pred: Tensor,          # (B, N_masked, D_s)
    teacher_at_masked: Tensor,     # (B, N_masked, D_t_proj)
    valid_mask: Optional[Tensor] = None,  # (B, N_masked) bool
) -> Tensor:
    """
    V-JEPA-style masked tube prediction: student predicts teacher latents
    at masked spatiotemporal positions.
    """
    return cosine_loss(student_pred, teacher_at_masked, mask=valid_mask)


def _video_tube_target_mask(
    tube_mask: Tensor,
    n_tokens_per_frame: int,
    padding_mask: Optional[Tensor] = None,
    valid_frames: Optional[Tensor] = None,
) -> Tensor:
    """
    Flat bool mask (B, T*n_tokens) at student F4 resolution.

    Max-pools stride-16 ``tube_mask`` to F4, intersects valid/padding regions,
    then nearest-neighbour resizes to the student's per-frame token count when
    Hiera geometry differs from the collator grid (non-square crops, etc.).
    """
    B, T, ph16, pw16 = tube_mask.shape
    m = tube_mask.reshape(B * T, 1, ph16, pw16).float()
    tube_f4 = F.max_pool2d(m, kernel_size=2, stride=2)
    ph32, pw32 = int(tube_f4.shape[-2]), int(tube_f4.shape[-1])
    n_f4 = ph32 * pw32
    tube_f4 = tube_f4.reshape(B, T, n_f4).bool()

    real = torch.ones(B, T, n_f4, dtype=torch.bool, device=tube_mask.device)
    if valid_frames is not None:
        real &= valid_frames[:, :, None]
    if padding_mask is not None:
        pm = padding_mask.float().unsqueeze(1)
        if pm.shape[-2] != ph16 or pm.shape[-1] != pw16:
            pm = F.interpolate(pm, size=(ph16, pw16), mode="nearest")
        pm32 = F.max_pool2d(pm, kernel_size=2, stride=2).squeeze(1)
        pm_flat = _align_token_mask(pm32, n_f4)
        real &= pm_flat[:, None, :]

    flat = (tube_f4 & real).reshape(B, T * n_f4)
    return _align_token_mask(flat, T * n_tokens_per_frame)


def vid_tube_masked_distill_loss(
    student_tubes: Tensor,              # (B, T, N_s, D)
    teacher_tubes_flat: Tensor,         # (B, N_teacher, D) V-JEPA tubelet tokens
    tube_mask: Tensor,                  # (B, T, ph16, pw16) True = masked target
    padding_mask: Optional[Tensor] = None,   # (B, ph16, pw16) teacher stride
    valid_frames: Optional[Tensor] = None,
    tubelet_size: int = 2,
) -> Tensor:
    """
    Masked tube distillation on the student F4 grid.

    ``tube_mask`` is at the V-JEPA patch stride (16px).  Student ``tube_proj``
    lives on the coarser F4 grid (32px), so the mask is max-pooled and teacher
    tokens are spatially downsampled + repeated across tubelet frames.
    """
    B, T, N_s, D = student_tubes.shape
    ph16, pw16 = int(tube_mask.shape[2]), int(tube_mask.shape[3])

    mask_flat = _video_tube_target_mask(
        tube_mask, N_s, padding_mask=padding_mask, valid_frames=valid_frames,
    )
    if not mask_flat.any():
        return student_tubes.new_tensor(0.0)

    s_flat = student_tubes.reshape(B, T * N_s, D)

    N16 = ph16 * pw16
    n_teacher = teacher_tubes_flat.shape[1]
    if N16 <= 0 or n_teacher < N16:
        return student_tubes.new_tensor(0.0)
    T_t = n_teacher // N16
    tt = teacher_tubes_flat[:, : T_t * N16].reshape(B, T_t, N16, D)
    tt_down = _downsample_tokens(tt.reshape(B * T_t, N16, D), N_s).reshape(B, T_t, N_s, D)

    reps = max(1, tubelet_size)
    tt_exp = tt_down.unsqueeze(2).expand(-1, -1, reps, -1, -1).reshape(B, T_t * reps, N_s, D)
    if tt_exp.shape[1] < T:
        pad = tt_exp[:, -1:].expand(-1, T - tt_exp.shape[1], -1, -1)
        tt_exp = torch.cat([tt_exp, pad], dim=1)
    t_flat = tt_exp[:, :T].reshape(B, T * N_s, D)

    return cosine_loss(s_flat, t_flat, mask=mask_flat)


def vid_temporal_consistency(
    frame_tokens: Tensor,          # (B, T, D)  per-frame embeddings
    padding_mask: Optional[Tensor] = None,  # (B, ph, pw)
    valid_frames: Optional[Tensor] = None,  # (B, T) bool
) -> Tensor:
    """
    Smooth temporal consistency: consecutive frames should have similar
    global representations when there is no abrupt probe motion.

    L = mean cosine_dist(f_t, f_{t+1}) over valid adjacent pairs.
    """
    if frame_tokens.shape[1] <= 1:
        return frame_tokens.new_tensor(0.0)

    t1 = frame_tokens[:, :-1]   # (B, T-1, D)
    t2 = frame_tokens[:, 1:]    # (B, T-1, D)

    pair_mask = None
    if valid_frames is not None:
        pair_mask = valid_frames[:, :-1] & valid_frames[:, 1:]  # (B, T-1)

    return cosine_loss(t1, t2, mask=pair_mask)


def vid_proto_loss(
    student_tubes: Tensor,    # (B, N, D)
    teacher_tubes: Tensor,    # (B, N_t, D)  V-JEPA tube_proj; stopgrad by caller
    prototypes: Tensor,       # (K, D)
    temperature: float = 0.1,
    sinkhorn_eps: float = 0.05,
    queue=None,               # optional ProtoQueue for small-batch video
) -> tuple:
    """
    V-JEPA-aligned prototype loss on mean-pooled tube tokens.

    Without queue (queue=None):
        Returns (loss, None).  Falls back to 0.0 when B_global < K.

    With queue (ProtoQueue instance):
        Runs Sinkhorn on queue + current batch, which allows the loss to fire
        even when video_batch_size × n_ranks < K.  Returns (loss, teacher_logits)
        so the caller can enqueue teacher_logits AFTER the backward pass.

    In both cases the first return value is the scalar loss tensor.
    """
    from .proto_loss import swav_proto_loss_from_tokens, swav_proto_loss_with_queue, _is_dist
    import torch.distributed as dist
    import torch.nn.functional as F

    def _pool_and_norm(t: Tensor) -> Tensor:
        v = t.float()
        if v.dim() == 3:
            v = v.mean(1)
        return F.normalize(v, dim=-1)

    if queue is None:
        loss = img_proto_loss(
            student_tubes, teacher_tubes, prototypes,
            temperature=temperature, sinkhorn_eps=sinkhorn_eps,
        )
        return loss, None

    # Queue path — compute logits manually to return teacher_logits for enqueueing
    proto = F.normalize(prototypes.float(), dim=-1)
    feat_t = _pool_and_norm(teacher_tubes)   # (B, D)
    feat_s = _pool_and_norm(student_tubes)   # (B, D)
    teacher_logits = (feat_t @ proto.T) / sinkhorn_eps   # (B, K)
    student_logits = (feat_s @ proto.T) / temperature    # (B, K)

    loss = swav_proto_loss_with_queue(
        teacher_logits.detach(),   # queue path uses frozen-teacher logits, no grad needed
        student_logits,
        queue,
    )
    return loss, teacher_logits.detach()


def prototype_assignment_stats(
    tokens: Tensor,           # (B, N, D) or (B, D)
    prototypes: Tensor,       # (K, D)
    temperature: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Per-batch prototype assignment entropy and max probability (for logging)."""
    with torch.no_grad():
        pooled = tokens.mean(1) if tokens.dim() == 3 else tokens
        p = prototype_assign(pooled, prototypes, temperature, use_sinkhorn=False)
        entropy = -(p * (p + 1e-8).log()).sum(-1)
        max_prob = p.max(dim=-1).values
    return entropy.mean(), max_prob.mean()


# ---------------------------------------------------------------------------
# Paired-batch losses (Stage 3)
# ---------------------------------------------------------------------------

def fused_distill_loss(
    student_tubes: Tensor,     # (B, N_vid, D_s)
    fused_target: Tensor,      # (B, N_vid, D_align)  z_fused.detach()
    padding_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Main cross-distillation: student video tubes match the fused
    DINO+V-JEPA teacher target.

    fused_target must already be detached by the caller (FusionTargetBuilder
    returns z_fused.detach()).
    """
    mask = None
    if padding_mask is not None:
        mask = padding_mask.flatten(1)
        if mask.shape[1] != student_tubes.shape[1]:
            mask = None   # skip masking if grid sizes mismatch

    if student_tubes.shape[1] != fused_target.shape[1]:
        fused_target = _interpolate_tokens(
            fused_target, student_tubes.shape[1], tgt_pmask=padding_mask,
        )

    return cosine_loss(student_tubes, fused_target, mask=mask)


def preservation_loss(
    z_fused: Tensor,    # (B, N_vid, D_align)  NOT detached — has grad
    z_vid: Tensor,      # (B, N_vid, D_align)  stopgrad
) -> Tensor:
    """
    Prevent the DINO semantic injection from overwriting V-JEPA temporal
    structure.  Anchors z_fused close to the original z_vid.

    L = cosine_dist(z_fused, stopgrad(z_vid))
    """
    return cosine_loss(z_fused, z_vid.detach())


def frame_clip_consistency(
    student_frame_tokens: Tensor,  # (B, N, D_s)  student(frame_t as T=1)
    student_clip_frame_tokens: Tensor,  # (B, N, D_s) student(clip)[frame_t slot]
    padding_mask: Optional[Tensor] = None,  # (B, ph, pw)
) -> Tensor:
    """
    Encourage consistent representation whether a frame is seen alone
    or in the context of a clip.

    student(frame_t as T=1).dense ≈ student(clip)[frame_t tokens]

    When token grids differ (e.g. residual stride mismatch), clip tokens
    are bilinearly interpolated to the frame grid before comparison.
    """
    clip_tokens = student_clip_frame_tokens.detach()
    if student_frame_tokens.shape[1] != clip_tokens.shape[1]:
        clip_tokens = _interpolate_tokens(
            clip_tokens, student_frame_tokens.shape[1], tgt_pmask=padding_mask,
        )

    mask = _align_token_mask(padding_mask, student_frame_tokens.shape[1])

    return cosine_loss(
        student_frame_tokens,
        clip_tokens,
        mask=mask,
    )


# ---------------------------------------------------------------------------
# Utility: token interpolation when grids differ
# ---------------------------------------------------------------------------

def _align_token_mask(mask: Optional[Tensor], n_tokens: int) -> Optional[Tensor]:
    """
    Resize a (B, ph, pw) or (B, N) bool mask to (B, n_tokens).

    Fallback for residual grid mismatch (e.g. SAM2 F1 geometry vs stride-4
    input masks).  Prefer ``s_out['pmasks'][0]`` from the student backbone.
    """
    if mask is None:
        return None
    flat = mask.flatten(1) if mask.dim() > 2 else mask
    if flat.shape[1] == n_tokens:
        return flat.bool()
    aligned = F.interpolate(flat.float().unsqueeze(1), size=n_tokens, mode="nearest")
    return aligned.squeeze(1).bool()


def _factor_token_grid(
    n_tokens: int,
    pmask: Optional[Tensor] = None,
) -> tuple[int, int]:
    """Return (ph, pw) with ph * pw == n_tokens for spatial token layouts."""
    if pmask is not None and pmask.dim() >= 2:
        ph, pw = int(pmask.shape[-2]), int(pmask.shape[-1])
        if ph * pw == n_tokens:
            return ph, pw
    n_tokens = max(1, n_tokens)
    ph = max(1, int(n_tokens ** 0.5))
    while ph > 1 and n_tokens % ph != 0:
        ph -= 1
    return ph, n_tokens // ph


def _align_patch_grids(
    student_patches: Tensor,
    teacher_patches: Tensor,
) -> tuple[Tensor, Tensor]:
    """Match patch-token counts; downsample the finer grid when possible."""
    n_s = student_patches.shape[1]
    n_t = teacher_patches.shape[1]
    if n_s == n_t:
        return student_patches, teacher_patches
    if n_s > n_t:
        return _downsample_tokens(student_patches, n_t), teacher_patches
    return student_patches, _interpolate_tokens(teacher_patches, n_s)


def _downsample_tokens(
    tokens: Tensor,
    target_n: int,
    tgt_pmask: Optional[Tensor] = None,
) -> Tensor:
    """Area-pool token sequence to target_n tokens (for fine → coarse grids)."""
    B, N_src, D = tokens.shape
    if N_src == target_n:
        return tokens

    src_ph, src_pw = _factor_token_grid(N_src)
    tgt_ph, tgt_pw = _factor_token_grid(target_n, tgt_pmask)

    feat = tokens.reshape(B, src_ph, src_pw, D).permute(0, 3, 1, 2)
    feat = F.interpolate(feat, size=(tgt_ph, tgt_pw), mode="area")
    return feat.permute(0, 2, 3, 1).reshape(B, tgt_ph * tgt_pw, D)


def _interpolate_tokens(
    tokens: Tensor,
    target_n: int,
    tgt_pmask: Optional[Tensor] = None,
) -> Tensor:
    """
    Bilinearly interpolate token sequence to target_n tokens.

    Parameters
    ----------
    tokens   : (B, N_src, D)
    target_n : target number of tokens

    Returns
    -------
    (B, target_n, D)
    """
    B, N_src, D = tokens.shape
    if N_src == target_n:
        return tokens

    src_ph, src_pw = _factor_token_grid(N_src)
    tgt_ph, tgt_pw = _factor_token_grid(target_n, tgt_pmask)

    feat = tokens.reshape(B, src_ph, src_pw, D).permute(0, 3, 1, 2)  # (B, D, h, w)
    feat = F.interpolate(
        feat, size=(tgt_ph, tgt_pw), mode="bilinear", align_corners=False
    )
    return feat.permute(0, 2, 3, 1).reshape(B, tgt_ph * tgt_pw, D)
