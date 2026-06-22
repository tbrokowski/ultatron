"""
train/student_phase_steps.py  ·  Phase step functions for the single-student pipeline
======================================================================================

Contract (identical to existing phase_steps.py)
------------------------------------------------
  - Pure functions: no optimizer.step(), no logging, no checkpointing
  - Accept tensors + nn.Modules, return a dict of loss scalars + "loss" tensor
  - Caller does: loss.backward(); clip_grad; optimizer.step(); ema_update()

Four stages (student pretrain)
------------------------------
student_stage1_step            DINO warm-start (image-heavy)
student_stage2_step            V-JEPA warm-start (mixed image + video)
student_stage3_step            Cross-modal fusion (paired + image + video)
student_stage4_divergence_step EMA self-distillation divergence
student_stage5_supervised_step Optional supervised heads (not in default pretrain)

Loss weights (lam dict keys)
-----------------------------
  lam_global    img_global_distill / vid_global_distill
  lam_patch     img_patch_distill (valid & ~freq-masked tokens)
  lam_masked    img_masked_semantic (valid & freq-masked tokens)
  lam_proto     img_proto / vid_proto
  lam_tube      vid_tube_prediction
  lam_temp      vid_temporal_consistency
  lam_fused     fused_distill (Stage 3 paired; warm-up from 0.1 → 0.5)
  lam_preserve  preservation_loss (Stage 3 paired)
  lam_fc        frame_clip_consistency
  lam_ema_max   peak weight for EMA self-distillation (ramps from lam_ema_floor)
  lam_ema_max_by_stage  optional {stage: weight} overrides (e.g. {2: 1.0, 3: 1.0})
  lam_ema_floor optional early floor (default 0)
  lam_ema_warmup_steps  steps to ramp EMA weight to lam_ema_max (per-stage peak)
  lam_ema_divergence_peak  peak EMA weight in stage 4 (default 0.8)
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.checkpoint import checkpoint

from data.pipeline.student_datamodule import (
    SOURCE_MASK_PATCH_STRIDE,
    downsample_mask_grid,
)
from models.losses.student_losses import (
    _align_token_mask,
    img_global_distill_loss,
    img_patch_distill_loss,
    img_masked_semantic_loss,
    img_proto_loss,
    img_ema_global_loss,
    img_ema_patch_loss,
    vid_global_distill_loss,
    vid_tube_masked_distill_loss,
    _video_tube_target_mask,
    vid_temporal_consistency,
    vid_proto_loss,
    prototype_assignment_stats,
    fused_distill_loss,
    preservation_loss,
    frame_clip_consistency,
)


# ---------------------------------------------------------------------------
# Padding-mask helpers (same pattern as existing phase_steps.py)
# ---------------------------------------------------------------------------

def _pmask(batch: dict, key: str = "padding_mask") -> Optional[Tensor]:
    return batch.get(key)


def _release_teacher_gpu(teacher: nn.Module) -> None:
    """Park a frozen teacher on CPU (float32) to free GPU memory before student forwards."""
    if teacher is None:
        return
    try:
        p = next(teacher.parameters())
    except StopIteration:
        return
    if p.device.type != "cuda":
        return
    teacher.cpu().float()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _paired_student_video_forward(
    student,
    clip: Tensor,
    padding_mask: Optional[Tensor],
) -> dict:
    """Student video forward for paired batches (matches stage-2 video memory profile)."""
    return student(clip, padding_mask=padding_mask, with_patch_proj=False)


def _forward_frozen_teacher(
    teacher: nn.Module,
    out_device: torch.device,
    forward: Callable[[torch.device], dict],
) -> dict:
    """
    Run a frozen teacher on its current device and return outputs on ``out_device``.

    Lets stage-3 keep DINO on CPU while student / V-JEPA tensors stay on GPU.
    """
    t_dev = next(teacher.parameters()).device
    with torch.no_grad():
        out = forward(t_dev)
        if t_dev != out_device:
            out = {
                k: v.to(out_device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in out.items()
            }
    return out


def _slice_batch_dim0(batch: dict, idx: int) -> dict:
    """Slice a collated batch to a single sample (dim-0)."""
    out: dict = {}
    for key, val in batch.items():
        if key == "alignment_pairs":
            pairs = val or []
            picked = [
                p for p in pairs
                if p.img_batch_idx == idx and p.vid_batch_idx == idx
            ]
            out[key] = picked or ([pairs[idx]] if idx < len(pairs) else [])
        elif isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] > 1:
            out[key] = val[idx : idx + 1]
        elif isinstance(val, list) and len(val) > 1:
            out[key] = [val[idx]]
        else:
            out[key] = val
    return out


def _paired_anchor_t(batch: dict, t_default: int) -> int:
    """Temporal slot of the paired image frame inside the video clip."""
    pairs = batch.get("alignment_pairs") or []
    if pairs:
        return int(pairs[0].frame_offset)
    offsets = batch.get("frame_offsets")
    if offsets is not None:
        if torch.is_tensor(offsets):
            return int(offsets[0].item())
        return int(offsets[0])
    return t_default


def _flat_pmask(batch: dict, key: str = "padding_mask") -> Optional[Tensor]:
    pm = batch.get(key)
    if pm is None:
        return None
    return pm.flatten(1)  # (B, N)


def _mask_stride_factor(batch: dict) -> int:
    teacher = batch.get("teacher_mask_stride", SOURCE_MASK_PATCH_STRIDE)
    student = batch.get("student_mask_stride", 4)
    return max(1, teacher // student)


def _student_input_pmask(batch: dict, key: str = "global_pmasks") -> Optional[Tensor]:
    """Stride-4 (student input) padding mask, shape (B, ph, pw)."""
    pm = _pmask(batch, key)
    if pm is not None and pm.dim() == 4:
        pm = pm[:, 0]
    return pm


def _teacher_image_pmask(
    batch: dict,
    student_pmask: Optional[Tensor],
    *,
    key: str = "global_pmasks",
) -> Optional[Tensor]:
    """Stride-16 padding mask for DINO (B, ph, pw)."""
    s16_key = f"{key}_s16"
    pm = batch.get(s16_key)
    if pm is not None:
        return pm[:, 0] if pm.dim() == 4 else pm
    if student_pmask is not None:
        return downsample_mask_grid(student_pmask, _mask_stride_factor(batch))
    if key == "padding_masks":
        pm = batch.get("padding_masks_s16")
        if pm is not None:
            return pm[:, 0] if pm.dim() == 4 else pm
    return None


def _align_patch_scores(
    scores: Tensor,          # (B, N_src)
    target_n: int,
    src_pmask: Optional[Tensor] = None,
    tgt_pmask: Optional[Tensor] = None,
) -> Tensor:
    """Bilinearly resample per-patch saliency/hardness to student token count."""
    B, n_src = scores.shape
    if n_src == target_n:
        out = scores
    else:
        src_ph, src_pw = _factor_token_grid(n_src, src_pmask)
        tgt_ph, tgt_pw = _factor_token_grid(target_n, tgt_pmask)
        grid = scores.reshape(B, 1, src_ph, src_pw)
        grid = F.interpolate(grid, size=(tgt_ph, tgt_pw), mode="bilinear", align_corners=False)
        out = grid.reshape(B, tgt_ph * tgt_pw)
    if tgt_pmask is not None and tgt_pmask.shape[-2] * tgt_pmask.shape[-1] == out.shape[1]:
        valid = tgt_pmask.reshape(B, -1)
        out = out.masked_fill(~valid, 0.0)
    return out


def _dino_teacher_saliency(
    t_img: dict,
    t_patch_proj: Tensor,
    target_n: int,
    dino_pmask: Optional[Tensor],
    student_pmask: Optional[Tensor],
) -> Tensor:
    """
    OpenUS-style teacher saliency A_t from DINO CLS→patch attention.

    Falls back to projected patch L2 norm if attentions were not requested.
    """
    sal = t_img.get("cls_patch_attention")
    if sal is None:
        sal = t_patch_proj.norm(dim=-1)
    else:
        sal = sal.float()
    if (
        dino_pmask is not None
        and dino_pmask.shape[-2] * dino_pmask.shape[-1] == sal.shape[1]
    ):
        sal = sal.masked_fill(~dino_pmask.reshape(sal.shape[0], -1), 0.0)
    return _align_patch_scores(sal, target_n, dino_pmask, student_pmask)


def _teacher_video_pmask(
    batch: dict,
    student_pmask: Optional[Tensor],
) -> Optional[Tensor]:
    pm = batch.get("padding_masks_s16")
    if pm is not None:
        return pm
    if student_pmask is not None:
        return downsample_mask_grid(student_pmask, _mask_stride_factor(batch))
    return None


def _factor_token_grid(
    n_tokens: int,
    pmask: Optional[Tensor] = None,
) -> tuple[int, int]:
    """Return (ph, pw) with ph * pw == n_tokens for tube/patch grid mapping."""
    if pmask is not None and pmask.dim() >= 2:
        ph, pw = int(pmask.shape[-2]), int(pmask.shape[-1])
        if ph * pw == n_tokens:
            return ph, pw
    n_tokens = max(1, n_tokens)
    ph = max(1, int(n_tokens ** 0.5))
    while ph > 1 and n_tokens % ph != 0:
        ph -= 1
    return ph, n_tokens // ph


def _teacher_tubelet_temporal_len(T_frames: int, vjepa_teacher) -> int:
    """V-JEPA teacher ``tube_tokens`` are at tubelet resolution, not frame resolution."""
    tubelet_size = getattr(getattr(vjepa_teacher, "backbone", None), "tubelet_size", 2)
    return max(1, T_frames // max(1, tubelet_size))


def _teacher_tubes_spatial_mean(
    tube_tokens: Tensor,
    B: int,
    T_frames: int,
    vjepa_teacher,
) -> Tensor:
    """Collapse teacher tubelet tokens to per-spatial-position means, shape (B, n_spat, D)."""
    T_tok = _teacher_tubelet_temporal_len(T_frames, vjepa_teacher)
    n_spat = tube_tokens.shape[1] // max(1, T_tok)
    return tube_tokens.reshape(B, T_tok, n_spat, tube_tokens.shape[-1]).mean(1)


def _student_f1_flat_pmask(s_out: dict) -> Optional[Tensor]:
    """F1 valid-token mask aligned to the student backbone, shape (B, N)."""
    pmasks = s_out.get("pmasks")
    if not pmasks or pmasks[0] is None:
        return None
    return pmasks[0].reshape(pmasks[0].shape[0], -1)


def _student_f1_grid_pmask(s_out: dict) -> Optional[Tensor]:
    """F1 valid-token mask, shape (B, ph, pw)."""
    pmasks = s_out.get("pmasks")
    if not pmasks or pmasks[0] is None:
        return None
    return pmasks[0]


def _student_f4_grid_pmask(s_out: dict) -> Optional[Tensor]:
    """F4 valid-token mask, shape (B, ph, pw)."""
    pmasks = s_out.get("pmasks")
    if not pmasks or len(pmasks) < 4 or pmasks[3] is None:
        return None
    return pmasks[3]


def _student_global(s_out: dict) -> Tensor:
    return s_out.get("global_proj", s_out["global"])


def _student_patches(s_out: dict, t: int = 0) -> Tensor:
    if "patch_proj" in s_out:
        return s_out["patch_proj"][:, t]
    return s_out["F1"][:, t]


def _student_tubes_at_t(s_out: dict, t: int = 0) -> Tensor:
    """F4 / tube_proj tokens for frame t — used for frame-clip consistency."""
    if "tube_proj" in s_out:
        return s_out["tube_proj"][:, t]
    return s_out["F4"][:, t]


def _dino_patch_masks(
    s_out: dict,
    patch_masks: Optional[Tensor],
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Disjoint flat masks for iBOT-style DINO patch losses.

    Returns (unmasked, masked) as (B, N) bool:
      unmasked = valid F1 tokens that were NOT freq-masked
      masked   = valid F1 tokens that were freq-masked (None if no patch_masks)
    """
    n_tokens = _student_patches(s_out, t=0).shape[1]
    valid = _student_f1_flat_pmask(s_out)
    if patch_masks is None:
        return valid, None

    masked = _align_token_mask(patch_masks, n_tokens)
    if valid is not None:
        masked = masked & valid
        unmasked = valid & ~masked
    else:
        unmasked = ~masked
    return unmasked, masked


def _student_tubes_flat(s_out: dict, B: int) -> Tensor:
    if "tube_proj" in s_out:
        tubes = s_out["tube_proj"]
        return tubes.reshape(B, -1, tubes.shape[-1])
    f4 = s_out["F4"]
    return f4.reshape(B, -1, f4.shape[-1])


def _teacher_tube_mask(batch: dict) -> Optional[Tensor]:
    """Stride-16 tube mask (B, T, ph, pw); True = masked target."""
    mask = batch.get("tube_masks_s16")
    if mask is not None:
        return mask
    return batch.get("tube_masks")


def _teacher_padding_mask(batch: dict, fallback: Optional[Tensor]) -> Optional[Tensor]:
    """Stride-16 padding mask for V-JEPA / tube_mask alignment."""
    pm = batch.get("padding_masks_s16")
    if pm is not None:
        return pm
    return fallback


def _tube_mask_f4_grid(
    tube_mask: Tensor,
    n_tokens: int,
    padding_mask: Optional[Tensor],
    valid_frames: Optional[Tensor],
) -> Tensor:
    """Downsample stride-16 tube mask to student F4 token grid, shape (B, T*N)."""
    B, T, _, _ = tube_mask.shape
    return _video_tube_target_mask(
        tube_mask, n_tokens, padding_mask=padding_mask, valid_frames=valid_frames,
    )


def _video_ssl_losses(
    batch: dict,
    student,
    ema_student,
    vjepa_teacher,
    proto_head,
    lam: dict,
    global_step: int = 0,
    tubelet_size: int = 2,
    proto_queue=None,          # optional ProtoQueue for small-batch video proto loss
    stage: int = 2,
) -> dict:
    """
    Shared video SSL losses for stage 2 and stage 3 video batches.

    Student sees ``visible_clips`` (masked); EMA teacher sees ``full_clips``
    (clean).  V-JEPA teacher sees ``full_clips`` for distillation targets.
    """
    full_clips    = batch["full_clips"]
    visible_clips = batch.get("visible_clips", full_clips)
    tube_mask     = _teacher_tube_mask(batch)
    pmask_student = _pmask(batch, "padding_masks")
    pmask_vjepa   = _teacher_video_pmask(batch, pmask_student)
    pmask_s16     = _teacher_padding_mask(batch, pmask_vjepa)
    valid_frames  = batch.get("valid_frames")

    B, T = full_clips.shape[:2]
    _vid_kw = dict(padding_mask=pmask_student, with_patch_proj=False)

    with torch.no_grad():
        t_vid = vjepa_teacher(
            full_clips,
            tube_mask=None,
            padding_mask=pmask_vjepa,
            valid_frames=valid_frames,
        )
        t_ema_out = ema_student(full_clips, **_vid_kw)

    s_out = student(visible_clips, **_vid_kw)
    s_global = _student_global(s_out)

    if "tube_proj" in s_out:
        s_frame_globals = s_out["tube_proj"].mean(dim=2)
        s_tubes = s_out["tube_proj"]
    else:
        s_frame_globals = s_out["F4"].mean(dim=2)
        s_tubes = s_out["F4"]

    L_vid_global = vid_global_distill_loss(s_global, t_vid["clip_proj"])

    if tube_mask is not None and "tube_proj" in t_vid:
        L_tube = vid_tube_masked_distill_loss(
            s_tubes,
            t_vid["tube_proj"],
            tube_mask,
            padding_mask=pmask_s16,
            valid_frames=valid_frames,
            tubelet_size=tubelet_size,
        )
    else:
        L_tube = s_global.new_tensor(0.0)

    L_temp = vid_temporal_consistency(
        s_frame_globals, padding_mask=pmask_student, valid_frames=valid_frames,
    )

    proto_entropy = float("nan")
    proto_max_prob = float("nan")
    # Video prototype loss requires a queue when video_batch_size << n_prototypes.
    # Without a queue the Sinkhorn assignment is degenerate, so we skip it entirely.
    # Enable by passing a ProtoQueue (proto_queue_size > n_prototypes in config).
    if proto_queue is not None and "tube_proj" in t_vid:
        s_tubes_flat = _student_tubes_flat(s_out, B)
        t_tubes_flat = t_vid["tube_proj"].reshape(B, -1, t_vid["tube_proj"].shape[-1])
        L_proto, teacher_logits_for_queue = vid_proto_loss(
            s_tubes_flat,
            t_tubes_flat,
            proto_head.prototypes,
            queue=proto_queue,
        )
        if teacher_logits_for_queue is not None:
            proto_queue.enqueue(teacher_logits_for_queue)
        with torch.no_grad():
            ent, mx = prototype_assignment_stats(s_tubes_flat, proto_head.prototypes)
            proto_entropy = float(ent.item())
            proto_max_prob = float(mx.item())
    else:
        L_proto = s_global.new_tensor(0.0)

    lam_ema_eff = _lam_ema_eff(lam, global_step, stage=stage)
    L_ema_global = img_ema_global_loss(s_out["global"], t_ema_out["global"])
    n_f4 = s_out["F4"].shape[2]
    tube_flat = (
        _tube_mask_f4_grid(tube_mask, n_f4, pmask_s16, valid_frames)
        if tube_mask is not None else None
    )
    s_tube = s_out.get("tube_proj", s_out["F4"])
    t_tube = t_ema_out.get("tube_proj", t_ema_out["F4"])
    L_ema_patch = img_ema_patch_loss(
        s_tube.reshape(B, T * n_f4, -1),
        t_tube.reshape(B, T * n_f4, -1),
        masked_positions=tube_flat,
        padding_mask=_student_f4_grid_pmask(s_out),
    )
    L_ema = L_ema_global + L_ema_patch

    loss = (
        lam.get("lam_global", 1.0) * L_vid_global
      + lam.get("lam_tube",   1.0) * L_tube
      + lam.get("lam_temp",   0.5) * L_temp
      + lam.get("lam_proto",  0.5) * L_proto
      + lam_ema_eff               * L_ema
    )

    return {
        "loss":             loss,
        "loss_vid_global":  L_vid_global.item(),
        "loss_tube":        L_tube.item(),
        "loss_temporal":    L_temp.item(),
        "loss_proto":       L_proto.item(),
        "loss_ema":         L_ema.item(),
        "loss_ema_global":  L_ema_global.item(),
        "loss_ema_patch":   L_ema_patch.item(),
        "proto_entropy":    proto_entropy,
        "proto_max_prob":   proto_max_prob,
        "lam_ema_eff":      lam_ema_eff,
        "sample_type":      "video",
    }


def _student_crop_pmask(
    batch: dict,
    crop_idx: int,
    key: str = "global_pmasks",
) -> Optional[Tensor]:
    """Stride-4 padding mask for global_crops[:, crop_idx], shape (B, ph, pw)."""
    pm = batch.get(key)
    if pm is None:
        return None
    if pm.dim() == 4:
        return pm[:, crop_idx]
    return pm if crop_idx == 0 else None


def _lam_ema_peak(lam: dict, stage: int) -> float:
    """Per-curriculum-stage peak EMA weight (falls back to lam_ema_max)."""
    by_stage = lam.get("lam_ema_max_by_stage") or {}
    if stage in by_stage:
        return float(by_stage[stage])
    return float(lam.get("lam_ema_max", 0.15))


def _lam_ema_eff(
    lam: dict,
    global_step: int,
    *,
    stage: int = 0,
    stage4_start: int = 0,
    stage4_end: int = 0,
) -> float:
    """Ramp EMA loss weight: warmup in stages 1–3; divergence ramp in stage 4."""
    if stage == 4 and stage4_end > stage4_start:
        return _stage4_ema_scale(lam, global_step, stage4_start, stage4_end)
    peak = _lam_ema_peak(lam, stage) if stage > 0 else float(lam.get("lam_ema_max", 0.15))
    floor = lam.get("lam_ema_floor", 0.0)
    warmup = lam.get("lam_ema_warmup_steps", 10_000)
    if peak <= 0:
        return 0.0
    t = min(1.0, global_step / max(1, warmup))
    return floor + t * (peak - floor)


def _stage_bounds_from_fracs(fracs: List[float], total: int) -> List[int]:
    """Return cumulative step boundaries [0, end_s1, end_s2, …, total]."""
    bounds = [0]
    for f in fracs:
        bounds.append(bounds[-1] + int(total * f))
    bounds[-1] = total
    return bounds


def _cosine_ramp(step: int, start: int, end: int, v0: float, v1: float) -> float:
    """Cosine interpolate between v0 (at start) and v1 (at end)."""
    if end <= start:
        return v1
    if step <= start:
        return v0
    if step >= end:
        return v1
    t = (step - start) / (end - start)
    return v0 + 0.5 * (1.0 - math.cos(math.pi * t)) * (v1 - v0)


def _stage4_ema_scale(lam: dict, step: int, s4_start: int, s4_end: int) -> float:
    """Ramp EMA weight from stage-3 peak to lam_ema_divergence_peak over stage 4."""
    v0 = _lam_ema_peak(lam, 3)
    v1 = lam.get("lam_ema_divergence_peak", 0.8)
    return _cosine_ramp(step, s4_start, s4_end, v0, v1)


def _teacher_scale_for_stage(stage: int) -> float:
    """Frozen teachers are active in stages 1–3 only."""
    return 0.0 if stage == 4 else 1.0


def _ensure_image_batch(batch: dict) -> dict:
    """
    Normalize a batch for image-only stage steps.

    Stage-1 mixes may include video samples (single-frame warm-start).  When
    ``global_crops`` is absent, synthesize it from the middle frame of
    ``full_clips``.  Stage 1 needs two global crops (masked student view +
    clean teacher view); duplicate the extracted frame when only one is available.
    """
    if "global_crops" in batch and batch["global_crops"].shape[1] >= 2:
        return batch

    out = dict(batch)
    if "global_crops" in batch:
        frame = batch["global_crops"][:, 0]
        src_pmask = out.get("global_pmasks")
    elif "full_clips" in batch:
        clips = out["full_clips"]
        frame = clips[:, clips.shape[1] // 2]
        src_pmask = out.get("padding_masks")
        out["sample_type"] = "image"
    else:
        raise KeyError(
            "Image step requires 'global_crops' or 'full_clips'; "
            f"got keys: {sorted(batch)}"
        )

    out["global_crops"] = frame.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    if src_pmask is not None:
        if src_pmask.dim() == 3:
            out["global_pmasks"] = src_pmask.unsqueeze(1).repeat(1, 2, 1, 1)
        elif src_pmask.dim() == 4 and src_pmask.shape[1] == 1:
            out["global_pmasks"] = src_pmask.repeat(1, 2, 1, 1, 1)
    return out


# ---------------------------------------------------------------------------
# Stage 1: Image semantic warm-start
# ---------------------------------------------------------------------------

def student_stage1_step(
    batch: dict,
    student,                   # HieraStudentBackbone
    ema_student,               # EMA copy of student (nn.Module)
    dino_teacher,              # FrozenDINOTeacher
    proto_head,                # PrototypeHead (shared, from models/branches/shared.py)
    lam: dict,
    global_step: int = 0,
    alp_feedback=None,           # optional HardnessFeedback → ALPScoreCache
    stage: int = 1,
) -> dict:
    """
    Image-only SSL step (T=1 batches).

    Student sees masked global crop 0; DINO and EMA see clean global crop 1.
    DINO losses use align_dim projections; EMA uses native hidden space.

    Losses
    ------
    L = lam_global  * L_img_global   (DINO)
      + lam_patch   * L_img_patch    (DINO, valid & ~freq-masked)
      + lam_masked  * L_img_masked   (DINO, valid & freq-masked)
      + lam_proto   * L_img_proto    (DINO)
      + lam_ema_eff * L_ema          (EMA, auxiliary)
    """
    batch = _ensure_image_batch(batch)
    s_crop = batch["global_crops"][:, 0]          # masked student view
    t_crop = batch["global_crops"][:, 1]          # clean teacher / EMA view
    pmask_s = _student_crop_pmask(batch, 0)
    pmask_t = _student_crop_pmask(batch, 1)
    pmask_dino = _teacher_image_pmask(batch, pmask_t)

    patch_masks = batch.get("patch_masks")        # (B, ph, pw) stride-4 from collator

    x_s = s_crop.unsqueeze(1)                     # (B, 1, 3, H, W)
    x_t = t_crop.unsqueeze(1)

    need_alp = alp_feedback is not None and bool(batch.get("sample_ids"))

    with torch.no_grad():
        # Eager attention (required for cls_patch_attention) produces NaN
        # teacher projections under multi-GPU DDP on GH200.  ALP saliency falls
        # back to patch L2 norm when attention is omitted (see _dino_teacher_saliency).
        t_img = dino_teacher(
            t_crop, padding_mask=pmask_dino, return_attention=False,
        )
        t_ema_out = ema_student(x_t, padding_mask=pmask_t)
        t_cls_proj = t_img["cls_proj"]
        t_patch_proj = t_img["patch_proj"]
        t_ema_global = t_ema_out["global"]
        t_ema_f1 = t_ema_out["F1"][:, 0]

    s_out = student(x_s, padding_mask=pmask_s)
    s_global  = _student_global(s_out)
    s_patches = _student_patches(s_out, t=0)
    pmask_f1  = _student_f1_grid_pmask(s_out)

    unmasked_flat, masked_flat = _dino_patch_masks(s_out, patch_masks)

    L_global = img_global_distill_loss(s_global, t_cls_proj)
    L_patch = img_patch_distill_loss(
        s_patches, t_patch_proj,
        mask=unmasked_flat,
    )
    L_masked = (
        img_masked_semantic_loss(
            s_patches, t_patch_proj, masked_flat, pmask_f1,
        )
        if masked_flat is not None
        else s_global.new_tensor(0.0)
    )
    L_proto = img_proto_loss(
        s_global.unsqueeze(1),
        t_cls_proj.detach().unsqueeze(1),
        proto_head.prototypes,
    )

    lam_ema_eff = _lam_ema_eff(lam, global_step, stage=stage)
    L_ema_global = img_ema_global_loss(s_out["global"], t_ema_global)
    L_ema_patch = img_ema_patch_loss(
        s_out["F1"][:, 0],
        t_ema_f1,
        masked_positions=patch_masks,
        padding_mask=pmask_f1,
    )
    L_ema = L_ema_global + L_ema_patch

    loss = (
        lam.get("lam_global", 1.0)  * L_global
      + lam.get("lam_patch",  1.0)  * L_patch
      + lam.get("lam_masked", 1.0)  * L_masked
      + lam.get("lam_proto",  0.5)  * L_proto
      + lam_ema_eff               * L_ema
    )

    if need_alp:
        with torch.no_grad():
            from models.losses.student_losses import _interpolate_tokens
            tp = t_patch_proj
            if tp.shape[1] != s_patches.shape[1]:
                tp = _interpolate_tokens(tp, s_patches.shape[1])
            per_patch = (1.0 - F.cosine_similarity(s_patches, tp, dim=-1)).clamp(min=0.0)
            # Weight masked tokens higher — these are the student's active learning targets.
            if masked_flat is not None and masked_flat.any():
                patch_hardness = torch.where(
                    masked_flat,
                    per_patch * 1.5,
                    per_patch * 0.25,
                )
            else:
                patch_hardness = per_patch
            saliency = _dino_teacher_saliency(
                t_img, t_patch_proj, s_patches.shape[1], pmask_dino, pmask_f1,
            )
            alp_feedback.update_from_distill(
                batch["sample_ids"], patch_hardness, saliency, global_step,
            )

    return {
        "loss":            loss,
        "loss_global":     L_global.item(),
        "loss_patch":      L_patch.item(),
        "loss_masked":     L_masked.item(),
        "loss_proto":      L_proto.item(),
        "loss_ema":        L_ema.item(),
        "loss_ema_global": L_ema_global.item(),
        "loss_ema_patch":  L_ema_patch.item(),
        "lam_ema_eff":     lam_ema_eff,
    }


# ---------------------------------------------------------------------------
# Stage 2: Video temporal warm-start (mixed image + video batches)
# ---------------------------------------------------------------------------

def student_stage2_step(
    batch: dict,
    student,
    ema_student,
    dino_teacher,
    vjepa_teacher,
    proto_head,
    lam: dict,
    global_step: int = 0,
    alp_feedback=None,
    proto_queue=None,     # optional ProtoQueue; pass None to skip video proto loss
    stage: int = 2,
) -> dict:
    """
    Mixed image + video step.

    Routes by batch["sample_type"]:
      "image" → image SSL losses (same as stage 1)
      "video" → V-JEPA-style tube prediction + temporal consistency

    Both sub-batches are accumulated and summed.
    """
    sample_type = batch.get("sample_type", "image")

    if sample_type == "image":
        return student_stage1_step(
            batch, student, ema_student, dino_teacher, proto_head, lam,
            global_step=global_step, alp_feedback=alp_feedback, stage=stage,
        )

    return _video_ssl_losses(
        batch, student, ema_student, vjepa_teacher, proto_head, lam,
        global_step=global_step, proto_queue=proto_queue, stage=stage,
    )


# ---------------------------------------------------------------------------
# Stage 3: Cross-view coupling (batch-conditional)
# ---------------------------------------------------------------------------

def _student_stage3_paired_single(
    batch: dict,
    student,
    dino_teacher,
    vjepa_teacher,
    fusion_builder,
    lam: dict,
    fused_ramp: float,
) -> dict:
    """
    Paired image+clip step for batch size 1.

    Teachers run sequentially (DINO may live on CPU). Student video path uses
    ``visible_clips`` when present to match stage-2 memory profile.
    """
    frame = batch["frame"]
    clip = batch["full_clips"]
    clip_student = batch.get("visible_clips", clip)
    pmask_student = _pmask(batch, "padding_masks")
    pmask_vjepa = _teacher_video_pmask(batch, pmask_student)
    pmask_img_student = _pmask(batch, "frame_pmask")
    if pmask_img_student is None:
        pmask_img_student = pmask_student
    pmask_img_dino = _teacher_image_pmask(
        batch, pmask_img_student, key="global_pmasks",
    )

    B = frame.shape[0]
    T = clip.shape[1]
    anchor_t = _paired_anchor_t(batch, T // 2)
    out_device = frame.device

    t_img = _forward_frozen_teacher(
        dino_teacher,
        out_device,
        lambda dev: dino_teacher(
            frame.to(dev, non_blocking=True),
            padding_mask=(
                pmask_img_dino.to(dev, non_blocking=True)
                if pmask_img_dino is not None else None
            ),
        ),
    )
    valid_frames = batch.get("valid_frames")
    t_vid = _forward_frozen_teacher(
        vjepa_teacher,
        out_device,
        lambda dev: vjepa_teacher(
            clip.to(dev, non_blocking=True),
            tube_mask=None,
            padding_mask=(
                pmask_vjepa.to(dev, non_blocking=True)
                if pmask_vjepa is not None else None
            ),
            valid_frames=(
                valid_frames.to(dev, non_blocking=True)
                if valid_frames is not None else None
            ),
        ),
    )

    if out_device.type == "cuda":
        torch.cuda.empty_cache()

    N_img = t_img["patch_tokens"].shape[1]
    ph_dino, pw_dino = _factor_token_grid(N_img, pmask_img_dino)
    T_tok = _teacher_tubelet_temporal_len(T, vjepa_teacher)
    n_spat = t_vid["tube_tokens"].shape[1] // max(1, T_tok)
    ph_vid, pw_vid = _factor_token_grid(n_spat, pmask_vjepa)

    z_vid_spatial = _teacher_tubes_spatial_mean(
        t_vid["tube_tokens"], B, T, vjepa_teacher,
    )

    z_fused, z_vid_base = fusion_builder(
        z_img_raw=t_img["patch_tokens"],
        z_vid_raw=z_vid_spatial,
        ph_dino=ph_dino, pw_dino=pw_dino,
        ph_vid=ph_vid,   pw_vid=pw_vid,
        device=out_device,
        detach_output=False,
        return_vid_baseline=True,
    )

    t_img_cls = t_img["cls_proj"]
    t_img_patch = t_img["patch_proj"]
    t_vid_clip = t_vid["clip_proj"]
    t_vid_tubes = t_vid.get("tube_proj")
    has_tube_proj = t_vid_tubes is not None
    del t_img, t_vid, z_vid_spatial
    # V-JEPA stays GPU-resident on GH200 120 GB; no mid-step release needed.

    # Heavy clip forward first (checkpointed); image forward is T=1.
    if clip_student.requires_grad:
        s_vid_out = checkpoint(
            _paired_student_video_forward,
            student,
            clip_student,
            pmask_student,
            use_reentrant=False,
        )
    else:
        s_vid_out = _paired_student_video_forward(
            student, clip_student, pmask_student,
        )

    s_img_out = student(frame.unsqueeze(1), padding_mask=pmask_img_student)
    s_img_f1 = _student_patches(s_img_out, t=0)
    s_img_glob = _student_global(s_img_out)
    s_vid_glob = _student_global(s_vid_out)

    if "tube_proj" in s_vid_out:
        s_clip_frame = s_vid_out["tube_proj"][:, anchor_t]
        s_vid_spatial = s_vid_out["tube_proj"].mean(dim=1)
    else:
        s_vid_f4 = s_vid_out["F4"]
        s_clip_frame = s_vid_f4[:, anchor_t]
        s_vid_spatial = s_vid_f4.reshape(B, T, -1, s_vid_f4.shape[-1]).mean(1)

    L_img_global = img_global_distill_loss(s_img_glob, t_img_cls)
    L_img_patch = img_patch_distill_loss(
        s_img_f1, t_img_patch, mask=_student_f1_flat_pmask(s_img_out),
    )
    L_vid_global = vid_global_distill_loss(s_vid_glob, t_vid_clip)

    tube_mask = _teacher_tube_mask(batch)
    pmask_s16 = _teacher_padding_mask(batch, pmask_vjepa)
    if tube_mask is not None and has_tube_proj:
        s_tubes = s_vid_out.get("tube_proj", s_vid_out["F4"])
        L_tube = vid_tube_masked_distill_loss(
            s_tubes,
            t_vid_tubes,
            tube_mask,
            padding_mask=pmask_s16,
            valid_frames=batch.get("valid_frames"),
        )
    else:
        L_tube = s_img_glob.new_tensor(0.0)

    L_fused = fused_distill_loss(
        s_vid_spatial, z_fused.detach(), padding_mask=pmask_student,
    )
    L_preserve = preservation_loss(z_fused, z_vid_base)
    L_fc = frame_clip_consistency(
        _student_tubes_at_t(s_img_out, t=0),
        s_clip_frame,
        padding_mask=_student_f4_grid_pmask(s_img_out),
    )

    lam_fused_eff = lam.get("lam_fused", 0.1) * fused_ramp
    loss = (
        lam.get("lam_global",   1.0) * L_img_global
      + lam.get("lam_patch",    1.0) * L_img_patch
      + lam.get("lam_global",   1.0) * L_vid_global
      + lam.get("lam_tube",     1.0) * L_tube
      + lam_fused_eff                * L_fused
      + lam.get("lam_preserve", 0.1) * L_preserve
      + lam.get("lam_fc",       0.5) * L_fc
    )

    return {
        "loss":            loss,
        "loss_img_global": L_img_global.item(),
        "loss_img_patch":  L_img_patch.item(),
        "loss_vid_global": L_vid_global.item(),
        "loss_tube":       L_tube.item(),
        "loss_fused":      L_fused.item(),
        "loss_preserve":   L_preserve.item(),
        "loss_fc":         L_fc.item(),
    }


def student_stage3_step(
    batch: dict,
    student,
    ema_student,
    dino_teacher,
    vjepa_teacher,
    fusion_builder,       # FusionTargetBuilder
    proto_head,
    lam: dict,
    curriculum_stage: int = 1,   # 1, 2, or 3 — controls weight ramps
    global_step: int = 0,
    alp_feedback=None,
    proto_queue=None,     # optional ProtoQueue for video batches
) -> dict:
    """
    Three-way batch-conditional step for Stage 3.

    Routing by batch["sample_type"]:
      "image"  → DINO distillation + masked patch + proto (same as stage 1)
      "video"  → V-JEPA tube prediction + temporal + proto (same as stage 2)
      "paired" → Full cross-distillation: DINO + JEPA + fused target +
                 frame-clip consistency

    curriculum_stage controls lam_fused ramp-up:
      stage=1: lam_fused × 0.1  (gentle warmup)
      stage=2: lam_fused × 0.5
      stage=3: lam_fused × 1.0
    """
    fused_ramp = {1: 0.1, 2: 0.5, 3: 1.0}.get(curriculum_stage, 1.0)
    sample_type = batch.get("sample_type", "image")

    if sample_type == "image":
        return student_stage1_step(
            batch, student, ema_student, dino_teacher, proto_head, lam,
            global_step=global_step, alp_feedback=alp_feedback, stage=3,
        )

    if sample_type == "video":
        return _video_ssl_losses(
            batch, student, ema_student, vjepa_teacher, proto_head, lam,
            global_step=global_step, proto_queue=proto_queue, stage=3,
        )

    # ── Paired branch ─────────────────────────────────────────────────
    frame = batch.get("frame")
    clip = batch.get("full_clips")
    if frame is None or clip is None:
        return student_stage2_step(
            batch, student, ema_student, dino_teacher, vjepa_teacher, proto_head, lam,
            global_step=global_step, alp_feedback=alp_feedback, proto_queue=proto_queue,
            stage=3,
        )

    B = frame.shape[0]
    if B <= 1:
        metrics = _student_stage3_paired_single(
            batch, student, dino_teacher, vjepa_teacher, fusion_builder, lam, fused_ramp,
        )
    else:
        # Micro-batch paired samples to cap peak VRAM (two student forwards each).
        loss_sum = None
        metric_lists: dict[str, list[float]] = {}
        for i in range(B):
            sub = _slice_batch_dim0(batch, i)
            out_i = _student_stage3_paired_single(
                sub, student, dino_teacher, vjepa_teacher, fusion_builder, lam, fused_ramp,
            )
            li = out_i["loss"]
            loss_sum = li if loss_sum is None else loss_sum + li
            for key, val in out_i.items():
                if key == "loss":
                    metric_lists.setdefault(key, []).append(float(val.item()))
                elif key.startswith("loss_"):
                    metric_lists.setdefault(key, []).append(float(val))
        loss = loss_sum / B
        metrics = {key: sum(vals) / len(vals) for key, vals in metric_lists.items()}
        metrics["loss"] = loss

    return {
        "loss":              metrics["loss"],
        "loss_img_global":   metrics["loss_img_global"],
        "loss_img_patch":    metrics["loss_img_patch"],
        "loss_vid_global":   metrics["loss_vid_global"],
        "loss_tube":         metrics["loss_tube"],
        "loss_fused":        metrics["loss_fused"],
        "loss_preserve":     metrics["loss_preserve"],
        "loss_fc":           metrics["loss_fc"],
        "sample_type":       "paired",
        "curriculum_stage":  curriculum_stage,
        "fused_ramp":        fused_ramp,
    }


# ---------------------------------------------------------------------------
# Stage 4: EMA self-distillation divergence (no frozen teachers)
# ---------------------------------------------------------------------------

def _image_ema_ssl_losses(
    batch: dict,
    student,
    ema_student,
    proto_head,
    lam: dict,
    ema_scale: float,
    global_step: int = 0,
    alp_feedback=None,
) -> dict:
    """
    Image-only EMA self-distillation (stage 4).

    Student sees masked global crop 0; EMA teacher sees clean global crop 1.
    No frozen DINO forward.
    """
    batch = _ensure_image_batch(batch)
    s_crop = batch["global_crops"][:, 0]
    t_crop = batch["global_crops"][:, 1]
    pmask_s = _student_crop_pmask(batch, 0)
    pmask_t = _student_crop_pmask(batch, 1)
    patch_masks = batch.get("patch_masks")

    x_s = s_crop.unsqueeze(1)
    x_t = t_crop.unsqueeze(1)

    need_alp = alp_feedback is not None and bool(batch.get("sample_ids"))

    with torch.no_grad():
        t_ema_out = ema_student(x_t, padding_mask=pmask_t)
        t_ema_global = t_ema_out["global"]
        t_ema_f1 = t_ema_out["F1"][:, 0]

    s_out = student(x_s, padding_mask=pmask_s)
    s_global = _student_global(s_out)
    s_patches = _student_patches(s_out, t=0)
    pmask_f1 = _student_f1_grid_pmask(s_out)

    unmasked_flat, masked_flat = _dino_patch_masks(s_out, patch_masks)

    L_ema_global = img_ema_global_loss(s_out["global"], t_ema_global)

    L_ema_visible = (
        img_ema_patch_loss(
            s_out["F1"][:, 0], t_ema_f1,
            masked_positions=unmasked_flat,
            padding_mask=pmask_f1,
        )
        if unmasked_flat is not None and unmasked_flat.any()
        else s_global.new_tensor(0.0)
    )
    L_ema_masked = (
        img_ema_patch_loss(
            s_out["F1"][:, 0], t_ema_f1,
            masked_positions=masked_flat,
            padding_mask=pmask_f1,
        )
        if masked_flat is not None and masked_flat.any()
        else s_global.new_tensor(0.0)
    )

    L_proto = img_proto_loss(
        s_global.unsqueeze(1),
        t_ema_global.detach().unsqueeze(1),
        proto_head.prototypes,
    )

    loss = ema_scale * (
        L_ema_global
        + lam.get("lam_patch", 1.0) * L_ema_visible
        + lam.get("lam_masked", 1.0) * L_ema_masked
    ) + lam.get("lam_proto", 0.5) * L_proto

    if need_alp:
        with torch.no_grad():
            from models.losses.student_losses import _interpolate_tokens
            tp = t_ema_f1
            if tp.shape[1] != s_patches.shape[1]:
                tp = _interpolate_tokens(tp, s_patches.shape[1])
            per_patch = (1.0 - F.cosine_similarity(s_patches, tp, dim=-1)).clamp(min=0.0)
            if masked_flat is not None and masked_flat.any():
                patch_hardness = torch.where(
                    masked_flat, per_patch * 1.5, per_patch * 0.25,
                )
            else:
                patch_hardness = per_patch
            saliency = tp.norm(dim=-1)
            alp_feedback.update_from_distill(
                batch["sample_ids"], patch_hardness, saliency, global_step,
            )

    return {
        "loss":            loss,
        "loss_ema":        (L_ema_global + L_ema_visible + L_ema_masked).item(),
        "loss_ema_global": L_ema_global.item(),
        "loss_ema_patch":  (L_ema_visible + L_ema_masked).item(),
        "loss_proto":      L_proto.item(),
        "ema_scale":       ema_scale,
        "sample_type":     "image",
    }


def _video_ema_ssl_losses(
    batch: dict,
    student,
    ema_student,
    proto_head,
    lam: dict,
    ema_scale: float,
    tubelet_size: int = 2,
    proto_queue=None,
) -> dict:
    """
    Video-only EMA self-distillation (stage 4).

    Student sees masked clips; EMA teacher sees clean clips.  No V-JEPA forward.
    """
    full_clips = batch["full_clips"]
    visible_clips = batch.get("visible_clips", full_clips)
    tube_mask = _teacher_tube_mask(batch)
    pmask_student = _pmask(batch, "padding_masks")
    pmask_s16 = _teacher_padding_mask(batch, _teacher_video_pmask(batch, pmask_student))
    valid_frames = batch.get("valid_frames")

    B, T = full_clips.shape[:2]
    _vid_kw = dict(padding_mask=pmask_student, with_patch_proj=False)

    with torch.no_grad():
        t_ema_out = ema_student(full_clips, **_vid_kw)

    s_out = student(visible_clips, **_vid_kw)

    if "tube_proj" in s_out:
        s_frame_globals = s_out["tube_proj"].mean(dim=2)
    else:
        s_frame_globals = s_out["F4"].mean(dim=2)

    L_ema_global = img_ema_global_loss(s_out["global"], t_ema_out["global"])
    n_f4 = s_out["F4"].shape[2]
    tube_flat = (
        _tube_mask_f4_grid(tube_mask, n_f4, pmask_s16, valid_frames)
        if tube_mask is not None else None
    )
    s_tube = s_out.get("tube_proj", s_out["F4"])
    t_tube = t_ema_out.get("tube_proj", t_ema_out["F4"])
    L_ema_patch = img_ema_patch_loss(
        s_tube.reshape(B, T * n_f4, -1),
        t_tube.reshape(B, T * n_f4, -1),
        masked_positions=tube_flat,
        padding_mask=_student_f4_grid_pmask(s_out),
    )
    L_ema = L_ema_global + L_ema_patch

    L_temp = vid_temporal_consistency(
        s_frame_globals, padding_mask=pmask_student, valid_frames=valid_frames,
    )

    proto_entropy = float("nan")
    proto_max_prob = float("nan")
    if proto_queue is not None:
        s_tubes_flat = _student_tubes_flat(s_out, B)
        t_tubes_flat = t_tube.reshape(B, -1, t_tube.shape[-1])
        L_proto, teacher_logits_for_queue = vid_proto_loss(
            s_tubes_flat,
            t_tubes_flat.detach(),
            proto_head.prototypes,
            queue=proto_queue,
        )
        if teacher_logits_for_queue is not None:
            proto_queue.enqueue(teacher_logits_for_queue)
        with torch.no_grad():
            ent, mx = prototype_assignment_stats(s_tubes_flat, proto_head.prototypes)
            proto_entropy = float(ent.item())
            proto_max_prob = float(mx.item())
    else:
        L_proto = s_out["global"].new_tensor(0.0)

    loss = (
        ema_scale * L_ema
        + lam.get("lam_temp", 0.5) * L_temp
        + lam.get("lam_proto", 0.5) * L_proto
    )

    return {
        "loss":             loss,
        "loss_ema":         L_ema.item(),
        "loss_ema_global":  L_ema_global.item(),
        "loss_ema_patch":   L_ema_patch.item(),
        "loss_temporal":    L_temp.item(),
        "loss_proto":       L_proto.item(),
        "proto_entropy":    proto_entropy,
        "proto_max_prob":   proto_max_prob,
        "ema_scale":        ema_scale,
        "sample_type":      "video",
    }


def _student_stage4_paired_single(
    batch: dict,
    student,
    ema_student,
    proto_head,
    lam: dict,
    ema_scale: float,
) -> dict:
    """Paired EMA divergence: no fusion, no frozen teachers."""
    frame = batch["frame"]
    clip = batch["full_clips"]
    clip_student = batch.get("visible_clips", clip)
    pmask_student = _pmask(batch, "padding_masks")
    pmask_img_student = _pmask(batch, "frame_pmask")
    if pmask_img_student is None:
        pmask_img_student = pmask_student

    B = frame.shape[0]
    T = clip.shape[1]
    anchor_t = _paired_anchor_t(batch, T // 2)

    clean_frame = batch.get("clean_frame")
    if clean_frame is None:
        clean_frame = clip[:, anchor_t]

    pmask_clean = batch.get("clean_frame_pmask")
    if pmask_clean is None and batch.get("global_pmasks") is not None:
        gpm = batch["global_pmasks"]
        if gpm.dim() == 4 and gpm.shape[1] >= 2:
            pmask_clean = gpm[:, 1]
        elif gpm.dim() == 4:
            pmask_clean = gpm[:, 0]

    tube_mask = _teacher_tube_mask(batch)
    pmask_s16 = _teacher_padding_mask(
        batch, _teacher_video_pmask(batch, pmask_student),
    )

    with torch.no_grad():
        t_ema_img = ema_student(frame.unsqueeze(1), padding_mask=pmask_img_student)
        t_ema_vid = ema_student(
            clip, padding_mask=pmask_student, with_patch_proj=False,
        )
        t_ema_frame = ema_student(
            clean_frame.unsqueeze(1), padding_mask=pmask_clean,
        )

    if clip_student.requires_grad:
        s_vid_out = checkpoint(
            _paired_student_video_forward,
            student, clip_student, pmask_student,
            use_reentrant=False,
        )
    else:
        s_vid_out = _paired_student_video_forward(
            student, clip_student, pmask_student,
        )

    s_img_out = student(frame.unsqueeze(1), padding_mask=pmask_img_student)

    L_ema_img_global = img_ema_global_loss(
        s_img_out["global"], t_ema_img["global"],
    )
    L_ema_img_patch = img_ema_patch_loss(
        s_img_out["F1"][:, 0],
        t_ema_frame["F1"][:, 0],
        padding_mask=_student_f1_grid_pmask(s_img_out),
    )

    L_ema_vid_global = img_ema_global_loss(
        _student_global(s_vid_out), t_ema_vid["global"],
    )

    if tube_mask is not None:
        n_f4 = s_vid_out["F4"].shape[2]
        tube_flat = _tube_mask_f4_grid(
            tube_mask, n_f4, pmask_s16, batch.get("valid_frames"),
        )
        s_tube = s_vid_out.get("tube_proj", s_vid_out["F4"])
        t_tube = t_ema_vid.get("tube_proj", t_ema_vid["F4"])
        L_ema_tube = img_ema_patch_loss(
            s_tube.reshape(B, T * n_f4, -1),
            t_tube.reshape(B, T * n_f4, -1),
            masked_positions=tube_flat,
            padding_mask=_student_f4_grid_pmask(s_vid_out),
        )
    else:
        L_ema_tube = s_img_out["global"].new_tensor(0.0)

    if "tube_proj" in s_vid_out:
        s_clip_frame = s_vid_out["tube_proj"][:, anchor_t]
    else:
        s_clip_frame = s_vid_out["F4"][:, anchor_t]

    L_fc = frame_clip_consistency(
        _student_tubes_at_t(s_img_out, t=0),
        s_clip_frame,
        padding_mask=_student_f4_grid_pmask(s_img_out),
    )

    L_ema = L_ema_img_global + L_ema_img_patch + L_ema_vid_global + L_ema_tube
    loss = (
        ema_scale * L_ema
        + lam.get("lam_fc", 0.5) * L_fc
    )

    return {
        "loss":            loss,
        "loss_ema":        L_ema.item(),
        "loss_ema_global": (L_ema_img_global + L_ema_vid_global).item(),
        "loss_ema_patch":  (L_ema_img_patch + L_ema_tube).item(),
        "loss_fc":         L_fc.item(),
        "ema_scale":       ema_scale,
    }


def student_stage4_divergence_step(
    batch: dict,
    student,
    ema_student,
    proto_head,
    lam: dict,
    global_step: int = 0,
    stage4_start: int = 0,
    stage4_end: int = 0,
    alp_feedback=None,
    proto_queue=None,
) -> dict:
    """
    Stage 4: EMA-primary self-distillation (no frozen DINO/V-JEPA).

    Routing by batch["sample_type"]:
      "image"  → _image_ema_ssl_losses
      "video"  → _video_ema_ssl_losses
      "paired" → _student_stage4_paired_single + frame_clip_consistency
    """
    ema_scale = _lam_ema_eff(
        lam, global_step,
        stage=4, stage4_start=stage4_start, stage4_end=stage4_end,
    )
    sample_type = batch.get("sample_type", "image")

    if sample_type == "image":
        out = _image_ema_ssl_losses(
            batch, student, ema_student, proto_head, lam, ema_scale,
            global_step=global_step, alp_feedback=alp_feedback,
        )
        out["lam_ema_eff"] = ema_scale
        return out

    if sample_type == "video":
        out = _video_ema_ssl_losses(
            batch, student, ema_student, proto_head, lam, ema_scale,
            proto_queue=proto_queue,
        )
        out["lam_ema_eff"] = ema_scale
        return out

    frame = batch.get("frame")
    clip = batch.get("full_clips")
    if frame is None or clip is None:
        if batch.get("sample_type") == "video":
            out = _video_ema_ssl_losses(
                batch, student, ema_student, proto_head, lam, ema_scale,
                proto_queue=proto_queue,
            )
        else:
            out = _image_ema_ssl_losses(
                batch, student, ema_student, proto_head, lam, ema_scale,
                global_step=global_step, alp_feedback=alp_feedback,
            )
        out["lam_ema_eff"] = ema_scale
        return out

    B = frame.shape[0]
    if B <= 1:
        metrics = _student_stage4_paired_single(
            batch, student, ema_student, proto_head, lam, ema_scale,
        )
    else:
        loss_sum = None
        metric_lists: dict[str, list[float]] = {}
        for i in range(B):
            sub = _slice_batch_dim0(batch, i)
            out_i = _student_stage4_paired_single(
                sub, student, ema_student, proto_head, lam, ema_scale,
            )
            li = out_i["loss"]
            loss_sum = li if loss_sum is None else loss_sum + li
            for key, val in out_i.items():
                if key == "loss":
                    metric_lists.setdefault(key, []).append(float(val.item()))
                elif key.startswith("loss_") or key == "ema_scale":
                    metric_lists.setdefault(key, []).append(float(val))
        loss = loss_sum / B
        metrics = {key: sum(vals) / len(vals) for key, vals in metric_lists.items()}
        metrics["loss"] = loss

    return {
        "loss":              metrics["loss"],
        "loss_ema":          metrics.get("loss_ema", float("nan")),
        "loss_ema_global":   metrics.get("loss_ema_global", float("nan")),
        "loss_ema_patch":    metrics.get("loss_ema_patch", float("nan")),
        "loss_fc":           metrics.get("loss_fc", float("nan")),
        "sample_type":       "paired",
        "lam_ema_eff":       ema_scale,
        "ema_scale":         ema_scale,
    }


# ---------------------------------------------------------------------------
# Stage 5: Supervised head training (optional; not in default pretrain)
# ---------------------------------------------------------------------------

def student_stage5_supervised_step(
    batch: dict,
    student,                   # HieraStudentBackbone (frozen or light unfreeze)
    seg_head,                  # UPerNetDecoder or None
    cls_heads: dict,           # {"domain": DomainHead, ...} — any subset
    lam: dict,
    backbone_frozen: bool = True,
) -> dict:
    """
    Supervised downstream head training step.

    The backbone is frozen by default (backbone_frozen=True).
    The trainer unfreezes it selectively via trainable_stages config.

    Segmentation
    ------------
    If batch["seg_masks"] is present and seg_head is not None:
      pred = seg_head(features, padding_mask)   → (B, C, h, w)
      loss = Dice + BCE

    Classification / concept
    ------------------------
    For each key in cls_heads:
      if batch[f"{key}_labels"] is present:
        pred = cls_heads[key](features)
        loss += head.loss(pred[...], labels)
    """
    pmask  = _pmask(batch, "padding_masks")
    if pmask is None:
        pmask = _pmask(batch, "global_pmasks")
    if pmask is not None and pmask.dim() == 4:
        pmask = pmask[:, 0]

    sample_type = batch.get("sample_type", "image")
    is_image    = (sample_type == "image")

    pixel_values = (
        batch["global_crops"][:, 0].unsqueeze(1)
        if is_image
        else batch.get("full_clips")
    )

    ctx = torch.no_grad() if backbone_frozen else torch.enable_grad()
    with ctx:
        features = student(pixel_values, padding_mask=pmask)

    loss = pixel_values.new_tensor(0.0, requires_grad=True)
    out  = {}

    # ── Segmentation ─────────────────────────────────────────────────
    if seg_head is not None and batch.get("seg_masks") is not None:
        seg_masks = batch["seg_masks"].float()   # (B, C, H, W) or (B, T, C, H, W)

        if pixel_values.shape[1] == 1:
            # Image: decode single frame
            pred_seg = seg_head(features, padding_mask=pmask, frame_indices=[0])
        else:
            # Video: decode frames that have labels
            labeled_frames = batch.get("labeled_frame_indices")
            pred_seg = seg_head(features, padding_mask=pmask,
                                frame_indices=labeled_frames)
            if labeled_frames is not None and seg_masks.dim() == 5:
                seg_masks = seg_masks[:, labeled_frames]

        seg_masks = _align_seg_masks_to_pred(seg_masks, pred_seg)

        # Compute BCE + Dice loss
        bce_loss  = F.binary_cross_entropy_with_logits(pred_seg, seg_masks)
        dice_loss = _dice_loss(pred_seg, seg_masks)
        seg_loss  = bce_loss + lam.get("lam_seg_dice", 1.0) * dice_loss

        loss = loss + lam.get("lam_seg", 1.0) * seg_loss
        out["loss_seg"]  = seg_loss.item()
        out["loss_bce"]  = bce_loss.item()
        out["loss_dice"] = dice_loss.item()

    # ── Classification heads ──────────────────────────────────────────
    for head_name, head in cls_heads.items():
        label_key = f"{head_name}_labels"
        if label_key not in batch:
            continue
        labels = batch[label_key]
        preds  = head(features)
        # Get the logits tensor (first value in the output dict)
        logits_key = next(iter(preds))
        head_loss  = head.loss(preds[logits_key], labels)
        lam_key    = f"lam_{head_name}"
        loss = loss + lam.get(lam_key, 1.0) * head_loss
        out[f"loss_{head_name}"] = head_loss.item()

    out["loss"] = loss
    return out


# Backward-compatible alias (legacy name before stage-4 EMA divergence)
student_stage4_step = student_stage5_supervised_step


def _align_seg_masks_to_pred(seg_masks: Tensor, pred_seg: Tensor) -> Tensor:
    """
    Resize ground-truth masks to UPerNet logit resolution.

    Collators often emit ``seg_masks`` at native image resolution while the
    decoder outputs at the F1 token grid; nearest-neighbour keeps labels binary.
    """
    if seg_masks.shape[-2:] == pred_seg.shape[-2:]:
        return seg_masks.to(device=pred_seg.device)

    if pred_seg.dim() == 4:
        tgt = seg_masks
        if tgt.dim() == 5:
            tgt = tgt[:, 0]
        if tgt.dim() == 3:
            tgt = tgt.unsqueeze(1)
        if tgt.shape[1] != pred_seg.shape[1]:
            if tgt.shape[1] == 1:
                tgt = tgt.expand(-1, pred_seg.shape[1], -1, -1)
            else:
                tgt = tgt[:, : pred_seg.shape[1]]
        aligned = F.interpolate(
            tgt.float(), size=pred_seg.shape[-2:], mode="nearest",
        )
        return aligned.to(device=pred_seg.device, dtype=seg_masks.dtype)

    if pred_seg.dim() == 5:
        B, T, C, h, w = pred_seg.shape
        tgt = seg_masks
        if tgt.dim() == 4:
            tgt = tgt.unsqueeze(1).expand(-1, T, -1, -1, -1)
        tgt = tgt.reshape(B * T, C, tgt.shape[-2], tgt.shape[-1])
        aligned = F.interpolate(tgt.float(), size=(h, w), mode="nearest")
        return aligned.reshape(B, T, C, h, w).to(device=pred_seg.device, dtype=seg_masks.dtype)

    return seg_masks.to(device=pred_seg.device)


# ---------------------------------------------------------------------------
# Dice loss utility
# ---------------------------------------------------------------------------

def _dice_loss(pred: Tensor, target: Tensor, smooth: float = 1.0) -> Tensor:
    """
    Differentiable Dice loss for binary segmentation.
    pred   : (B, C, H, W) raw logits
    target : (B, C, H, W) float 0/1
    """
    p = torch.sigmoid(pred).flatten(2)   # (B, C, N)
    t = target.flatten(2).float()
    intersection = (p * t).sum(2)
    union        = p.sum(2) + t.sum(2)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()
