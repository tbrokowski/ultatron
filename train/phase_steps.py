"""
train/phase_steps.py  ·  Pure phase step functions
========================================================

Each function here takes tensors and nn.Modules and returns a loss dict.
No optimizer.step(), no logging, no checkpointing — those belong in
trainer.py.  This makes each step independently unit-testable with tiny
synthetic tensors.

Contract
--------
Every phase*_step function:
  - Accepts a batch dict, model objects, and scalar hyperparameters
  - Performs one forward + backward pass under autocast
  - Returns a dict of loss scalars (Python floats, not tensors)
  - Does NOT call optimizer.step() — the trainer does that
  - Does NOT call branch.update_teacher() — the trainer does that

The trainer (trainer.py) calls:
    losses = phaseN_step(batch, ..., scaler)
    scaler.scale(losses["loss"]).backward()
    clip_grad_norm_(...)
    scaler.step(optimizer)
    scaler.update()
    img_branch.update_teacher(momentum)

Loss weights (lam dict)
-----------------------
  lam1          CE CLS loss weight (global + local crops)
  lam2          CE iBOT patch loss weight (masked positions)
  lam3          local crop CLS weight (applied inside lam1 multi-crop)
  lam4          video CLS loss weight
  lam5          tube prediction loss weight
  lam6          cross-branch distillation weight
  lam7          prototype consistency weight
  lam_7b        DINOv3-7B teacher distillation weight
  lam_gram      Gram anchoring weight
  lam_ctx       V-JEPA 2.1 dense context loss (visible tokens, distance-weighted)
  lam_recon     pixel reconstruction loss (L1 on masked patches)
  lam_sam       SAM2/SAM3 RADIO-style patch distillation weight
  lam_deep      per-layer weight for deep self-supervision intermediate layers
                (each layer gets lam_deep / n_intermediate_layers)
  tau_student   softmax temperature for student (CE losses)
  tau_teacher   softmax temperature for teacher (CE losses)

Loss mode (lam["loss_mode"])
----------------------------
  "multi"    (default) — all individual lambda weights active, current behaviour
  "grouped"  — 3 grouped terms:
                 L_SPC   (lam_align)     energy-weighted CE over all tokens
                 L_decode (lam_decode)   pixel recon + SAM distil at masked positions
                 L_cross  (lam_cross_grouped)  FILIP only (Phase 3)
  "unified"  — same as grouped with lam_align=1.0 (SPC is base objective)

Additional lam keys used by grouped / unified modes:
  lam_align         L_SPC weight (grouped only; unified treats it as 1.0)
  lam_decode        L_decode weight
  lam_cross_grouped L_cross (FILIP) weight in phase3 grouped/unified
  spc_ctx_base      floor weight added to visible token positions in SPC
  spc_alpha_recon   pixel-recon sub-weight inside L_decode
  spc_alpha_ext     external-teacher sub-weight inside L_decode
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler

from models.losses.image_losses import (
    dino_cls_loss,
    dino_cls_loss_multicrop,
    dino_cls_loss_ce_multicrop,
    ibot_patch_loss,
    ibot_patch_loss_ce,
    dense_context_loss,
    pixel_reconstruction_loss,
    koleo_loss,
    spectral_predictive_coding_loss,
    grouped_decode_loss,
)
from models.losses.video_losses import jepa_tube_loss, clip_cls_loss
from models.losses.cross_branch import (
    cross_branch_loss_from_tokens,
    filip_cross_loss_from_pairs,
    infonce_cross_loss_from_pairs,
)
from models.losses.proto_loss import (
    proto_loss_from_tokens,
    swav_proto_loss_from_tokens,
)
from .gram import GramTeacher, gram_loss
from .alp import HardnessFeedback


# ── Padding mask helpers ──────────────────────────────────────────────────────

def _get_pmask(batch: dict, crop_idx: int = 0) -> Optional[Tensor]:
    """Extract (B, ph, pw) padding mask for a specific global crop index."""
    pm = batch.get("global_pmasks")
    return pm[:, crop_idx] if pm is not None else None


def _active_patch_mask(
    pmask: Optional[Tensor],   # (B, ph, pw) bool
    freq_mask: Tensor,          # (B, ph, pw) bool
) -> Tensor:
    """
    Combine padding mask and frequency energy mask into a flat active mask.
    active = real content AND frequency-masked → (B, N) bool
    """
    flat_freq = freq_mask.flatten(1)                          # (B, N)
    if pmask is None:
        return flat_freq
    return pmask.flatten(1) & flat_freq


# ── Energy-map helpers ────────────────────────────────────────────────────────

def _build_energy_flat(batch: dict, device) -> Tensor:
    """Return (B, N) float energy weight map for SPC loss.

    Uses ``energy_maps`` from the batch when available (continuous spectral
    energy from FFT masking).  Falls back to the binary ``patch_masks`` cast
    to float for spatial / no-energy-map cases.
    """
    energy_map = batch.get("energy_maps")
    if energy_map is None:
        energy_map = batch["patch_masks"].float()
    return energy_map.flatten(1).to(device)   # (B, N)


def _build_sam_external(batch, img_branch, s_pmask, device):
    """Fetch SAM teacher patches for use in grouped_decode_loss.

    Returns a single-element list ``[(teacher_patches, proj_head)]`` when the
    SAM teacher is present, otherwise an empty list.
    """
    if img_branch.teacher_sam is None or img_branch.proj_sam_patch is None:
        return []
    ph = s_pmask.shape[1] if s_pmask is not None else None
    pw = s_pmask.shape[2] if s_pmask is not None else None
    with torch.no_grad():
        t_sam = img_branch.forward_teacher_sam(
            batch["global_crops"][:, 1].to(device),
            target_ph=ph, target_pw=pw,
        )
    if t_sam is None:
        return []
    return [(t_sam["patch_tokens"].to(device), img_branch.proj_sam_patch)]


# ── Phase 1: Image branch warm-start ─────────────────────────────────────────

def _phase1_multi(
    s_out: dict, t_out: dict, local_cls_list: List[Tensor],
    active: Tensor, batch: dict, img_branch, gram_teacher: Optional[GramTeacher],
    lam: dict, s_pmask: Optional[Tensor], tau_s: float, tau_t: float,
    global_step: int, use_koleo: bool,
) -> dict:
    """multi mode — all individual lambda weights active (current behaviour)."""
    device = s_out["cls"].device

    # ── CE CLS loss: teacher global → student global + student local ──────────
    loss_cls = dino_cls_loss_ce_multicrop(
        s_out["cls"], t_out["cls"], local_cls_list,
        student_temp=tau_s, teacher_temp=tau_t,
        local_weight=lam.get("lam3", 0.5),
    )

    # ── CE patch loss at real + freq-masked positions ─────────────────────────
    loss_patch = ibot_patch_loss_ce(
        s_out["patch_tokens"], t_out["patch_tokens"], active,
        student_temp=tau_s, teacher_temp=tau_t,
    )

    # ── Deep self-supervision: CE loss at each intermediate layer ─────────────
    loss_deep = s_out["cls"].new_tensor(0.0)
    lam_deep  = lam.get("lam_deep", 0.0)
    if lam_deep > 0:
        s_inter = s_out.get("intermediate_patch_tokens")
        t_inter = t_out.get("intermediate_patch_tokens")
        if s_inter is not None and t_inter is not None:
            n_inter = len(s_inter)
            per_layer_w = lam_deep / max(n_inter, 1)
            for s_layer, t_layer in zip(s_inter, t_inter):
                loss_deep = loss_deep + per_layer_w * ibot_patch_loss_ce(
                    s_layer, t_layer, active,
                    student_temp=tau_s, teacher_temp=tau_t,
                )

    # ── V-JEPA 2.1 dense context loss on visible tokens ───────────────────────
    loss_ctx = s_out["cls"].new_tensor(0.0)
    if lam.get("lam_ctx", 0.0) > 0:
        loss_ctx = dense_context_loss(
            s_out["patch_tokens"], t_out["patch_tokens"],
            batch["patch_masks"],
            lam_ctx=lam.get("lam_ctx", 1.0),
        )

    # ── Pixel reconstruction at masked patch positions ────────────────────────
    loss_recon = s_out["cls"].new_tensor(0.0)
    if lam.get("lam_recon", 0.0) > 0:
        raw_crop = batch.get("raw_crops")
        if raw_crop is not None:
            pred_px = img_branch.recon_head(s_out["patch_tokens"])
            loss_recon = pixel_reconstruction_loss(
                pred_px, raw_crop.to(device), active,
                patch_size=img_branch.patch_size,
            )

    # ── KoLeo uniformity (optional) ───────────────────────────────────────────
    loss_koleo = koleo_loss(s_out["cls"]) if use_koleo else s_out["cls"].new_tensor(0.0)

    # ── DINOv3-7B frozen teacher distillation ─────────────────────────────────
    loss_7b = s_out["cls"].new_tensor(0.0)
    if img_branch.teacher_d is not None and lam.get("lam_7b", 0.0) > 0:
        with torch.no_grad():
            t7b = img_branch.forward_teacher_d(batch["global_crops"][:, 1])
        if t7b is not None and img_branch.proj_d is not None:
            t_proj = F.normalize(
                img_branch.proj_d(t7b["cls"].float().to(device)), dim=-1
            )
            loss_7b = dino_cls_loss(s_out["cls"], t_proj)

    # ── SAM2/SAM3 RADIO-style patch distillation ─────────────────────────────
    loss_sam = s_out["cls"].new_tensor(0.0)
    if img_branch.teacher_sam is not None and lam.get("lam_sam", 0.0) > 0:
        ph = s_pmask.shape[1] if s_pmask is not None else None
        pw = s_pmask.shape[2] if s_pmask is not None else None
        with torch.no_grad():
            t_sam = img_branch.forward_teacher_sam(
                batch["global_crops"][:, 1].to(device),
                target_ph=ph, target_pw=pw,
            )
        if t_sam is not None and img_branch.proj_sam_patch is not None:
            dt = img_branch.proj_sam_patch.weight.dtype
            sam_proj = F.normalize(
                img_branch.proj_sam_patch(
                    t_sam["patch_tokens"].float().to(dt).to(device)
                ), dim=-1,
            )
            patch_valid = (
                s_pmask.flatten(1) if s_pmask is not None
                else torch.ones(s_out["patch_tokens"].shape[:2],
                                dtype=torch.bool, device=device)
            )
            loss_sam = ibot_patch_loss(s_out["patch_tokens"], sam_proj, patch_valid)

    # ── Gram anchoring ────────────────────────────────────────────────────────
    loss_gram = s_out["cls"].new_tensor(0.0)
    if gram_teacher is not None and gram_teacher.is_active(global_step):
        gram_teacher.maybe_refresh(img_branch.student, global_step)
        X_S = F.normalize(s_out["patch_tokens"], dim=-1)
        X_G = gram_teacher.forward(batch["global_crops"][:, 0], padding_mask=s_pmask)
        loss_gram = gram_loss(X_S, X_G, padding_mask=s_pmask)

    loss = (
        lam.get("lam1",      1.0) * loss_cls
        + lam.get("lam2",    1.0) * loss_patch
        + loss_deep
        + lam.get("lam_ctx",   0.0) * loss_ctx
        + lam.get("lam_recon", 0.0) * loss_recon
        + lam.get("lam_koleo", 0.1) * loss_koleo
        + lam.get("lam_7b",    0.0) * loss_7b
        + lam.get("lam_sam",   0.0) * loss_sam
        + lam.get("lam_gram",  1.0) * loss_gram
    )

    return {
        "loss":        loss,
        "loss_cls":    loss_cls.item(),
        "loss_patch":  loss_patch.item(),
        "loss_deep":   loss_deep.item(),
        "loss_ctx":    loss_ctx.item(),
        "loss_recon":  loss_recon.item(),
        "loss_koleo":  loss_koleo.item(),
        "loss_7b":     loss_7b.item(),
        "loss_sam":    loss_sam.item(),
        "loss_gram":   loss_gram.item(),
    }


def _phase1_grouped(
    s_out: dict, t_out: dict, local_cls_list: List[Tensor],
    active: Tensor, batch: dict, img_branch, gram_teacher: Optional[GramTeacher],
    lam: dict, s_pmask: Optional[Tensor], tau_s: float, tau_t: float,
    global_step: int, use_koleo: bool,
) -> dict:
    """grouped mode — L_SPC + L_decode (+ optional gram/koleo/7b).

    L_SPC   subsumes CLS + patch + intermediate + local-crop CE alignment,
            all energy-weighted by the FFT spectral map.
    L_decode combines pixel reconstruction + SAM patch distillation at
            masked (high-energy) positions.
    """
    device      = s_out["cls"].device
    energy_flat = _build_energy_flat(batch, device)    # (B, N)

    # Padding mask for SPC: True = padding token (not real content)
    pad_mask: Optional[Tensor] = (
        ~s_pmask.flatten(1) if s_pmask is not None else None
    )

    # Intermediate layer pairs for SPC
    s_inter = s_out.get("intermediate_patch_tokens")
    t_inter = t_out.get("intermediate_patch_tokens")
    inter_pairs = list(zip(s_inter, t_inter)) if s_inter and t_inter else None

    # ── L_SPC ─────────────────────────────────────────────────────────────────
    loss_spc = spectral_predictive_coding_loss(
        student_cls=s_out["cls"],
        student_patches=s_out["patch_tokens"],
        teacher_cls=t_out["cls"],
        teacher_patches=t_out["patch_tokens"],
        energy_map=energy_flat,
        local_cls_list=local_cls_list,
        tau_s=tau_s,
        tau_t=tau_t,
        padding_mask=pad_mask,
        intermediate_pairs=inter_pairs,
        ctx_base=lam.get("spc_ctx_base", 0.0),
        local_weight=lam.get("lam3", 0.5),
    )

    # ── L_decode ──────────────────────────────────────────────────────────────
    external = _build_sam_external(batch, img_branch, s_pmask, device)
    loss_decode = grouped_decode_loss(
        student_patches=s_out["patch_tokens"],
        active_mask=active,
        recon_head=img_branch.recon_head if lam.get("lam_recon", 0.0) > 0 else None,
        raw_crops=batch.get("raw_crops"),
        patch_size=img_branch.patch_size,
        external_teachers=external if lam.get("lam_sam", 0.0) > 0 else [],
        alpha_recon=lam.get("spc_alpha_recon", 0.5),
        alpha_ext=lam.get("spc_alpha_ext", 0.5),
    )

    # ── Optional additive terms (gram / koleo / 7b kept for continuity) ───────
    loss_koleo = koleo_loss(s_out["cls"]) if use_koleo else s_out["cls"].new_tensor(0.0)

    loss_7b = s_out["cls"].new_tensor(0.0)
    if img_branch.teacher_d is not None and lam.get("lam_7b", 0.0) > 0:
        with torch.no_grad():
            t7b = img_branch.forward_teacher_d(batch["global_crops"][:, 1])
        if t7b is not None and img_branch.proj_d is not None:
            t_proj = F.normalize(
                img_branch.proj_d(t7b["cls"].float().to(device)), dim=-1
            )
            loss_7b = dino_cls_loss(s_out["cls"], t_proj)

    loss_gram = s_out["cls"].new_tensor(0.0)
    if gram_teacher is not None and gram_teacher.is_active(global_step):
        gram_teacher.maybe_refresh(img_branch.student, global_step)
        X_S = F.normalize(s_out["patch_tokens"], dim=-1)
        X_G = gram_teacher.forward(batch["global_crops"][:, 0], padding_mask=s_pmask)
        loss_gram = gram_loss(X_S, X_G, padding_mask=s_pmask)

    loss = (
        lam.get("lam_align",  1.0) * loss_spc
        + lam.get("lam_decode", 0.5) * loss_decode
        + lam.get("lam_koleo",  0.1) * loss_koleo
        + lam.get("lam_7b",     0.0) * loss_7b
        + lam.get("lam_gram",   1.0) * loss_gram
    )

    return {
        "loss":         loss,
        "loss_spc":     loss_spc.item(),
        "loss_decode":  loss_decode.item(),
        "loss_koleo":   loss_koleo.item(),
        "loss_7b":      loss_7b.item(),
        "loss_gram":    loss_gram.item(),
    }


def _phase1_unified(
    s_out: dict, t_out: dict, local_cls_list: List[Tensor],
    active: Tensor, batch: dict, img_branch, gram_teacher: Optional[GramTeacher],
    lam: dict, s_pmask: Optional[Tensor], tau_s: float, tau_t: float,
    global_step: int, use_koleo: bool,
) -> dict:
    """unified mode — SPC as base objective (lam_align=1.0) + lam_decode * L_decode.

    Reduces to two meaningful hyperparameters: ``lam_decode`` and
    ``spc_ctx_base``.  All alignment losses are subsumed into a single
    energy-weighted CE formula.
    """
    device      = s_out["cls"].device
    energy_flat = _build_energy_flat(batch, device)

    pad_mask: Optional[Tensor] = (
        ~s_pmask.flatten(1) if s_pmask is not None else None
    )

    s_inter = s_out.get("intermediate_patch_tokens")
    t_inter = t_out.get("intermediate_patch_tokens")
    inter_pairs = list(zip(s_inter, t_inter)) if s_inter and t_inter else None

    # ── L_SPC (implicit weight 1.0) ───────────────────────────────────────────
    loss_spc = spectral_predictive_coding_loss(
        student_cls=s_out["cls"],
        student_patches=s_out["patch_tokens"],
        teacher_cls=t_out["cls"],
        teacher_patches=t_out["patch_tokens"],
        energy_map=energy_flat,
        local_cls_list=local_cls_list,
        tau_s=tau_s,
        tau_t=tau_t,
        padding_mask=pad_mask,
        intermediate_pairs=inter_pairs,
        ctx_base=lam.get("spc_ctx_base", 0.0),
        local_weight=lam.get("lam3", 0.5),
    )

    # ── L_decode ──────────────────────────────────────────────────────────────
    external = _build_sam_external(batch, img_branch, s_pmask, device)
    loss_decode = grouped_decode_loss(
        student_patches=s_out["patch_tokens"],
        active_mask=active,
        recon_head=img_branch.recon_head if lam.get("lam_recon", 0.0) > 0 else None,
        raw_crops=batch.get("raw_crops"),
        patch_size=img_branch.patch_size,
        external_teachers=external if lam.get("lam_sam", 0.0) > 0 else [],
        alpha_recon=lam.get("spc_alpha_recon", 0.5),
        alpha_ext=lam.get("spc_alpha_ext", 0.5),
    )

    loss_koleo = koleo_loss(s_out["cls"]) if use_koleo else s_out["cls"].new_tensor(0.0)

    loss_gram = s_out["cls"].new_tensor(0.0)
    if gram_teacher is not None and gram_teacher.is_active(global_step):
        gram_teacher.maybe_refresh(img_branch.student, global_step)
        X_S = F.normalize(s_out["patch_tokens"], dim=-1)
        X_G = gram_teacher.forward(batch["global_crops"][:, 0], padding_mask=s_pmask)
        loss_gram = gram_loss(X_S, X_G, padding_mask=s_pmask)

    loss = (
        loss_spc                                    # lam_align implicit = 1.0
        + lam.get("lam_decode", 0.5) * loss_decode
        + lam.get("lam_koleo",  0.1) * loss_koleo
        + lam.get("lam_gram",   1.0) * loss_gram
    )

    return {
        "loss":         loss,
        "loss_spc":     loss_spc.item(),
        "loss_decode":  loss_decode.item(),
        "loss_koleo":   loss_koleo.item(),
        "loss_gram":    loss_gram.item(),
    }


def phase1_step(
    batch: dict,
    img_branch,                          # ImageBranch
    gram_teacher: Optional[GramTeacher],
    lam: dict,
    global_step: int = 0,
    use_koleo: bool = False,
    feedback: Optional[HardnessFeedback] = None,
) -> dict:
    """
    Image SSL step — dispatches to one of three loss modes.

    ``lam["loss_mode"]`` selects the objective:
      ``"multi"``   (default) all individual λ weights, current behaviour
      ``"grouped"`` 3 terms: L_SPC + L_decode + optional gram/koleo
      ``"unified"`` SPC as base (lam_align=1.0) + lam_decode * L_decode

    Forward passes are shared across all modes.  Only the loss computation
    is mode-specific.
    """
    t_pmask = _get_pmask(batch, 1)
    s_pmask = _get_pmask(batch, 0)

    with torch.no_grad():
        t_out = img_branch.forward_teacher(
            batch["global_crops"][:, 1], padding_mask=t_pmask
        )

    s_out = img_branch.forward_student(
        batch["global_crops"][:, 0],
        padding_mask=s_pmask,
    )

    # ── ALP feedback: saliency from teacher attention, hardness from patch error ─
    if feedback is not None:
        sample_ids = batch.get("sample_ids") or batch.get("sample_id")
        if sample_ids is not None:
            try:
                attn = img_branch.teacher.get_last_attention()
                feedback.update_saliency(sample_ids, attn, global_step)
            except (AttributeError, NotImplementedError):
                pass
            patch_errors = (
                s_out["patch_tokens"].detach() - t_out["patch_tokens"].detach()
            ).pow(2).mean(-1)
            feedback.update(sample_ids, patch_errors, global_step)

    tau_s = lam.get("tau_student", 0.1)
    tau_t = lam.get("tau_teacher", 0.04)

    # Local crops — student only
    n_local = batch["local_crops"].shape[1]
    local_cls_list = []
    for i in range(n_local):
        lpm = batch["local_pmasks"][:, i] if "local_pmasks" in batch else None
        local_out = img_branch.forward_student(
            batch["local_crops"][:, i], padding_mask=lpm
        )
        local_cls_list.append(local_out["cls"])

    active = _active_patch_mask(s_pmask, batch["patch_masks"])

    common = dict(
        s_out=s_out, t_out=t_out, local_cls_list=local_cls_list,
        active=active, batch=batch, img_branch=img_branch,
        gram_teacher=gram_teacher, lam=lam, s_pmask=s_pmask,
        tau_s=tau_s, tau_t=tau_t, global_step=global_step,
        use_koleo=use_koleo,
    )

    loss_mode = lam.get("loss_mode", "multi")
    if loss_mode == "grouped":
        return _phase1_grouped(**common)
    elif loss_mode == "unified":
        return _phase1_unified(**common)
    else:
        return _phase1_multi(**common)


# ── Phase 2: Video branch warm-start ─────────────────────────────────────────

def phase2_step(
    batch: dict,
    vid_branch,            # VideoBranch
    lam: dict,
) -> dict:
    """
    V-JEPA video SSL step.

    Losses computed
    ---------------
    L = lam4·L_clip_cls + lam5·L_tube
    """
    v_pmask = batch.get("padding_masks")
    valid   = batch.get("valid_frames")

    with torch.no_grad():
        t_out = vid_branch.forward_teacher(
            batch["full_clips"], padding_mask=v_pmask, valid_frames=valid
        )

    s_out = vid_branch.forward_student(
        batch["visible_clips"],
        tube_mask=batch["tube_masks"],
        padding_mask=v_pmask,
        valid_frames=valid,
    )

    loss_cls = clip_cls_loss(s_out["clip_cls"], t_out["clip_cls"])

    # Tube loss: compare student predictor output at target positions with
    # the teacher's tokens at those same positions.
    # s_out["predicted"] has shape (B, N_tgt, D) — already at target positions.
    # s_out["tgt_indices"] has shape (B, N_tgt) — tubelet-level target indices.
    # We gather teacher tokens at those indices and compute cosine loss.
    if "predicted" in s_out and "tgt_indices" in s_out:
        tgt_idx = s_out["tgt_indices"]                           # (B, N_tgt)
        D       = t_out["tube_tokens"].shape[-1]
        N_tok   = t_out["tube_tokens"].shape[1]
        # Expand index for gather and clamp to valid range
        idx_exp = tgt_idx.unsqueeze(-1).expand(-1, -1, D).clamp(0, N_tok - 1)
        teacher_at_tgt = torch.gather(t_out["tube_tokens"], 1, idx_exp)  # (B, N_tgt, D)
        loss_tube = jepa_tube_loss(
            s_out["predicted"],
            teacher_at_tgt,
            torch.ones(s_out["predicted"].shape[:2], dtype=torch.bool,
                       device=s_out["predicted"].device),
        )
    else:
        loss_tube = t_out["clip_cls"].new_tensor(0.0)

    loss = lam.get("lam4", 1.0) * loss_cls + lam.get("lam5", 1.0) * loss_tube

    return {
        "loss":      loss,
        "loss_cls":  loss_cls.item(),
        "loss_tube": loss_tube.item(),
    }


# ── Phase 3: Hybrid joint training ───────────────────────────────────────────

def phase3_step(
    img_batch: dict,
    vid_batch: dict,
    img_branch,                             # ImageBranch
    vid_branch,                             # VideoBranch
    cross_distill,                          # CrossBranchDistillation
    proto_head,                             # PrototypeHead
    gram_teacher: Optional[GramTeacher],
    lam: dict,
    global_step: int,
    stage: int,                             # 1, 2, or 3 — from dm.current_stage()
    alignment_pairs: Optional[List] = None, # List[AlignmentPair] from AlignedDualStreamBatch
    feedback: Optional[HardnessFeedback] = None,
) -> dict:
    """
    Hybrid joint step: image + video + cross-branch + prototype + gram.

    Cross-modal losses (stage 2+)
    ------------------------------
    lam6        FILIP per-pair token-level loss (ramped: ×0.5 stage 2, ×1.0 stage 3)
    lam6_nce    InfoNCE contrastive loss (stage 3 only, ≥2 pairs required)
    lam7        SwAV prototype loss — Sinkhorn on teacher, softmax on student
    lam_koleo_cross  KoLeo uniformity on projected img/vid features (stage 2+)

    Fallback behaviour
    ------------------
    When alignment_pairs is empty (disjoint-study batch or stage 1), FILIP and
    InfoNCE return 0 gracefully. SwAV and KoLeo use global mean-pooled features
    and still provide useful uniformity/diversity signal.
    """
    t_pmask = _get_pmask(img_batch, 1)
    s_pmask = _get_pmask(img_batch, 0)

    tau_s = lam.get("tau_student", 0.1)
    tau_t = lam.get("tau_teacher", 0.04)

    # ── Image branch ──────────────────────────────────────────────────────────
    with torch.no_grad():
        t_img = img_branch.forward_teacher(
            img_batch["global_crops"][:, 1], padding_mask=t_pmask
        )

    s_img = img_branch.forward_student(
        img_batch["global_crops"][:, 0], padding_mask=s_pmask
    )

    active_img = _active_patch_mask(s_pmask, img_batch["patch_masks"])

    loss_mode  = lam.get("loss_mode", "multi")
    device_img = s_img["cls"].device

    if loss_mode in ("grouped", "unified"):
        # ── SPC-based image alignment ─────────────────────────────────────────
        energy_flat = _build_energy_flat(img_batch, device_img)
        pad_mask_img: Optional[Tensor] = (
            ~s_pmask.flatten(1) if s_pmask is not None else None
        )
        s_inter = s_img.get("intermediate_patch_tokens")
        t_inter = t_img.get("intermediate_patch_tokens")
        inter_pairs = list(zip(s_inter, t_inter)) if s_inter and t_inter else None

        loss_spc_img = spectral_predictive_coding_loss(
            student_cls=s_img["cls"],
            student_patches=s_img["patch_tokens"],
            teacher_cls=t_img["cls"],
            teacher_patches=t_img["patch_tokens"],
            energy_map=energy_flat,
            local_cls_list=[],
            tau_s=tau_s,
            tau_t=tau_t,
            padding_mask=pad_mask_img,
            intermediate_pairs=inter_pairs,
            ctx_base=lam.get("spc_ctx_base", 0.0),
        )

        external_img = _build_sam_external(img_batch, img_branch, s_pmask, device_img)
        loss_decode_img = grouped_decode_loss(
            student_patches=s_img["patch_tokens"],
            active_mask=active_img,
            recon_head=img_branch.recon_head if lam.get("lam_recon", 0.0) > 0 else None,
            raw_crops=img_batch.get("raw_crops"),
            patch_size=img_branch.patch_size,
            external_teachers=external_img if lam.get("lam_sam", 0.0) > 0 else [],
            alpha_recon=lam.get("spc_alpha_recon", 0.5),
            alpha_ext=lam.get("spc_alpha_ext", 0.5),
        )

        lam_align_eff = lam.get("lam_align", 1.0) if loss_mode == "grouped" else 1.0
        loss_img = (
            lam_align_eff               * loss_spc_img
            + lam.get("lam_decode", 0.5) * loss_decode_img
        )

        # Re-expose scalars for the return dict
        loss_cls_img   = loss_spc_img       # aliased for logging
        loss_patch     = s_img["cls"].new_tensor(0.0)
        loss_deep_img  = s_img["cls"].new_tensor(0.0)
        loss_ctx_img   = s_img["cls"].new_tensor(0.0)
        loss_recon     = loss_decode_img    # aliased for logging
        loss_sam       = s_img["cls"].new_tensor(0.0)

    else:
        # ── multi mode: individual lambda weights ─────────────────────────────
        loss_cls_img = dino_cls_loss_ce_multicrop(
            s_img["cls"], t_img["cls"], [],
            student_temp=tau_s, teacher_temp=tau_t,
        )
        loss_patch = ibot_patch_loss_ce(
            s_img["patch_tokens"], t_img["patch_tokens"], active_img,
            student_temp=tau_s, teacher_temp=tau_t,
        )

        loss_deep_img = s_img["cls"].new_tensor(0.0)
        lam_deep = lam.get("lam_deep", 0.0)
        if lam_deep > 0:
            s_inter = s_img.get("intermediate_patch_tokens")
            t_inter = t_img.get("intermediate_patch_tokens")
            if s_inter and t_inter:
                per_layer_w = lam_deep / max(len(s_inter), 1)
                for sl, tl in zip(s_inter, t_inter):
                    loss_deep_img = loss_deep_img + per_layer_w * ibot_patch_loss_ce(
                        sl, tl, active_img, student_temp=tau_s, teacher_temp=tau_t,
                    )

        loss_ctx_img = s_img["cls"].new_tensor(0.0)
        if lam.get("lam_ctx", 0.0) > 0:
            loss_ctx_img = dense_context_loss(
                s_img["patch_tokens"], t_img["patch_tokens"],
                img_batch["patch_masks"],
                lam_ctx=lam.get("lam_ctx", 1.0),
            )

        loss_recon = s_img["cls"].new_tensor(0.0)
        if lam.get("lam_recon", 0.0) > 0:
            raw_crop = img_batch.get("raw_crops")
            if raw_crop is not None:
                pred_px = img_branch.recon_head(s_img["patch_tokens"])
                loss_recon = pixel_reconstruction_loss(
                    pred_px, raw_crop.to(device_img),
                    active_img, patch_size=img_branch.patch_size,
                )

        loss_sam = s_img["cls"].new_tensor(0.0)
        if img_branch.teacher_sam is not None and lam.get("lam_sam", 0.0) > 0:
            ph = s_pmask.shape[1] if s_pmask is not None else None
            pw = s_pmask.shape[2] if s_pmask is not None else None
            with torch.no_grad():
                t_sam = img_branch.forward_teacher_sam(
                    img_batch["global_crops"][:, 1].to(device_img),
                    target_ph=ph, target_pw=pw,
                )
            if t_sam is not None and img_branch.proj_sam_patch is not None:
                dt = img_branch.proj_sam_patch.weight.dtype
                sam_proj = F.normalize(
                    img_branch.proj_sam_patch(
                        t_sam["patch_tokens"].float().to(dt).to(device_img)
                    ), dim=-1,
                )
                patch_valid = (
                    s_pmask.flatten(1) if s_pmask is not None
                    else torch.ones(s_img["patch_tokens"].shape[:2],
                                    dtype=torch.bool, device=device_img)
                )
                loss_sam = ibot_patch_loss(s_img["patch_tokens"], sam_proj, patch_valid)

        loss_img = (
            lam.get("lam1",      1.0) * loss_cls_img
            + lam.get("lam2",    1.0) * loss_patch
            + loss_deep_img
            + lam.get("lam_ctx",   0.0) * loss_ctx_img
            + lam.get("lam_recon", 0.0) * loss_recon
            + lam.get("lam_sam",   0.0) * loss_sam
        )

    # ── ALP feedback (image branch) ───────────────────────────────────────────
    if feedback is not None:
        sample_ids = img_batch.get("sample_ids") or img_batch.get("sample_id")
        if sample_ids is not None:
            try:
                attn = img_branch.teacher.get_last_attention()
                feedback.update_saliency(sample_ids, attn, global_step)
            except (AttributeError, NotImplementedError):
                pass
            patch_errors = (
                s_img["patch_tokens"].detach() - t_img["patch_tokens"].detach()
            ).pow(2).mean(-1)
            feedback.update(sample_ids, patch_errors, global_step)

    # ── Video branch ──────────────────────────────────────────────────────────
    v_pmask = vid_batch.get("padding_masks")
    valid   = vid_batch.get("valid_frames")

    with torch.no_grad():
        t_vid = vid_branch.forward_teacher(
            vid_batch["full_clips"], padding_mask=v_pmask, valid_frames=valid
        )

    s_vid = vid_branch.forward_student(
        vid_batch["visible_clips"],
        tube_mask=vid_batch["tube_masks"],
        padding_mask=v_pmask, valid_frames=valid,
    )

    loss_cls_vid = clip_cls_loss(s_vid["clip_cls"], t_vid["clip_cls"])

    if "predicted" in s_vid and "tgt_indices" in s_vid:
        tgt_idx = s_vid["tgt_indices"]                           # (B, N_tgt)
        D       = t_vid["tube_tokens"].shape[-1]
        N_tok   = t_vid["tube_tokens"].shape[1]
        idx_exp = tgt_idx.unsqueeze(-1).expand(-1, -1, D).clamp(0, N_tok - 1)
        teacher_at_tgt = torch.gather(t_vid["tube_tokens"], 1, idx_exp)
        loss_tube = jepa_tube_loss(
            s_vid["predicted"],
            teacher_at_tgt,
            torch.ones(s_vid["predicted"].shape[:2], dtype=torch.bool,
                       device=s_vid["predicted"].device),
        )
    else:
        loss_tube = t_vid["clip_cls"].new_tensor(0.0)

    loss_vid = lam.get("lam4", 1.0) * loss_cls_vid + lam.get("lam5", 1.0) * loss_tube

    # ── Infer patch-grid spatial dimensions (ph, pw) from tube_tokens shape ──
    # tube_tokens: (B, T*ph*pw, D_vid).  We need ph*pw to slice per frame.
    # Derive from the spatial padding mask when available, else approximate.
    if v_pmask is not None:
        ph_vid = v_pmask.shape[1]   # (B, ph, pw) — ph from mask
        pw_vid = v_pmask.shape[2]
    else:
        # Fall back: assume square spatial grid from the tube token count and
        # the number of frames in the video batch.
        T_vid   = vid_batch["full_clips"].shape[1]
        N_total = s_vid["tube_tokens"].shape[1]
        n_spat  = max(1, N_total // max(1, T_vid))
        side    = max(1, int(n_spat ** 0.5))
        ph_vid  = side
        pw_vid  = max(1, n_spat // side)

    pairs = alignment_pairs or []

    # ── FILIP token-level cross-branch loss ───────────────────────────────────
    # In grouped/unified modes use lam_cross_grouped instead of lam6 so the
    # caller only needs to tune the single L_cross weight.
    if loss_mode in ("grouped", "unified"):
        lam6_eff = lam.get("lam_cross_grouped", 1.0) if stage >= 2 else 0.0
    else:
        lam6_eff = lam.get("lam6", 1.0) * (0.5 if stage == 2 else 1.0) if stage >= 2 else 0.0
    if lam6_eff > 0:
        loss_filip = filip_cross_loss_from_pairs(
            t_img["patch_tokens"], s_vid["tube_tokens"],
            cross_distill.proj_img, cross_distill.proj_vid,
            getattr(cross_distill, "predictor_vid", None),
            pairs, ph_vid, pw_vid,
        )
        if not pairs:
            # Fallback when no aligned pairs: global mean-pool cosine loss
            loss_filip = cross_branch_loss_from_tokens(
                t_img["patch_tokens"], s_vid["tube_tokens"],
                cross_distill.proj_img, cross_distill.proj_vid,
                getattr(cross_distill, "predictor_vid", None),
            )
    else:
        loss_filip = s_img["cls"].new_tensor(0.0)

    # ── InfoNCE contrastive loss (Stage 3 only, ≥2 pairs) ────────────────────
    lam6_nce = lam.get("lam6_nce", 0.5) if stage >= 3 else 0.0
    if lam6_nce > 0 and len(pairs) >= 2:
        loss_nce = infonce_cross_loss_from_pairs(
            t_img["patch_tokens"], s_vid["tube_tokens"],
            cross_distill.proj_img, cross_distill.proj_vid,
            getattr(cross_distill, "predictor_vid", None),
            pairs, ph_vid, pw_vid,
            temperature=lam.get("lam6_nce_temp", 0.07),
        )
    else:
        loss_nce = s_img["cls"].new_tensor(0.0)

    # ── SwAV prototype loss ───────────────────────────────────────────────────
    lam7_eff = lam.get("lam7", 0.5) if stage >= 2 else 0.0
    if lam7_eff > 0:
        dt = cross_distill.proj_img.weight.dtype
        # Project both modalities to align_dim; teacher tokens fed with no_grad above
        img_proj_mean = cross_distill.proj_img(
            t_img["patch_tokens"].float().mean(1).to(dt)
        )  # (B_img, align_dim)
        vid_proj_mean = cross_distill.proj_vid(
            s_vid["tube_tokens"].float().mean(1).to(dt)
        )  # (B_vid, align_dim)
        loss_proto = swav_proto_loss_from_tokens(
            img_proj_mean.unsqueeze(1), vid_proj_mean.unsqueeze(1),
            proto_head.prototypes, proto_head.temperature,
        )
    else:
        loss_proto = s_img["cls"].new_tensor(0.0)

    # ── KoLeo uniformity in align_dim space (Stage 2+) ───────────────────────
    lam_koleo_cross = lam.get("lam_koleo_cross", 0.1) if stage >= 2 else 0.0
    if lam_koleo_cross > 0:
        dt = cross_distill.proj_img.weight.dtype
        img_proj_kl = F.normalize(
            cross_distill.proj_img(t_img["patch_tokens"].float().mean(1).to(dt)), dim=-1
        )
        vid_proj_kl = F.normalize(
            cross_distill.proj_vid(s_vid["tube_tokens"].float().mean(1).to(dt)), dim=-1
        )
        loss_koleo_cross = koleo_loss(img_proj_kl) + koleo_loss(vid_proj_kl)
    else:
        loss_koleo_cross = s_img["cls"].new_tensor(0.0)

    # ── 7B distillation ───────────────────────────────────────────────────────
    loss_7b = s_img["cls"].new_tensor(0.0)
    if img_branch.teacher_d is not None and lam.get("lam_7b", 0.0) > 0:
        with torch.no_grad():
            t7b = img_branch.forward_teacher_d(img_batch["global_crops"][:, 1])
        if t7b is not None and img_branch.proj_d is not None:
            t_proj = F.normalize(
                img_branch.proj_d(t7b["cls"].float().to(s_img["cls"].device)), dim=-1
            )
            loss_7b = dino_cls_loss(s_img["cls"], t_proj)

    # ── Gram anchoring ────────────────────────────────────────────────────────
    loss_gram = s_img["cls"].new_tensor(0.0)
    if gram_teacher is not None and gram_teacher.is_active(global_step):
        gram_teacher.maybe_refresh(img_branch.student, global_step)
        X_S = F.normalize(s_img["patch_tokens"], dim=-1)
        X_G = gram_teacher.forward(img_batch["global_crops"][:, 0], padding_mask=s_pmask)
        loss_gram = gram_loss(X_S, X_G, padding_mask=s_pmask)

    # ── Total loss ────────────────────────────────────────────────────────────
    loss = (
        loss_img                                # already includes deep/ctx/recon/sam
        + loss_vid
        + lam6_eff         * loss_filip
        + lam6_nce         * loss_nce
        + lam7_eff         * loss_proto
        + lam_koleo_cross  * loss_koleo_cross
        + lam.get("lam_7b",   0.0) * loss_7b
        + lam.get("lam_gram", 1.0) * loss_gram
    )

    return {
        "loss":              loss,
        "loss_img":          loss_img.item(),
        "loss_vid":          loss_vid.item(),
        "loss_deep_img":     loss_deep_img.item(),
        "loss_ctx_img":      loss_ctx_img.item(),
        "loss_recon":        loss_recon.item(),
        "loss_sam":          loss_sam.item(),
        "loss_filip":        loss_filip.item(),
        "loss_nce":          loss_nce.item(),
        "loss_proto":        loss_proto.item(),
        "loss_koleo_cross":  loss_koleo_cross.item(),
        "loss_7b":           loss_7b.item(),
        "loss_gram":         loss_gram.item(),
        "n_align_pairs":     len(pairs),
        "stage":             stage,
    }


# ── Phase 4: Downstream head training ────────────────────────────────────────

def phase4_step(
    batch: dict,
    img_branch,        # ImageBranch  (backbone frozen by caller)
    seg_head,          # LinearSegHead / DPTSegHead
    cls_head,          # LinearClsHead / MLPClsHead
) -> dict:
    """
    Supervised downstream fine-tuning step.

    The backbone is frozen by the trainer before calling this function.
    Only seg_head and cls_head parameters receive gradients.

    Returns loss tensor (scalar) for backward.
    """
    pmask = _get_pmask(batch, 0)

    with torch.no_grad():
        feats = img_branch.forward_teacher(
            batch["global_crops"][:, 0], padding_mask=pmask
        )

    loss = feats["cls"].new_tensor(0.0, requires_grad=True)

    if batch.get("seg_masks") is not None:
        pred_seg = seg_head(feats["patch_tokens"], padding_mask=pmask)
        # seg_masks: (B, 1, ph, pw) float — binary
        loss = loss + F.binary_cross_entropy_with_logits(
            pred_seg, batch["seg_masks"]
        )

    valid_cls = batch["cls_labels"] >= 0
    if valid_cls.any():
        pred_cls = cls_head(feats["cls"])
        loss = loss + F.cross_entropy(
            pred_cls[valid_cls], batch["cls_labels"][valid_cls]
        )

    return {"loss": loss, "loss_finetune": loss.item()}
