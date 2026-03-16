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
  lam1      DINO CLS loss weight
  lam2      iBOT patch loss weight
  lam3      local crop CLS weight
  lam4      video CLS loss weight
  lam5      tube prediction loss weight
  lam6      cross-branch distillation weight
  lam7      prototype consistency weight
  lam_7b    7B teacher distillation weight
  lam_gram  Gram anchoring weight
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler

from models.losses.image_losses import (
    dino_cls_loss,
    dino_cls_loss_multicrop,
    ibot_patch_loss,
    koleo_loss,
)
from models.losses.video_losses import jepa_tube_loss, clip_cls_loss
from models.losses.cross_branch import cross_branch_loss_from_tokens
from models.losses.proto_loss import proto_loss_from_tokens
from .gram import GramTeacher, gram_loss


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


# ── Phase 1: Image branch warm-start ─────────────────────────────────────────

def phase1_step(
    batch: dict,
    img_branch,                          # ImageBranch
    gram_teacher: Optional[GramTeacher],
    lam: dict,
    global_step: int = 0,
    use_koleo: bool = False,
) -> dict:
    """
    DINO-style image SSL step.

    Losses computed
    ---------------
    L = lam1·L_cls + lam2·L_patch + lam3·L_local
      + lam_7b·L_7b  (if frozen teacher available)
      + lam_gram·L_gram  (if active at this step)
      + koleo·L_koleo    (optional uniformity regulariser)

    Returns dict of scalar losses (no tensors — already .item()'d).
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

    # Local crops
    n_local = batch["local_crops"].shape[1]
    local_cls_list = []
    for i in range(n_local):
        lpm = batch["local_pmasks"][:, i] if "local_pmasks" in batch else None
        local_out = img_branch.forward_student(
            batch["local_crops"][:, i], padding_mask=lpm
        )
        local_cls_list.append(local_out["cls"])

    # CLS loss (global + local)
    loss_cls = dino_cls_loss_multicrop(
        s_out["cls"], t_out["cls"], local_cls_list,
        local_weight=lam.get("lam3", 0.5),
    )

    # Patch prediction loss at real + freq-masked positions
    active = _active_patch_mask(s_pmask, batch["patch_masks"])
    loss_patch = ibot_patch_loss(s_out["patch_tokens"], t_out["patch_tokens"], active)

    # KoLeo uniformity (optional)
    loss_koleo = koleo_loss(s_out["cls"]) if use_koleo else s_out["cls"].new_tensor(0.0)

    # 7B frozen teacher distillation (optional)
    loss_7b = s_out["cls"].new_tensor(0.0)
    if img_branch.teacher_d is not None and lam.get("lam_7b", 0.0) > 0:
        with torch.no_grad():
            t7b = img_branch.forward_teacher_d(batch["global_crops"][:, 1])
        if t7b is not None and img_branch.proj_d is not None:
            t_proj = F.normalize(
                img_branch.proj_d(t7b["cls"].float().to(s_out["cls"].device)), dim=-1
            )
            loss_7b = dino_cls_loss(s_out["cls"], t_proj)

    # Gram anchoring
    loss_gram = s_out["cls"].new_tensor(0.0)
    if gram_teacher is not None and gram_teacher.is_active(global_step):
        gram_teacher.maybe_refresh(img_branch.student, global_step)
        X_S = F.normalize(s_out["patch_tokens"], dim=-1)
        X_G = gram_teacher.forward(batch["global_crops"][:, 0], padding_mask=s_pmask)
        loss_gram = gram_loss(X_S, X_G, padding_mask=s_pmask)

    loss = (
        lam.get("lam1", 1.0) * loss_cls
        + lam.get("lam2", 1.0) * loss_patch
        + lam.get("lam_koleo", 0.1) * loss_koleo
        + lam.get("lam_7b", 0.0) * loss_7b
        + lam.get("lam_gram", 1.0) * loss_gram
    )

    return {
        "loss":       loss,           # tensor — caller does .backward()
        "loss_cls":   loss_cls.item(),
        "loss_patch": loss_patch.item(),
        "loss_koleo": loss_koleo.item(),
        "loss_7b":    loss_7b.item(),
        "loss_gram":  loss_gram.item(),
    }


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

    tube_flat = batch["tube_masks"].flatten(1, -1)
    if v_pmask is not None:
        real_flat = v_pmask.unsqueeze(1).expand_as(batch["tube_masks"]).flatten(1, -1)
        active    = tube_flat & real_flat
    else:
        active = tube_flat

    if "predicted" in s_out and active.any():
        loss_tube = jepa_tube_loss(s_out["predicted"], t_out["tube_tokens"], active)
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
    img_branch,            # ImageBranch
    vid_branch,            # VideoBranch
    cross_distill,         # CrossBranchDistillation
    proto_head,            # PrototypeHead
    gram_teacher: Optional[GramTeacher],
    lam: dict,
    global_step: int,
    stage: int,            # 1, 2, or 3 — from dm.current_stage()
) -> dict:
    """
    Hybrid joint step: image + video + cross-branch + prototype + gram.

    stage controls loss ramping:
      stage 1: lam6=0.0, lam7=0.0  (branches warm up independently)
      stage 2: lam6=lam6*0.5, lam7=lam7  (cross-branch introduced at half weight)
      stage 3: lam6=lam6, lam7=lam7  (full)
    """
    t_pmask = _get_pmask(img_batch, 1)
    s_pmask = _get_pmask(img_batch, 0)

    # ── Image branch ──────────────────────────────────────────────────────────
    with torch.no_grad():
        t_img = img_branch.forward_teacher(
            img_batch["global_crops"][:, 1], padding_mask=t_pmask
        )

    s_img = img_branch.forward_student(
        img_batch["global_crops"][:, 0], padding_mask=s_pmask
    )

    loss_cls_img = dino_cls_loss(s_img["cls"], t_img["cls"])
    active_img   = _active_patch_mask(s_pmask, img_batch["patch_masks"])
    loss_patch   = ibot_patch_loss(s_img["patch_tokens"], t_img["patch_tokens"], active_img)
    loss_img     = lam.get("lam1", 1.0) * loss_cls_img + lam.get("lam2", 1.0) * loss_patch

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
    tube_flat = vid_batch["tube_masks"].flatten(1, -1)
    if v_pmask is not None:
        real_v   = v_pmask.unsqueeze(1).expand_as(vid_batch["tube_masks"]).flatten(1, -1)
        active_v = tube_flat & real_v
    else:
        active_v = tube_flat

    if "predicted" in s_vid and active_v.any():
        loss_tube = jepa_tube_loss(s_vid["predicted"], t_vid["tube_tokens"], active_v)
    else:
        loss_tube = t_vid["clip_cls"].new_tensor(0.0)

    loss_vid = lam.get("lam4", 1.0) * loss_cls_vid + lam.get("lam5", 1.0) * loss_tube

    # ── Cross-branch distillation ─────────────────────────────────────────────
    lam6_eff = lam.get("lam6", 1.0) * (0.5 if stage == 2 else 1.0) if stage >= 2 else 0.0
    if lam6_eff > 0:
        loss_cross = cross_branch_loss_from_tokens(
            t_img["patch_tokens"], s_vid["tube_tokens"],
            cross_distill.proj_img, cross_distill.proj_vid,
        )
    else:
        loss_cross = s_img["cls"].new_tensor(0.0)

    # ── Prototype consistency ─────────────────────────────────────────────────
    lam7_eff = lam.get("lam7", 0.5) if stage >= 2 else 0.0
    if lam7_eff > 0:
        loss_proto = proto_loss_from_tokens(
            t_img["patch_tokens"], s_vid["tube_tokens"],
            proto_head.prototypes, proto_head.temperature,
        )
    else:
        loss_proto = s_img["cls"].new_tensor(0.0)

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

    loss = (
        loss_img
        + loss_vid
        + lam6_eff * loss_cross
        + lam7_eff * loss_proto
        + lam.get("lam_7b", 0.0) * loss_7b
        + lam.get("lam_gram", 1.0) * loss_gram
    )

    return {
        "loss":        loss,
        "loss_img":    loss_img.item(),
        "loss_vid":    loss_vid.item(),
        "loss_cross":  loss_cross.item(),
        "loss_proto":  loss_proto.item(),
        "loss_7b":     loss_7b.item(),
        "loss_gram":   loss_gram.item(),
        "stage":       stage,
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
