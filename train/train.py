"""
train.py  ·  Ultatron foundation model training — all 4 phases
===========================================================

Usage (single node, for testing):
    python train.py --config configs/data_config.yaml

Usage (multi-node via torchrun, called by run_training_job.sh):
    torchrun --nnodes=$NNODES --nproc_per_node=4 \\
             --node_rank=$SLURM_NODEID \\
             --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
             train.py --config configs/data_config.yaml

Training phases
---------------
  Phase 1  (0%  – 10%)  Image branch warm-start
                          DINO CLS loss + patch prediction loss
                          Optional 7B distillation loss (λ_7b)
  Phase 2  (10% – 20%)  Video branch warm-start
                          V-JEPA clip CLS loss + tube prediction loss
  Phase 3  (20% – 90%)  Hybrid joint training
                          Image + Video + cross-branch (λ6) + prototype (λ7)
                          Gram anchoring activated at step 100k (λ_gram)
  Phase 4  (90% – 100%) Downstream head attachment
                          Backbone frozen, seg + cls heads trained supervised

Resolution curriculum (updated inside training loop):
  Steps   0– 30k: max_global_crop_px = 512
  Steps  30–150k: max_global_crop_px = 672
  Steps 150–270k: max_global_crop_px = 896

Checkpointing:
  - Best checkpoint (by val loss): $SCRATCH/checkpoints/current_run/best.pt
  - Periodic: every checkpoint_every steps
  - Phase boundary: always checkpoint at end of each phase

Distributed:
  DDP wraps student models only (teacher is not DDP — it's a local replica).
  The 7B frozen teacher is NOT DDP-wrapped (it's read-only).
  Gradient sync happens only for student parameters.

Loss weights (overrideable via config):
  λ1 = 1.0   DINO CLS
  λ2 = 1.0   patch prediction
  λ3 = 0.5   local crop
  λ4 = 1.0   video CLS
  λ5 = 1.0   video tube prediction
  λ6 = 1.0   cross-branch distillation (Phase 3+)
  λ7 = 0.5   prototype consistency (Phase 3+)
  λ8 = 0.5   7B teacher distillation (optional)
  λ_gram = 1.0  Gram anchoring (after step 100k)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

import yaml

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    ModelConfig,
    build_image_branch, build_video_branch,
    ImageBranch, VideoBranch,
    CrossBranchDistillation, PrototypeHead,
    ema_update,
)
from models.heads import build_seg_head, build_cls_head
from train.gram import GramTeacher, gram_loss
from data.pipeline.datamodule import USFoundationDataModule
from data.pipeline.transforms import ImageSSLTransformConfig, VideoSSLTransformConfig
from data.infra.cscs_paths import CSCSConfig

log = logging.getLogger(__name__)


# ── Distributed helpers ────────────────────────────────────────────────────────

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def all_reduce_mean(tensor: torch.Tensor) -> float:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / dist.get_world_size()
    return tensor.item()


# ── Config loading ─────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Loss functions ─────────────────────────────────────────────────────────────

def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric cosine distance: mean(1 - cos(a,b))."""
    a_n = F.normalize(a.float(), dim=-1)
    b_n = F.normalize(b.float(), dim=-1)
    return (1 - (a_n * b_n).sum(-1)).mean()


def patch_prediction_loss(
    s_patches: torch.Tensor,   # (B, N, D)
    t_patches: torch.Tensor,   # (B, N, D)
    active_mask: torch.Tensor, # (B, N) bool  — real AND freq-masked
) -> torch.Tensor:
    if not active_mask.any():
        return torch.tensor(0.0, device=s_patches.device, requires_grad=True)
    s_flat = s_patches.reshape(-1, s_patches.shape[-1])
    t_flat = t_patches.reshape(-1, t_patches.shape[-1])
    return cosine_loss(s_flat[active_mask.flatten()], t_flat[active_mask.flatten()])


def distill_7b_loss(
    student_cls: torch.Tensor,   # (B, D_L)
    teacher7b_cls: torch.Tensor, # (B, D_7B)
    proj: nn.Module,             # linear D_7B → D_L
) -> torch.Tensor:
    t_proj = proj(teacher7b_cls.float().to(student_cls.device))
    return cosine_loss(student_cls, t_proj)


# ── Learning rate schedule ─────────────────────────────────────────────────────

def get_lr(step: int, total_steps: int, base_lr: float,
           warmup_steps: int = 5000, min_lr_factor: float = 0.01) -> float:
    """
    Linear warmup + cosine decay.
    DINOv3 uses constant LR without decay — set min_lr_factor=1.0 to replicate.
    We use cosine decay for the downstream phase (Phase 4).
    """
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_factor * base_lr + (1 - min_lr_factor) * base_lr * 0.5 * (
        1 + math.cos(math.pi * progress)
    )


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for g in optimizer.param_groups:
        g["lr"] = lr


# ── Resolution curriculum ──────────────────────────────────────────────────────

def update_resolution(dm: USFoundationDataModule, global_step: int):
    """
    Update max_global_crop_px according to the resolution curriculum.
    Called every step; only has effect at boundary transitions.
    """
    if global_step < 30_000:
        px = 512
    elif global_step < 150_000:
        px = 672
    else:
        px = 896

    img_cfg = dm.image_cfg
    if img_cfg.max_global_crop_px != px:
        img_cfg.max_global_crop_px = px
        if is_main():
            log.info(f"[Step {global_step}] Resolution curriculum: max_global_crop_px → {px}")


# ── Padding mask helpers ───────────────────────────────────────────────────────

def _get_padding_mask(batch: dict, crop_idx: int = 0) -> Optional[torch.Tensor]:
    """Extract (B, ph, pw) padding mask for a specific global crop index."""
    pm = batch.get("global_pmasks")
    return pm[:, crop_idx] if pm is not None else None


# ── EMA momentum schedule ──────────────────────────────────────────────────────

def get_ema_momentum(step: int, base: float = 0.9995) -> float:
    """DINOv3 uses constant EMA momentum throughout training."""
    return base


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    global_step: int,
    phase: int,
    img_branch: ImageBranch,
    vid_branch: VideoBranch,
    cross_distill: CrossBranchDistillation,
    proto_head: PrototypeHead,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    best_val_loss: float,
    gram_teacher: Optional[GramTeacher] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP if needed
    def _unwrap(m):
        return m.module if isinstance(m, DDP) else m

    state = {
        "global_step":     global_step,
        "phase":           phase,
        "best_val_loss":   best_val_loss,
        "img_student":     _unwrap(img_branch).student.vit.state_dict(),
        "img_teacher":     _unwrap(img_branch).teacher.vit.state_dict(),
        "vid_student":     _unwrap(vid_branch).student.model.state_dict(),
        "vid_teacher":     _unwrap(vid_branch).teacher.model.state_dict(),
        "cross_distill":   _unwrap(cross_distill).state_dict(),
        "proto_head":      _unwrap(proto_head).state_dict(),
        "optimizer":       optimizer.state_dict(),
        "scaler":          scaler.state_dict(),
    }
    if gram_teacher is not None and gram_teacher._snapshot is not None:
        state["gram_teacher_snapshot"] = gram_teacher._snapshot.vit.state_dict()
        state["gram_last_refresh"]     = gram_teacher._last_refresh

    torch.save(state, path)
    if is_main():
        log.info(f"Checkpoint saved → {path}  (step {global_step})")


def load_checkpoint(
    path: Path,
    img_branch: ImageBranch,
    vid_branch: VideoBranch,
    cross_distill: CrossBranchDistillation,
    proto_head: PrototypeHead,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> tuple[int, int, float]:
    """Returns (global_step, phase, best_val_loss)."""
    state = torch.load(path, map_location="cpu")

    def _unwrap(m):
        return m.module if isinstance(m, DDP) else m

    _unwrap(img_branch).student.vit.load_state_dict(state["img_student"])
    _unwrap(img_branch).teacher.vit.load_state_dict(state["img_teacher"])
    _unwrap(vid_branch).student.model.load_state_dict(state["vid_student"])
    _unwrap(vid_branch).teacher.model.load_state_dict(state["vid_teacher"])
    _unwrap(cross_distill).load_state_dict(state["cross_distill"])
    _unwrap(proto_head).load_state_dict(state["proto_head"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])

    log.info(f"Checkpoint loaded from {path}  (step {state['global_step']})")
    return state["global_step"], state["phase"], state["best_val_loss"]


# ── Phase 1 step ───────────────────────────────────────────────────────────────

def phase1_step(
    batch: dict,
    img_branch: ImageBranch,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    global_step: int,
    lam: dict,
    gram_teacher: Optional[GramTeacher],
) -> dict:
    optimizer.zero_grad()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # Teacher: clean global crop [1]
        t_pmask = _get_padding_mask(batch, 1)
        s_pmask = _get_padding_mask(batch, 0)

        t_out = img_branch.forward_teacher(batch["global_crops"][:, 1], padding_mask=t_pmask)

        # Student: freq-masked global crop [0]
        s_out = img_branch.forward_student(
            batch["global_crops"][:, 0],
            padding_mask=s_pmask,
            patch_mask=batch["patch_masks"],
        )

        # CLS loss
        loss_cls = cosine_loss(s_out["cls"], t_out["cls"])

        # Patch prediction loss (real + freq-masked positions only)
        freq_m = batch["patch_masks"].flatten(1)                # (B, N)
        real_m = (s_pmask.flatten(1) & freq_m) if s_pmask is not None else freq_m
        loss_patch = patch_prediction_loss(
            s_out["patch_tokens"], t_out["patch_tokens"], real_m
        )

        # Local crop losses
        n_local = batch["local_crops"].shape[1]
        loss_local = torch.tensor(0.0, device=s_out["cls"].device)
        for i in range(n_local):
            lpm   = batch["local_pmasks"][:, i] if "local_pmasks" in batch else None
            s_loc = img_branch.forward_student(batch["local_crops"][:, i], padding_mask=lpm)
            loss_local = loss_local + cosine_loss(s_loc["cls"], t_out["cls"])
        loss_local = loss_local / max(n_local, 1)

        # 7B distillation loss (optional)
        loss_7b = torch.tensor(0.0, device=s_out["cls"].device)
        if img_branch.teacher_d is not None and lam.get("lam_7b", 0) > 0:
            t7b = img_branch.forward_teacher_d(batch["global_crops"][:, 1])
            if t7b is not None:
                loss_7b = distill_7b_loss(s_out["cls"], t7b["cls"], img_branch.proj_d)

        # Gram anchoring
        loss_gram = torch.tensor(0.0, device=s_out["cls"].device)
        if gram_teacher is not None and gram_teacher.is_active(global_step):
            gram_teacher.maybe_refresh(
                (img_branch.module if isinstance(img_branch, DDP) else img_branch).student,
                global_step
            )
            X_S = F.normalize(s_out["patch_tokens"], dim=-1)
            X_G = gram_teacher.forward(batch["global_crops"][:, 0], padding_mask=s_pmask)
            loss_gram = gram_loss(X_S, X_G, padding_mask=s_pmask)

        loss = (
            lam["lam1"] * loss_cls
            + lam["lam2"] * loss_patch
            + lam["lam3"] * loss_local
            + lam.get("lam_7b", 0) * loss_7b
            + lam.get("lam_gram", 1.0) * loss_gram
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in img_branch.parameters() if p.requires_grad], max_norm=1.0
    )
    scaler.step(optimizer)
    scaler.update()

    return {
        "loss":       loss.item(),
        "loss_cls":   loss_cls.item(),
        "loss_patch": loss_patch.item(),
        "loss_local": loss_local.item(),
        "loss_7b":    loss_7b.item(),
        "loss_gram":  loss_gram.item(),
    }


# ── Phase 2 step ───────────────────────────────────────────────────────────────

def phase2_step(
    batch: dict,
    vid_branch: VideoBranch,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    lam: dict,
) -> dict:
    optimizer.zero_grad()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        v_pmask = batch.get("padding_masks")
        valid   = batch.get("valid_frames")

        # Teacher: full unmasked clip
        t_out = vid_branch.forward_teacher(
            batch["full_clips"], padding_mask=v_pmask, valid_frames=valid
        )
        # Student: freq-masked visible clip
        s_out = vid_branch.forward_student(
            batch["visible_clips"],
            tube_mask=batch["tube_masks"],
            padding_mask=v_pmask,
            valid_frames=valid,
        )

        loss_cls = cosine_loss(s_out["clip_cls"], t_out["clip_cls"])

        # Tube prediction: predict at masked AND real (non-padding, valid-frame) positions
        tube_masks = batch["tube_masks"]                          # (B, T, ph, pw)
        real = torch.ones_like(tube_masks)                        # (B, T, ph, pw)
        if v_pmask is not None:
            real = real & v_pmask[:, None, :, :]                  # exclude spatial padding
        if valid is not None:
            real = real & valid[:, :, None, None]                 # exclude padding frames
        active = tube_masks.flatten(1) & real.flatten(1)          # (B, T*ph*pw)

        if "predicted" in s_out and active.any():
            loss_tube = patch_prediction_loss(
                s_out["predicted"], t_out["tube_tokens"], active
            )
        else:
            loss_tube = torch.tensor(0.0, device=t_out["clip_cls"].device)

        loss = lam["lam4"] * loss_cls + lam["lam5"] * loss_tube

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for p in vid_branch.parameters() if p.requires_grad], max_norm=1.0
    )
    scaler.step(optimizer)
    scaler.update()

    return {
        "loss":       loss.item(),
        "loss_cls":   loss_cls.item(),
        "loss_tube":  loss_tube.item(),
    }


# ── Phase 3 step ───────────────────────────────────────────────────────────────

def phase3_step(
    dual,
    img_branch: ImageBranch,
    vid_branch: VideoBranch,
    cross_distill: CrossBranchDistillation,
    proto_head: PrototypeHead,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    global_step: int,
    lam: dict,
    stage: int,
    gram_teacher: Optional[GramTeacher],
) -> dict:
    img_b = dual.image_batch
    vid_b = dual.video_batch
    optimizer.zero_grad()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        # ── Image branch ──────────────────────────────────────────────────────
        t_pmask = _get_padding_mask(img_b, 1)
        s_pmask = _get_padding_mask(img_b, 0)
        t_img = img_branch.forward_teacher(img_b["global_crops"][:, 1], padding_mask=t_pmask)
        s_img = img_branch.forward_student(
            img_b["global_crops"][:, 0], padding_mask=s_pmask,
            patch_mask=img_b["patch_masks"]
        )
        loss_cls_img = cosine_loss(s_img["cls"], t_img["cls"])
        freq_m  = img_b["patch_masks"].flatten(1)
        real_m  = (s_pmask.flatten(1) & freq_m) if s_pmask is not None else freq_m
        loss_patch_img = patch_prediction_loss(
            s_img["patch_tokens"], t_img["patch_tokens"], real_m
        )
        loss_img = loss_cls_img + loss_patch_img

        # ── Video branch ──────────────────────────────────────────────────────
        v_pmask = vid_b.get("padding_masks")
        valid   = vid_b.get("valid_frames")
        t_vid = vid_branch.forward_teacher(
            vid_b["full_clips"], padding_mask=v_pmask, valid_frames=valid
        )
        s_vid = vid_branch.forward_student(
            vid_b["visible_clips"], tube_mask=vid_b["tube_masks"],
            padding_mask=v_pmask, valid_frames=valid
        )
        loss_cls_vid = cosine_loss(s_vid["clip_cls"], t_vid["clip_cls"])
        vid_tube_masks = vid_b["tube_masks"]                       # (B, T, ph, pw)
        real_v = torch.ones_like(vid_tube_masks)
        if v_pmask is not None:
            real_v = real_v & v_pmask[:, None, :, :]              # exclude spatial padding
        if valid is not None:
            real_v = real_v & valid[:, :, None, None]             # exclude padding frames
        active_v = vid_tube_masks.flatten(1) & real_v.flatten(1)  # (B, T*ph*pw)
        if "predicted" in s_vid and active_v.any():
            loss_tube = patch_prediction_loss(
                s_vid["predicted"], t_vid["tube_tokens"], active_v
            )
        else:
            loss_tube = torch.tensor(0.0, device=t_vid["clip_cls"].device)
        loss_vid = loss_cls_vid + loss_tube

        # ── Cross-branch + prototype ───────────────────────────────────────────
        loss_cross = cross_distill(t_img["patch_tokens"], s_vid["tube_tokens"]) \
                     if stage >= 2 else torch.tensor(0.0)
        loss_proto = proto_head.consistency_loss(
            t_img["patch_tokens"], s_vid["tube_tokens"]
        ) if stage >= 2 else torch.tensor(0.0)

        # ── 7B distillation ────────────────────────────────────────────────────
        loss_7b = torch.tensor(0.0, device=loss_img.device)
        if img_branch.teacher_d is not None and lam.get("lam_7b", 0) > 0:
            t7b = img_branch.forward_teacher_d(img_b["global_crops"][:, 1])
            if t7b is not None:
                loss_7b = distill_7b_loss(s_img["cls"], t7b["cls"], img_branch.proj_d)

        # ── Gram anchoring ─────────────────────────────────────────────────────
        loss_gram = torch.tensor(0.0, device=loss_img.device)
        if gram_teacher is not None and gram_teacher.is_active(global_step):
            gram_teacher.maybe_refresh(
                (img_branch.module if isinstance(img_branch, DDP) else img_branch).student,
                global_step
            )
            X_S = F.normalize(s_img["patch_tokens"], dim=-1)
            X_G = gram_teacher.forward(img_b["global_crops"][:, 0], padding_mask=s_pmask)
            loss_gram = gram_loss(X_S, X_G, padding_mask=s_pmask)

        lam6 = lam["lam6"] * (1.0 if stage == 3 else 0.5)
        lam7 = lam["lam7"] if stage >= 2 else 0.0

        loss = (
            lam["lam1"] * loss_img
            + lam["lam4"] * loss_vid
            + lam6 * loss_cross
            + lam7 * loss_proto
            + lam.get("lam_7b", 0) * loss_7b
            + lam.get("lam_gram", 1.0) * loss_gram
        )

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    all_params = (
        list(img_branch.parameters()) + list(vid_branch.parameters())
        + list(cross_distill.parameters()) + list(proto_head.parameters())
    )
    torch.nn.utils.clip_grad_norm_(
        [p for p in all_params if p.requires_grad], max_norm=1.0
    )
    scaler.step(optimizer)
    scaler.update()

    return {
        "loss":        loss.item(),
        "loss_img":    loss_img.item(),
        "loss_vid":    loss_vid.item(),
        "loss_cross":  loss_cross.item(),
        "loss_proto":  loss_proto.item(),
        "loss_7b":     loss_7b.item(),
        "loss_gram":   loss_gram.item(),
    }


# ── Phase 4 step (downstream heads) ───────────────────────────────────────────

def phase4_step(
    batch: dict,
    img_branch: ImageBranch,
    seg_head: nn.Module,
    cls_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> dict:
    optimizer.zero_grad()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        pmask = _get_padding_mask(batch, 0)
        with torch.no_grad():
            feats = img_branch.forward_teacher(
                batch["global_crops"][:, 0], padding_mask=pmask
            )

        loss = torch.tensor(0.0, device=feats["cls"].device, requires_grad=False)

        if batch.get("seg_masks") is not None:
            pred = seg_head(feats["patch_tokens"])
            loss = loss + F.binary_cross_entropy_with_logits(pred, batch["seg_masks"])

        valid_cls = batch["cls_labels"] >= 0
        if valid_cls.any():
            pred_cls = cls_head(feats["cls"])
            loss = loss + F.cross_entropy(
                pred_cls[valid_cls], batch["cls_labels"][valid_cls]
            )

    if loss.requires_grad:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return {"loss_finetune": loss.item()}


# ── Build DataModule ───────────────────────────────────────────────────────────

def build_datamodule(cfg: dict, cscs: CSCSConfig) -> USFoundationDataModule:
    img_cfg_dict = dict(cfg["transforms"]["image"])
    vid_cfg_dict = dict(cfg["transforms"]["video"])

    # Strip nested freq_mask dict (handled inside dataclass)
    img_freq = img_cfg_dict.pop("freq_mask", {})
    vid_freq = vid_cfg_dict.pop("freq_mask", {})

    from data.pipeline.transforms import FreqMaskConfig
    img_cfg = ImageSSLTransformConfig(
        **img_cfg_dict,
        freq_mask=FreqMaskConfig(**img_freq) if img_freq else FreqMaskConfig(),
    )
    vid_cfg = VideoSSLTransformConfig(
        **vid_cfg_dict,
        freq_mask=FreqMaskConfig(**vid_freq) if vid_freq else FreqMaskConfig(),
    )

    train_manifest = cscs.manifest_path(Path(cfg["manifest"]["path"]).name)

    return USFoundationDataModule(
        manifest_path           = str(train_manifest),
        image_batch_size        = cfg["loaders"]["image_batch_size"],
        video_batch_size        = cfg["loaders"]["video_batch_size"],
        num_workers             = cfg["loaders"]["num_workers"],
        pin_memory              = cfg["loaders"]["pin_memory"],
        patch_size              = cfg["transforms"]["patch_size"],
        total_training_steps    = cfg["curriculum"]["total_training_steps"],
        image_samples_per_epoch = cfg["curriculum"]["image_samples_per_epoch"],
        video_samples_per_epoch = cfg["curriculum"]["video_samples_per_epoch"],
        anatomy_weights         = cfg.get("anatomy_weights", {}),
        root_remap              = cscs.remap_dict(),
        image_cfg               = img_cfg,
        video_cfg               = vid_cfg,
    )


# Downstream heads are imported from models/heads/ via build_seg_head / build_cls_head.


# ── Logging ────────────────────────────────────────────────────────────────────

class MetricLogger:
    def __init__(self, log_dir: Path, rank: int = 0):
        self.log_dir = log_dir
        self.rank    = rank
        self._buffer = []
        if rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(log_dir / "metrics.jsonl", "a")
        else:
            self._fh = None

    def log(self, step: int, phase: int, metrics: dict):
        row = {"step": step, "phase": phase, **metrics, "ts": time.time()}
        self._buffer.append(row)
        if self._fh is not None:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()
        if is_main() and step % 50 == 0:
            parts = [f"{k}={v:.4f}" for k, v in metrics.items()
                     if isinstance(v, float)]
            log.info(f"[Step {step:7d} P{phase}] " + "  ".join(parts))

    def close(self):
        if self._fh:
            self._fh.close()


def main():
    import sys as _sys
    import subprocess as _subprocess
    _scripts_train = str(Path(__file__).parent.parent / "scripts" / "train.py")
    raise SystemExit(
        _subprocess.call([_sys.executable, _scripts_train] + _sys.argv[1:])
    )


if __name__ == "__main__":
    main()  # redirects to scripts/train.py
