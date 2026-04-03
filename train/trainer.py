"""
train/trainer.py  · Trainer — production training orchestrator
========================================================================

Wraps phase_steps.py functions with all the production concerns:
  - optimizer.step() and scaler management
  - EMA teacher updates
  - checkpoint save/resume
  - curriculum updates (resolution, mask ratio, ALP alpha)
  - metric logging
  - DDP synchronisation
  - SIGUSR1 pre-emption checkpoint (for SLURM)


Usage
-----
    cfg = TrainConfig.from_yaml("configs/train/train_config.yaml")
    trainer = Trainer(cfg, img_branch, vid_branch, cross, proto, heads, dm)
    trainer.train()

Or to start from a checkpoint:
    trainer.load_checkpoint(path)
    trainer.train()
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from train.phase_steps import (
    phase1_step, phase2_step, phase3_step, phase4_step,
)
from train.gram import GramTeacher

log = logging.getLogger(__name__)


# ── Training config dataclass ─────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """
    All training hyperparameters.  Loaded from configs/train/train_config.yaml.

    Loss weights
    ------------
    lam1 – lam7, lam_7b, lam_gram:  see phase_steps.py docstring.

    Phase boundaries
    ----------------
    Fractions of total_steps:
        phase1_frac  [0,   phase1_end)   image warm-start
        phase2_frac  [p1,  phase2_end)   video warm-start
        phase3_frac  [p2,  phase3_end)   hybrid joint
        phase4_frac  [p3,  total_steps)  downstream heads

    Resolution curriculum
    ---------------------
    Controlled by step boundaries: res_step_1, res_step_2, res_step_3
    mapping to max_global_crop_px values res_px_1, res_px_2, res_px_3.
    """
    # Phase fractions of total_steps
    phase1_frac: float = 0.10
    phase2_frac: float = 0.20
    phase3_frac: float = 0.90
    # phase4 is 1.0

    # Optimiser
    base_lr:        float = 1e-4
    weight_decay:   float = 0.04
    beta1:          float = 0.90
    beta2:          float = 0.95
    grad_clip:      float = 1.0
    warmup_steps_p1: int  = 2_000
    warmup_steps_p2: int  = 1_000
    warmup_steps_p3: int  = 500

    # EMA
    ema_momentum: float = 0.9995

    # Loss weights
    lam1:     float = 1.0   # DINO CLS
    lam2:     float = 1.0   # iBOT patch
    lam3:     float = 0.5   # local crop
    lam4:     float = 1.0   # video CLS
    lam5:     float = 1.0   # tube prediction
    lam6:     float = 1.0   # cross-branch
    lam7:     float = 0.5   # prototype
    lam_7b:   float = 0.5   # 7B teacher distillation
    lam_gram: float = 1.0   # Gram anchoring
    lam_koleo:float = 0.1   # KoLeo uniformity

    # Gram anchoring
    gram_start_step:      int = 100_000
    gram_refresh_interval: int = 50_000

    # Image resolution curriculum (step boundaries + max_global_crop_px values)
    res_step_1: int = 30_000
    res_step_2: int = 150_000
    res_px_1:   int = 512
    res_px_2:   int = 672
    res_px_3:   int = 896

    # Video resolution curriculum (max_crop_px for VideoSSLTransformConfig).
    # Defaults mirror the image curriculum but at conservatively lower values
    # since 8-frame clips scale memory quadratically in spatial resolution.
    res_vid_step_1: int = 30_000
    res_vid_step_2: int = 150_000
    res_vid_px_1:   int = 224
    res_vid_px_2:   int = 256
    res_vid_px_3:   int = 336

    # Checkpointing
    checkpoint_every: int = 5_000
    log_every:        int = 50

    # Phase 4 optimiser (different from SSL phases)
    phase4_lr:          float = 1e-4
    phase4_weight_decay:float = 0.01

    # KoLeo
    use_koleo: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        filtered = {k: v for k, v in d.items() if hasattr(cls, k)}
        # Coerce numeric fields in case YAML loaded them as str (e.g. 1e-4)
        for key in ("base_lr", "phase4_lr", "weight_decay", "phase4_weight_decay",
                    "beta1", "beta2", "grad_clip", "ema_momentum",
                    "lam1", "lam2", "lam3", "lam4", "lam5", "lam6", "lam7",
                    "lam_7b", "lam_gram", "lam_koleo",
                    "phase1_frac", "phase2_frac", "phase3_frac"):
            if key in filtered and isinstance(filtered[key], str):
                filtered[key] = float(filtered[key])
        for key in ("warmup_steps_p1", "warmup_steps_p2", "warmup_steps_p3",
                    "gram_start_step", "gram_refresh_interval",
                    "res_step_1", "res_step_2", "res_px_1", "res_px_2", "res_px_3",
                    "res_vid_step_1", "res_vid_step_2",
                    "res_vid_px_1", "res_vid_px_2", "res_vid_px_3",
                    "checkpoint_every", "log_every"):
            if key in filtered and isinstance(filtered[key], str):
                filtered[key] = int(filtered[key])
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d.get("train", d))

    def lam_dict(self, no_7b: bool = False) -> dict:
        return {
            "lam1": self.lam1, "lam2": self.lam2, "lam3": self.lam3,
            "lam4": self.lam4, "lam5": self.lam5,
            "lam6": self.lam6, "lam7": self.lam7,
            "lam_7b":    0.0 if no_7b else self.lam_7b,
            "lam_gram":  self.lam_gram,
            "lam_koleo": self.lam_koleo,
        }


# ── Metric logger ─────────────────────────────────────────────────────────────

class MetricLogger:
    def __init__(self, log_dir: Path, rank: int = 0):
        self.log_dir = log_dir
        self.rank    = rank
        self._fh     = None
        if rank == 0:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(log_dir / "metrics.jsonl", "a")

    def log(self, step: int, phase: int, metrics: dict, log_every: int = 50):
        row = {"step": step, "phase": phase,
               **{k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)},
               "ts": time.time()}
        if self._fh:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()
        if self.rank == 0 and step % log_every == 0:
            parts = []
            # Always include integer debug fields when present (useful for Phase 3).
            for k in ("stage", "n_align_pairs"):
                if isinstance(row.get(k), int):
                    parts.append(f"{k}={row[k]}")
            # Then include float metrics.
            parts.extend(
                f"{k}={v:.4f}" for k, v in row.items()
                if isinstance(v, float) and k not in ("ts",)
            )
            log.info(f"[step {step:7d} P{phase}] " + "  ".join(parts))

    def close(self):
        if self._fh:
            self._fh.close()


# ── LR schedule ───────────────────────────────────────────────────────────────

def _get_lr(step: int, end_step: int, base_lr: float,
            warmup: int, min_factor: float = 1.0) -> float:
    """Linear warmup; flat thereafter (DINOv3 style with min_factor=1.0)."""
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if min_factor >= 1.0:
        return base_lr
    import math
    prog = (step - warmup) / max(end_step - warmup, 1)
    return min_factor * base_lr + (1 - min_factor) * base_lr * 0.5 * (
        1 + math.cos(math.pi * prog)
    )


# ── Trainer ───────────────────────────────────────────────────────────────

class Trainer:
    """
    Orchestrates the 4-phase training loop.

    ----------------
    - Phase transitions (step boundaries)
    - optimizer.step() + GradScaler + grad clip
    - EMA teacher updates
    - Curriculum updates (resolution, mask ratio, ALP alpha)
    - Checkpoint save/resume
    - Metric logging
    - SIGUSR1 handler for SLURM pre-emption

    Does NOT contain any forward pass logic — that lives in phase_steps.py.
    Does NOT contain any data loading — that lives in datamodule.py.

    Parameters
    ----------
    cfg          : TrainConfig
    img_branch   : ImageBranch (student + teacher)
    vid_branch          : VideoBranch (student + teacher)
    cross_distill       : CrossBranchDistillation
    proto_head          : PrototypeHead
    finetune_experiments: list of FinetuneExperiment for Phase 4.
                          Each experiment owns its head(s) and dataloader.
                          Pass [] to skip Phase 4 entirely.
                          Example:
                            from finetune import CAMUSFinetune, FinetuneConfig
                            cfg_ft = FinetuneConfig.from_yaml("configs/finetune/camus.yaml")
                            ft = CAMUSFinetune(data_root=..., output_dir=..., cfg=cfg_ft)
                            trainer = OuraTrainer(..., finetune_experiments=[ft])
    dm                  : USFoundationDataModule
    ckpt_dir            : directory for checkpoint files
    log_dir             : directory for metrics.jsonl
    rank                : DDP rank (0 = main process)
    total_steps         : total training steps from data config
    """

    def __init__(
        self,
        cfg:                   TrainConfig,
        img_branch,
        vid_branch,
        cross_distill,
        proto_head,
        dm,
        ckpt_dir:              Path,
        log_dir:               Path,
        finetune_experiments:  list = None,
        rank:                  int  = 0,
        local_rank:            int  = 0,
        total_steps:           int  = 300_000,
        no_7b:                 bool = False,
        use_amp_scaler:        bool = True,
    ):
        self.cfg                  = cfg
        self.use_amp_scaler       = use_amp_scaler
        self.img_branch           = img_branch
        self.vid_branch           = vid_branch
        self.cross_distill        = cross_distill
        self.proto_head           = proto_head
        self.finetune_experiments = finetune_experiments or []
        self.dm                   = dm
        self.ckpt_dir      = ckpt_dir
        self.log_dir       = log_dir
        self.rank          = rank
        self.local_rank    = local_rank
        self.total_steps   = total_steps
        self.no_7b         = no_7b
        self.device        = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

        # Phase boundaries
        self.phase1_end = int(cfg.phase1_frac * total_steps)
        self.phase2_end = int(cfg.phase2_frac * total_steps)
        self.phase3_end = int(cfg.phase3_frac * total_steps)

        # Loss weights dict
        self.lam = cfg.lam_dict(no_7b=no_7b)

        # Gram teacher
        self.gram_teacher = GramTeacher(
            img_branch.student,
            gram_start_step=cfg.gram_start_step,
            gram_refresh_interval=cfg.gram_refresh_interval,
        )

        # Optimiser — single AdamW for all trainable params
        self.optimizer = torch.optim.AdamW(
            [p for p in list(img_branch.parameters())
                      + list(vid_branch.parameters())
                      + list(cross_distill.parameters())
                      + list(proto_head.parameters())
                      if p.requires_grad],
            lr=cfg.base_lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )
        # BFloat16 does not need loss scaling; GradScaler unscale is not implemented for bfloat16 on some CUDA builds.
        self.scaler = GradScaler(enabled=use_amp_scaler and torch.cuda.is_available())

        # State
        self.global_step   = 0
        self.current_phase = 1
        self.best_val_loss = float("inf")

        # Logging
        self.logger = MetricLogger(log_dir, rank)

        # SIGUSR1 sentinel for SLURM pre-emption
        self._checkpoint_sentinel = ckpt_dir / ".checkpoint_now"
        signal.signal(signal.SIGUSR1, self._handle_sigusr1)

    # ── Signal handler ─────────────────────────────────────────────────────────

    def _handle_sigusr1(self, signum, frame):
        log.info(f"[rank {self.rank}] SIGUSR1 received — saving emergency checkpoint")
        if self.rank == 0:
            self._save_checkpoint("emergency.pt")

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str = "latest.pt"):
        path = self.ckpt_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)

        def _unwrap(m):
            return m.module if isinstance(m, DDP) else m

        state = {
            "global_step":    self.global_step,
            "current_phase":  self.current_phase,
            "best_val_loss":  self.best_val_loss,
            "img_student":    _unwrap(self.img_branch).student.state_dict(),
            "img_teacher":    _unwrap(self.img_branch).teacher.state_dict(),
            "vid_student":    _unwrap(self.vid_branch).student.state_dict(),
            "vid_teacher":    _unwrap(self.vid_branch).teacher.state_dict(),
            "cross_distill":  _unwrap(self.cross_distill).state_dict(),
            "proto_head":     _unwrap(self.proto_head).state_dict(),
            "optimizer":      self.optimizer.state_dict(),
            "scaler":         self.scaler.state_dict(),
        }
        if self.gram_teacher._snapshot is not None:
            state["gram_snapshot"]     = self.gram_teacher._snapshot.state_dict()
            state["gram_last_refresh"] = self.gram_teacher._last_refresh

        torch.save(state, path)
        log.info(f"[rank {self.rank}] Checkpoint → {path} (step {self.global_step})")

    def load_checkpoint(self, path: str | Path):
        state = torch.load(path, map_location="cpu")

        def _unwrap(m):
            return m.module if isinstance(m, DDP) else m

        _unwrap(self.img_branch).student.load_state_dict(state["img_student"])
        _unwrap(self.img_branch).teacher.load_state_dict(state["img_teacher"])
        _unwrap(self.vid_branch).student.load_state_dict(state["vid_student"])
        _unwrap(self.vid_branch).teacher.load_state_dict(state["vid_teacher"])
        # Allow non-breaking evolution of the cross-modal head (e.g. adding
        # predictor_vid) without invalidating older checkpoints.
        _unwrap(self.cross_distill).load_state_dict(state["cross_distill"], strict=False)
        _unwrap(self.proto_head).load_state_dict(state["proto_head"])
        # Optimizer/scaler state may become incompatible when param groups change
        # (e.g. adding new parameters). In that case, resume weights but reset opt.
        try:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scaler.load_state_dict(state["scaler"])
        except Exception as e:
            log.warning(f"Skipping optimizer/scaler state restore: {e}")

        self.global_step   = state["global_step"]
        self.current_phase = state["current_phase"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        log.info(f"Resumed from {path} — step {self.global_step}, phase {self.current_phase}")

    # ── Resolution curriculum ─────────────────────────────────────────────────

    def _update_resolution(self):
        s = self.global_step
        c = self.cfg

        # ── Image resolution ──────────────────────────────────────────────────
        img_px = c.res_px_1
        if s >= c.res_step_2:
            img_px = c.res_px_3
        elif s >= c.res_step_1:
            img_px = c.res_px_2
        img_cfg = self.dm.image_cfg
        if img_cfg.max_global_crop_px != img_px:
            img_cfg.max_global_crop_px = img_px
            if self.rank == 0:
                log.info(f"[step {s}] Image resolution → {img_px}px")

        # ── Video resolution ──────────────────────────────────────────────────
        vid_px = c.res_vid_px_1
        if s >= c.res_vid_step_2:
            vid_px = c.res_vid_px_3
        elif s >= c.res_vid_step_1:
            vid_px = c.res_vid_px_2
        vid_cfg = self.dm.video_cfg
        if vid_cfg.max_crop_px != vid_px:
            vid_cfg.max_crop_px = vid_px
            if self.rank == 0:
                log.info(f"[step {s}] Video resolution → {vid_px}px")

    # ── Per-step update helper ────────────────────────────────────────────────

    def _step_update(self, losses: dict, phase: int):
        """backward + clip + step + EMA + curriculum + log"""
        loss_tensor = losses["loss"]

        if self.scaler.is_enabled():
            self.scaler.scale(loss_tensor).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss_tensor.backward()

        all_params = (
            list(self.img_branch.parameters())
            + list(self.vid_branch.parameters())
            + list(self.cross_distill.parameters())
            + list(self.proto_head.parameters())
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in all_params if p.requires_grad and p.grad is not None],
            self.cfg.grad_clip,
        )

        if not torch.isfinite(grad_norm):
            # Non-finite gradients (NaN or Inf) — skip this optimizer step to
            # prevent corrupting model weights.  This can happen when a batch
            # contains pathological inputs (e.g. all-padded crops) before the
            # attention-bias construction is made numerically safe.
            if self.rank == 0:
                log.warning(
                    f"[step {self.global_step}] Non-finite grad norm "
                    f"({grad_norm:.4f}) — skipping optimizer step"
                )
            self.optimizer.zero_grad(set_to_none=True)
        else:
            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # EMA
        self.img_branch.update_teacher(self.cfg.ema_momentum)
        if phase >= 2:
            self.vid_branch.update_teacher(self.cfg.ema_momentum)

        self._update_resolution()
        self.dm.update_step(self.global_step)

        if self.rank == 0:
            self.logger.log(self.global_step, phase, losses, self.cfg.log_every)

            # Periodic + sentinel checkpoint
            sentinel = self._checkpoint_sentinel
            if (self.global_step % self.cfg.checkpoint_every == 0 and self.global_step > 0) \
               or sentinel.exists():
                self._save_checkpoint("latest.pt")
                if sentinel.exists():
                    sentinel.unlink()

        self.global_step += 1

    # ── Device helpers ────────────────────────────────────────────────────────

    def _to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    # ── Main training loop ────────────────────────────────────────────────────

    @staticmethod
    def _cycled(loader_fn):
        """
        Yields batches from loader_fn() in an infinite loop.
        loader_fn is called fresh each cycle so the DataLoader and its sampler
        re-shuffle indices, enabling true epoch-level data diversity.
        Used to prevent short-dataset streams from terminating a phase early.
        """
        while True:
            for batch in loader_fn():
                yield batch

    def train(self):
        # Apply resolution curriculum immediately so the very first batch is
        # loaded at the correct starting resolution (res_px_1 / res_vid_px_1).
        # Without this call, the DataLoader would use whatever value is in the
        # config YAML, which may not match the curriculum starting point.
        self._update_resolution()
        log.info(f"[rank {self.rank}] Starting training from step {self.global_step}")

        # ── Phase 1: Image warm-start ──────────────────────────────────────────
        if self.global_step < self.phase1_end and self.current_phase <= 1:
            self.current_phase = 1
            lr_fn = lambda s: _get_lr(s, self.phase1_end, self.cfg.base_lr,
                                       self.cfg.warmup_steps_p1)
            # Cycle image loader so small datasets (e.g. 780 BUSI images) don't
            # exhaust the iterator before reaching phase1_end.
            for batch in self._cycled(self.dm.image_loader):
                if self.global_step >= self.phase1_end:
                    break
                batch = self._to_device(batch)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr_fn(self.global_step)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = phase1_step(
                        batch, self.img_branch, self.gram_teacher,
                        self.lam, self.global_step, self.cfg.use_koleo,
                    )
                self._step_update(losses, phase=1)

            if self.rank == 0:
                self._save_checkpoint("phase1_end.pt")
                log.info("Phase 1 complete.")

        # ── Phase 2: Video warm-start ──────────────────────────────────────────
        if self.global_step < self.phase2_end and self.current_phase <= 2:
            self.current_phase = 2
            lr_fn = lambda s: _get_lr(s, self.phase2_end, self.cfg.base_lr,
                                       self.cfg.warmup_steps_p2)
            # Cycle video loader in case it exhausts before phase2_end.
            for batch in self._cycled(self.dm.video_loader):
                if self.global_step >= self.phase2_end:
                    break
                batch = self._to_device(batch)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr_fn(self.global_step)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = phase2_step(batch, self.vid_branch, self.lam)
                self._step_update(losses, phase=2)

            if self.rank == 0:
                self._save_checkpoint("phase2_end.pt")
                log.info("Phase 2 complete.")

        # ── Phase 3: Hybrid joint training ────────────────────────────────────
        if self.global_step < self.phase3_end and self.current_phase <= 3:
            self.current_phase = 3
            lr_fn = lambda s: _get_lr(s, self.phase3_end, self.cfg.base_lr,
                                       self.cfg.warmup_steps_p3)
            # Use the paired loader: each batch item is ONE clip that produces
            # both an image view AND a video view, guaranteeing n_align_pairs
            # == batch_size every step (no more hoping for study_id collisions
            # across independently-shuffled image and video loaders).
            for dual_raw in self._cycled(self.dm.paired_loader):
                if self.global_step >= self.phase3_end:
                    break
                dual = dual_raw.to(self.device)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr_fn(self.global_step)
                stage = self.dm.current_stage()

                _cd = self.cross_distill.module if isinstance(self.cross_distill, DDP) else self.cross_distill
                _ph = self.proto_head.module    if isinstance(self.proto_head,    DDP) else self.proto_head
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = phase3_step(
                        dual.image_batch, dual.video_batch,
                        self.img_branch, self.vid_branch,
                        _cd, _ph,
                        self.gram_teacher, self.lam,
                        self.global_step, stage,
                        alignment_pairs=dual.alignment_pairs,
                    )
                self._step_update(losses, phase=3)

            if self.rank == 0:
                self._save_checkpoint("phase3_end.pt")
                log.info("Phase 3 complete.")

        # ── Phase 4: Finetune experiments ──────────────────────────────────────
        if self.current_phase <= 4 and self.finetune_experiments:
            self.current_phase = 4
            log.info(f"Phase 4: running {len(self.finetune_experiments)} "
                     f"finetune experiment(s)")

            for experiment in self.finetune_experiments:
                if self.rank == 0:
                    log.info(f"  → {experiment.EXPERIMENT_NAME}")
                    # setup() freezes backbone and builds head
                    experiment.setup(
                        self.img_branch,
                        device    = self.device,
                        vid_branch= self.vid_branch,
                    )
                    # run() is the full finetune training loop with early stopping
                    experiment.run()
                    # evaluate() runs the test-set benchmark and saves results
                    results = experiment.evaluate("test")
                    # run_viz() produces figures
                    experiment.run_viz(results, experiment.output_dir)
                    log.info(f"  ✓ {experiment.EXPERIMENT_NAME}: {results}")

            if self.rank == 0:
                self._save_checkpoint("final.pt")
                log.info("Phase 4 complete. All finetune experiments finished.")

        elif self.current_phase <= 4 and not self.finetune_experiments:
            log.info("Phase 4: no finetune experiments configured — skipping.")

        self.logger.close()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
