"""
vlm/grpo/trainer.py  ·  GRPOTrainer
=====================================

Implements Group Relative Policy Optimization (GRPO) for the Ultatron VLM
student.

GRPO objective (DeepSeekMath / DeepEyes)
-----------------------------------------
  For each prompt i with G rollouts and advantages A_{i,g}:

    L_GRPO = -E_{(i,g)} [
        min(r_{i,g,t} * A_{i,g},  clip(r_{i,g,t}, 1-ε, 1+ε) * A_{i,g})
    ]

  where r_{i,g,t} = π_θ(a_t | s_t) / π_ref(a_t | s_t)   (probability ratio)
  and the expectation is over model-generated tokens only (loss_mask = 1).

KL regularization (optional, follows DeepEyes: β=0 initially)
  L_total = L_GRPO + β * KL(π_θ || π_ref)

Training parameters
-------------------
  Optimizer : AdamW with cosine LR schedule
  Precision : bfloat16 with torch.autocast
  Updates   : LoRA parameters + UltatronProjector only
  Saving    : StudentModel.save() every checkpoint_every steps
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Hyperparameters for GRPO training."""
    # Core GRPO
    group_size:        int   = 8       # G — rollouts per prompt
    clip_eps:          float = 0.2     # PPO clip ratio ε
    kl_coeff:          float = 0.0     # β — 0 = no KL (DeepEyes default)
    kl_warmup_steps:   int   = 5000    # steps to linearly increase β to kl_coeff_target
    kl_coeff_target:   float = 0.001   # final β value after warmup

    # Optimisation
    lr:                float = 1e-5
    lr_min:            float = 1e-6
    weight_decay:      float = 0.01
    grad_clip:         float = 1.0
    warmup_steps:      int   = 200
    total_steps:       int   = 18000   # stage1 3k + stage2 10k + stage3 5k

    # Training stages
    stage1_steps:      int   = 3000    # rule-based rewards, no SAM2
    stage2_steps:      int   = 10000   # SAM2 tool enabled
    # stage3 = remaining steps (Med-Gemini judge)

    # Logging / checkpointing
    log_every:         int   = 10
    checkpoint_every:  int   = 500
    ckpt_dir:          str   = "/tmp/vlm_grpo_ckpt"

    # Memory
    gradient_accumulation_steps: int  = 4
    max_seq_len:       int   = 8192

    @classmethod
    def from_dict(cls, d: dict) -> "GRPOConfig":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    @property
    def stage3_start(self) -> int:
        return self.stage1_steps + self.stage2_steps


class GRPOTrainer:
    """
    GRPO trainer for StudentModel.

    Parameters
    ----------
    student       : StudentModel (LoRA Qwen + UltatronProjector)
    rollout       : AgenticRollout
    reward_fn     : CompositeReward
    cfg           : GRPOConfig
    rank          : DDP rank (0 for logging)
    """

    def __init__(
        self,
        student:   Any,          # StudentModel
        rollout:   Any,          # AgenticRollout
        reward_fn: Any,          # CompositeReward
        cfg:       GRPOConfig,
        rank:      int = 0,
    ):
        self.student   = student
        self.rollout   = rollout
        self.reward_fn = reward_fn
        self.cfg       = cfg
        self.rank      = rank

        self.step      = 0
        self._grad_acc_step = 0

        self.optimizer, self.scheduler = self._build_optimizer()
        self._ckpt_dir = Path(cfg.ckpt_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._metrics: List[Dict[str, Any]] = []

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def _build_optimizer(self):
        """AdamW on LoRA + projector params only."""
        params = list(self.student.trainable_parameters())
        param_list = [{"params": [p for _, p in params], "lr": self.cfg.lr}]
        opt = torch.optim.AdamW(param_list, weight_decay=self.cfg.weight_decay)

        # Cosine schedule with linear warmup
        def lr_lambda(step: int) -> float:
            if step < self.cfg.warmup_steps:
                return step / max(self.cfg.warmup_steps, 1)
            progress = (step - self.cfg.warmup_steps) / max(
                self.cfg.total_steps - self.cfg.warmup_steps, 1
            )
            cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_frac = self.cfg.lr_min / self.cfg.lr
            return min_frac + (1.0 - min_frac) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return opt, scheduler

    # ── KL coefficient schedule ───────────────────────────────────────────────

    def _kl_coeff(self) -> float:
        """Linearly anneal β from 0 to kl_coeff_target over kl_warmup_steps."""
        if self.cfg.kl_coeff == 0.0 or self.cfg.kl_warmup_steps == 0:
            return 0.0
        frac = min(1.0, self.step / self.cfg.kl_warmup_steps)
        return frac * self.cfg.kl_coeff_target

    # ── Current training stage ────────────────────────────────────────────────

    @property
    def stage(self) -> int:
        """1 = rule-based, 2 = SAM2, 3 = Med-Gemini judge."""
        if self.step < self.cfg.stage1_steps:
            return 1
        if self.step < self.cfg.stage3_start:
            return 2
        return 3

    # ── Main training step ────────────────────────────────────────────────────

    def train_step(
        self,
        samples:           List[Any],           # List[VLMSample]
        ultatron_backbone: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        One GRPO update step (may span multiple gradient accumulation micro-steps).

        For each sample: rollout G trajectories → compute rewards → compute
        GRPO loss → accumulate gradients → optimizer step.

        Returns a dict of scalar metrics for logging.
        """
        device = self.cfg.ckpt_dir and next(self.student._qwen_model.parameters()).device

        total_loss   = 0.0
        total_reward = 0.0
        total_r_acc  = 0.0
        total_r_fmt  = 0.0
        total_r_tool = 0.0
        n_batches    = 0

        for sample in samples:
            # ── Rollout ───────────────────────────────────────────────────────
            rollout_batch = self.rollout.rollout(sample, ultatron_backbone)

            tokenizer = self.student._processor.tokenizer
            batch_tensors = self.rollout.prepare_for_training(
                rollout_batch, tokenizer, device=str(device)
            )

            # ── GRPO loss ─────────────────────────────────────────────────────
            loss = self._grpo_loss(rollout_batch, batch_tensors, sample)
            loss = loss / self.cfg.gradient_accumulation_steps

            loss.backward()
            self._grad_acc_step += 1

            total_loss   += loss.item() * self.cfg.gradient_accumulation_steps
            total_reward += float(sum(rollout_batch.rewards)) / len(rollout_batch.rewards)

            # Aggregate reward components
            for traj in rollout_batch.trajectories:
                info = traj.reward_info
                total_r_acc  += info.get("r_acc",  0.0)
                total_r_fmt  += info.get("r_fmt",  0.0)
                total_r_tool += info.get("r_tool", 0.0)
            n_batches += 1

        # ── Gradient step ─────────────────────────────────────────────────────
        if self._grad_acc_step >= self.cfg.gradient_accumulation_steps:
            nn.utils.clip_grad_norm_(
                [p for _, p in self.student.trainable_parameters()],
                self.cfg.grad_clip,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self._grad_acc_step = 0
            self.step += 1

        n = max(n_batches, 1)
        metrics = {
            "loss":      total_loss   / n,
            "reward":    total_reward / n,
            "r_acc":     total_r_acc  / (n * self.cfg.group_size),
            "r_fmt":     total_r_fmt  / (n * self.cfg.group_size),
            "r_tool":    total_r_tool / (n * self.cfg.group_size),
            "lr":        self.scheduler.get_last_lr()[0],
            "kl_coeff":  self._kl_coeff(),
            "stage":     self.stage,
            "step":      self.step,
        }
        self._metrics.append(metrics)
        return metrics

    # ── GRPO loss ─────────────────────────────────────────────────────────────

    def _grpo_loss(
        self,
        rollout_batch: Any,      # RolloutBatch
        batch_tensors: Dict[str, torch.Tensor],
        sample:        Any,      # VLMSample
    ) -> torch.Tensor:
        """
        Compute the clipped GRPO surrogate loss + optional KL.

        Per DeepEyes / DeepSeekMath:
          L = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        where expectation is over model-generated tokens (loss_mask=1).
        """
        input_ids    = batch_tensors["input_ids"]       # (G, L)
        loss_masks   = batch_tensors["loss_masks"]      # (G, L)
        advantages   = batch_tensors["advantages"]      # (G,)
        attention_mk = batch_tensors["attention_mask"]  # (G, L)

        G, L = input_ids.shape

        # ── Current policy log-probs ──────────────────────────────────────────
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Get patch/tube tokens for this sample (may be None for pure-text)
            patch_t = getattr(sample, "patch_tokens", None)
            tube_t  = getattr(sample, "tube_tokens",  None)

            fwd_out = self.student.forward(
                input_ids      = input_ids,
                attention_mask = attention_mk,
                pixel_values   = None,    # Qwen ViT tokens already in input_ids
                image_grid_thw = None,
                patch_tokens   = patch_t.expand(G, -1, -1) if patch_t is not None else None,
                tube_tokens    = tube_t.expand(G,  -1, -1) if tube_t  is not None else None,
            )
            log_probs = fwd_out["log_probs"]  # (G, L_full, V)

        # Gather log-prob of the chosen tokens (shifted by 1 for causal LM)
        # log_probs shape: (G, L_full, V); target shape: (G, L-1)
        L_eff     = min(L - 1, log_probs.shape[1] - 1)
        target    = input_ids[:, 1:L_eff + 1]                       # (G, L_eff)
        curr_lp   = log_probs[:, :L_eff, :]                         # (G, L_eff, V)
        curr_lp   = curr_lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (G, L_eff)
        lm_eff    = loss_masks[:, 1:L_eff + 1]                       # (G, L_eff)

        # ── Reference policy log-probs ────────────────────────────────────────
        beta = self._kl_coeff()
        if beta > 0 and self.student.ref_model is not None:
            with torch.no_grad():
                ref_lp_full = self.student.ref_log_probs(
                    input_ids=input_ids, attention_mask=attention_mk
                )  # (G, L_full, V)
                ref_lp = ref_lp_full[:, :L_eff, :].gather(
                    -1, target.unsqueeze(-1)
                ).squeeze(-1)  # (G, L_eff)
        else:
            ref_lp = curr_lp.detach()  # ratio = 1, KL = 0

        # ── Probability ratio ─────────────────────────────────────────────────
        ratio = torch.exp(curr_lp - ref_lp.detach())    # (G, L_eff)

        # Expand advantages to token level: (G,) → (G, L_eff)
        adv = advantages.unsqueeze(1).expand_as(ratio)  # (G, L_eff)

        # ── Clipped surrogate ─────────────────────────────────────────────────
        clip_eps   = self.cfg.clip_eps
        pg_loss    = -torch.min(
            ratio * adv,
            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv,
        )  # (G, L_eff)

        # Mask: only over model-generated tokens
        pg_loss    = (pg_loss * lm_eff).sum() / (lm_eff.sum() + 1e-6)

        # ── KL penalty ────────────────────────────────────────────────────────
        if beta > 0:
            kl = (torch.exp(ref_lp.detach()) * (ref_lp.detach() - curr_lp))
            kl = (kl * lm_eff).sum() / (lm_eff.sum() + 1e-6)
        else:
            kl = torch.tensor(0.0, device=pg_loss.device)

        return pg_loss + beta * kl

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, tag: Optional[str] = None):
        """Save StudentModel (LoRA + projector) and trainer state."""
        step_str = tag or f"step_{self.step:07d}"
        out_dir  = self._ckpt_dir / step_str
        self.student.save(str(out_dir))
        # Save trainer state (step, metrics history)
        state = {"step": self.step, "stage": self.stage}
        with open(out_dir / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        # Always update "latest"
        latest = self._ckpt_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink(missing_ok=True)
        latest.symlink_to(out_dir.name)
        if self.rank == 0:
            log.info(f"Checkpoint saved: {out_dir}")

    def load_checkpoint(self, ckpt_dir: str):
        """Resume from a saved checkpoint directory."""
        state_path = Path(ckpt_dir) / "trainer_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.step = state.get("step", 0)
            log.info(f"Resumed from step {self.step}")

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        data_module:       Any,             # VLMDataModule
        ultatron_backbone: Optional[Any] = None,
    ):
        """
        Full training loop.  Iterates over the VLMDataModule for total_steps.
        Handles stage transitions (enabling SAM2 tool in stage 2, Med-Gemini
        in stage 3).
        """
        loader = data_module.loader()
        data_iter = iter(loader)

        t0 = time.time()
        while self.step < self.cfg.total_steps:
            # Stage transitions
            self._maybe_update_stage()

            # Fetch a micro-batch of samples
            try:
                samples = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                samples   = next(data_iter)

            metrics = self.train_step(samples, ultatron_backbone)

            # Logging
            if self.step % self.cfg.log_every == 0 and self.rank == 0:
                elapsed = time.time() - t0
                log.info(
                    f"[step {self.step:6d} | stage {self.stage}] "
                    f"loss={metrics['loss']:.4f}  reward={metrics['reward']:.3f}  "
                    f"r_acc={metrics['r_acc']:.3f}  r_tool={metrics['r_tool']:.3f}  "
                    f"lr={metrics['lr']:.2e}  elapsed={elapsed:.0f}s"
                )

            # Checkpointing
            if self.step % self.cfg.checkpoint_every == 0 and self.rank == 0:
                self.save_checkpoint()

        if self.rank == 0:
            self.save_checkpoint(tag="final")
            log.info(f"Training complete at step {self.step}.")

    def _maybe_update_stage(self):
        """Enable/disable tools and reward components based on training stage."""
        prev_stage = getattr(self, "_prev_stage", 0)
        if self.stage != prev_stage:
            if self.stage == 2:
                log.info(f"[step {self.step}] Stage 2: SAM2 tool ENABLED.")
                # student.tool_registry already has sam2; just ensure it's set
            if self.stage == 3:
                log.info(f"[step {self.step}] Stage 3: Med-Gemini judge ACTIVE.")
                # reward_fn.acc_reward mode may be updated externally via config
            self._prev_stage = self.stage
