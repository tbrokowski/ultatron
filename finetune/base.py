"""
finetune/base.py  ·  FinetuneExperiment base class
========================================================

Every dataset-specific finetune script inherits from FinetuneExperiment.

Design rationale
----------------
The generic heads in models/heads/ (LinearSegHead, DPTSegHead, etc.) are
task-type abstractions

What is dataset-specific:
  - Which head class to instantiate, and with what parameters
  - Which loss function(s) and their weights
  - The dataloader (dataset-specific file loading, augmentation, splits)
  - Evaluation metrics and their benchmark runner
  - Which visualisations to produce on completion
  - YAML config keys (LR, epochs, batch size, head_type)

What is shared (lives here in base.py):
  - The backbone-frozen training loop
  - Early stopping logic
  - Checkpoint save/restore for the head(s) only
  - Logging and results dict construction
  - Integration with Trainer.phase4()

Calling convention
------------------
Each dataset finetune is a standalone runnable module:

    python -m finetune.camus \\
        --checkpoint /path/to/phase3_end.pt \\
        --data-root  /capstor/.../CAMUS \\
        --config     configs/finetune/camus.yaml \\
        --output-dir results/camus/

Or from the trainer (Phase 4 integration):

    experiment = CAMUSFinetune.from_yaml(cfg_path)
    experiment.setup(img_branch, device)
    experiment.run()
    results = experiment.evaluate()

The Trainer.phase4() iterates over a list of FinetuneExperiment
objects rather than hardcoding seg_head + cls_head.

Head ownership
--------------
Each FinetuneExperiment owns its head(s).  The backbone (img_branch /
vid_branch) is passed in at setup() time, frozen, and never modified.
This means multiple FinetuneExperiments can share the same backbone
without interfering with each other.
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    """
    Shared finetune hyperparameters.  Dataset-specific params are
    loaded from the same YAML under a 'finetune:' key.
    """
    # Optimiser
    lr:           float = 1e-4
    weight_decay: float = 0.01
    max_epochs:   int   = 50
    batch_size:   int   = 16
    num_workers:  int   = 4

    # Early stopping
    patience:     int   = 10
    monitor:      str   = "val_loss"     # "val_loss" | "val_dice" | "val_auc"
    monitor_mode: str   = "min"          # "min" | "max"

    # Backbone
    freeze_backbone: bool = True

    # Head type — interpreted by each subclass
    head_type:    str   = "linear"       # "linear" | "dpt" | "mlp" | "attentive_pool"

    # Output resolution (upsample head output to this before loss)
    output_size:  int   = 224

    # Logging
    log_every:    int   = 10
    checkpoint_best: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "FinetuneConfig":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    @classmethod
    def from_yaml(cls, path: str, key: str = "finetune") -> "FinetuneConfig":
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d.get(key, d))


class FinetuneExperiment(ABC):
    """
    Abstract base for all dataset-specific finetune experiments.

    Concrete subclasses implement:
      build_head(embed_dim, cfg) → nn.Module
      build_dataloader(split, cfg) → DataLoader
      compute_loss(batch, feats, head_output) → Tensor
      compute_val_metrics(dataloader) → dict
      run_viz(results, output_dir) → None

    The base class provides the training loop, early stopping,
    checkpointing, and integration with OuraTrainer.
    """

    # Set by subclass
    EXPERIMENT_NAME: str = "base"
    DATASET_ID:      str = "unknown"
    TASK:            str = "segmentation"
    BENCHMARK_CLS    = None  # Optional eval.benchmarks.* class

    def __init__(
        self,
        data_root:  str,
        output_dir: str,
        cfg:        FinetuneConfig,
    ):
        self.data_root  = Path(data_root)
        self.output_dir = Path(output_dir)
        self.cfg        = cfg

        # Set by setup()
        self.img_branch  = None
        self.vid_branch  = None
        self.head:  Optional[nn.Module] = None
        self.head2: Optional[nn.Module] = None   # optional second head (e.g. cls + seg)
        self.device = "cuda"

        # Training state
        self._best_metric  = float("inf") if cfg.monitor_mode == "min" else -float("inf")
        self._patience_ctr = 0
        self._train_log: list[dict] = []

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        """
        Instantiate the task head using the generic classes from models/heads/.
        E.g.:  return build_seg_head(embed_dim, n_classes=1, head_type=cfg.head_type)
        """
        ...

    @abstractmethod
    def build_dataloader(self, split: str) -> DataLoader:
        """Return a DataLoader for 'train', 'val', or 'test'."""
        ...

    @abstractmethod
    def compute_loss(
        self,
        batch:       dict,
        feats:       dict,      # {cls, patch_tokens} from backbone
        head_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar loss tensor from batch targets and head output."""
        ...

    @abstractmethod
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        """
        Run the full validation set and return a metrics dict.
        Must include the key self.cfg.monitor.
        E.g.: {"val_loss": 0.23, "val_dice": 0.81}
        """
        ...

    def run_viz(self, results: dict, output_dir: Path) -> None:
        """Optional: produce viz figures after training completes."""
        pass

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        img_branch,
        device:     str = "cuda",
        vid_branch  = None,
    ):
        """Wire in the backbone and build the head."""
        self.img_branch = img_branch
        self.vid_branch = vid_branch
        self.device     = device
        embed_dim = img_branch.embed_dim

        if self.cfg.freeze_backbone:
            for p in img_branch.parameters():
                p.requires_grad_(False)
            img_branch.eval()

        # Match the head dtype to the backbone so features and weights are
        # compatible without needing autocast in every forward pass.
        backbone_dtype = next(img_branch.parameters()).dtype
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[{self.EXPERIMENT_NAME}] Head: {self.head} (dtype={backbone_dtype})")

    # ── Training loop ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Full finetune training loop with early stopping.
        Returns the training log (list of per-epoch dicts).
        """
        assert self.head is not None, "Call setup() before run()"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_loader = self.build_dataloader("train")
        val_loader   = self.build_dataloader("val")

        optimiser = torch.optim.AdamW(
            self.head.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        # BF16 does not need loss scaling; GradScaler unscale is not implemented for BF16 on some CUDA builds.
        use_amp_scaler = next(self.head.parameters()).dtype != torch.bfloat16
        scaler = GradScaler(enabled=torch.cuda.is_available() and use_amp_scaler)

        log.info(f"[{self.EXPERIMENT_NAME}] Training for up to "
                 f"{self.cfg.max_epochs} epochs on {self.DATASET_ID}")

        for epoch in range(self.cfg.max_epochs):
            epoch_loss = self._train_epoch(train_loader, optimiser, scaler)
            val_metrics = self.compute_val_metrics(val_loader)
            val_metrics["epoch"]      = epoch
            val_metrics["train_loss"] = epoch_loss
            self._train_log.append(val_metrics)

            if epoch % self.cfg.log_every == 0:
                log.info(f"  Epoch {epoch:3d}  train_loss={epoch_loss:.4f}  "
                         + "  ".join(f"{k}={v:.4f}"
                                     for k, v in val_metrics.items()
                                     if isinstance(v, float) and k != "train_loss"))

            # Early stopping + best checkpoint
            monitor_val = val_metrics.get(self.cfg.monitor, epoch_loss)
            improved = (
                monitor_val < self._best_metric
                if self.cfg.monitor_mode == "min"
                else monitor_val > self._best_metric
            )
            if improved:
                self._best_metric = monitor_val
                self._patience_ctr = 0
                if self.cfg.checkpoint_best:
                    self._save_head("best_head.pt")
            else:
                self._patience_ctr += 1
                if self._patience_ctr >= self.cfg.patience:
                    log.info(f"  Early stopping at epoch {epoch} "
                             f"(patience={self.cfg.patience})")
                    break

        # Reload best head (uses overridable load_head so subclasses can restore
        # multiple heads from the same checkpoint file)
        best_path = self.output_dir / "best_head.pt"
        if best_path.exists():
            self.load_head(str(best_path))

        self._save_log()
        return {"train_log": self._train_log, "best_metric": self._best_metric}

    def _backward_step_with_scaler(
        self,
        loss: torch.Tensor,
        optimiser: torch.optim.Optimizer,
        scaler: GradScaler,
        clip_params,
    ) -> None:
        """Backward, grad clip, optimizer step. Scaler is off when training heads in BF16."""
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimiser.step()

    def _train_epoch(
        self,
        loader:    DataLoader,
        optimiser: torch.optim.Optimizer,
        scaler:    GradScaler,
    ) -> float:
        self.head.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in loader:
            batch = {
                k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    feats = self.img_branch.forward_teacher(
                        batch["image"], padding_mask=batch.get("padding_mask")
                    )
                head_out = self.head(
                    feats["patch_tokens"],
                    padding_mask=batch.get("padding_mask"),
                ) if self._head_takes_patch_tokens() else self.head(feats["cls"])

                loss = self.compute_loss(batch, feats, head_out)

            self._backward_step_with_scaler(
                loss, optimiser, scaler, self.head.parameters()
            )
            optimiser.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _head_takes_patch_tokens(self) -> bool:
        """
        Returns True if head.forward() takes patch_tokens (segmentation heads),
        False if it takes cls token (classification/regression heads).
        Inferred from head class name.
        """
        name = type(self.head).__name__.lower()
        return "seg" in name or "attentive" in name or "concept" in name

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, split: str = "test") -> dict:
        """
        Run evaluation on the given split.

        If the subclass defines BENCHMARK_CLS, we delegate to the central
        eval.benchmarks runner for that dataset (authoritative metrics).
        Otherwise we fall back to compute_val_metrics() on this experiment.
        """
        assert self.head is not None, "Call setup() and run() first"
        self.head.eval()

        benchmark_cls = getattr(self, "BENCHMARK_CLS", None)
        if benchmark_cls is not None:
            benchmark = benchmark_cls(
                img_branch=self.img_branch,
                head=self.head,
                device=self.device,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
            )
            results = benchmark.run(str(self.data_root), split=split)
        else:
            val_loader = self.build_dataloader(split)
            results = self.compute_val_metrics(val_loader)
            results["split"] = split

        results["experiment"] = self.EXPERIMENT_NAME
        self._save_results(results)
        return results

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_head(self, name: str = "best_head.pt"):
        path = self.output_dir / name
        torch.save(self.head.state_dict(), path)

    def load_head(self, path: str):
        self.head.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        log.info(f"[{self.EXPERIMENT_NAME}] Head loaded from {path}")

    def _save_log(self):
        path = self.output_dir / "train_log.json"
        path.write_text(json.dumps(self._train_log, indent=2))

    def _save_results(self, results: dict):
        path = self.output_dir / "results.json"
        path.write_text(json.dumps(results, indent=2))
        log.info(f"[{self.EXPERIMENT_NAME}] Results → {path}")

    # ── CLI entry point ───────────────────────────────────────────────────────

    @classmethod
    def main(cls):
        """
        Standard CLI for any finetune experiment.
        Subclasses call: if __name__ == "__main__": MyFinetune.main()
        """
        import argparse
        parser = argparse.ArgumentParser(
            description=f"Oura finetune: {cls.EXPERIMENT_NAME}"
        )
        parser.add_argument("--checkpoint", required=True,
                            help="Path to SSL pre-training checkpoint (phase3_end.pt)")
        parser.add_argument("--data-root",  required=True,
                            help="Dataset root directory")
        parser.add_argument("--config",     required=True,
                            help="Path to finetune YAML config")
        parser.add_argument("--output-dir", default="results/finetune",
                            help="Where to write results and checkpoints")
        parser.add_argument("--device",     default="cuda")
        parser.add_argument("--eval-only",  action="store_true",
                            help="Skip training, load best_head.pt and evaluate only")
        args = parser.parse_args()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s"
        )

        import yaml
        with open(args.config) as f:
            raw = yaml.safe_load(f)

        cfg        = FinetuneConfig.from_dict(raw.get("finetune", raw))
        experiment = cls(
            data_root  = args.data_root,
            output_dir = args.output_dir,
            cfg        = cfg,
        )

        # Load backbone
        from models import ModelConfig, build_image_branch
        model_cfg = ModelConfig.from_dict(raw.get("model", {}))
        model_cfg.frozen_teacher = None   # finetune uses student only
        img_branch = build_image_branch(model_cfg, device=args.device)

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        img_branch.teacher.load_state_dict(ckpt["img_teacher"])
        log.info(f"Backbone loaded from {args.checkpoint}")

        experiment.setup(img_branch, device=args.device)

        if args.eval_only:
            best = Path(args.output_dir) / "best_head.pt"
            if best.exists():
                experiment.load_head(str(best))
        else:
            experiment.run()

        results = experiment.evaluate("test")
        experiment.run_viz(results, Path(args.output_dir))

        print(json.dumps(results, indent=2))
