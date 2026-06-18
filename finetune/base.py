"""
finetune/base.py  ·  FinetuneExperiment base class
========================================================

Every dataset-specific finetune script inherits from FinetuneExperiment.

Design rationale
----------------
The generic heads in models/heads/ (LinearSegHead, DPTSegHead, etc.) are
task-type abstractions; dataset-specific experiments provide the glue that
wires a head to a particular dataset, loss, and evaluation protocol.

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
    head_type:    str   = "dpt"        # "linear" | "dpt" | "upernet" (seg tasks)

    # Output resolution (upsample head output to this before loss)
    output_size:  int   = 224
    input_size:   int   = 224          # fixed resize for all backbones (incl. student)
    native_resolution: bool = False    # opt-in pretrain-style crops (requires input_size=0)
    native_max_px: int = 512           # native crop cap when native_resolution=True

    # Logging
    log_every:    int   = 10
    checkpoint_best: bool = True

    # Multi-task loss weights (BUSI seg + cls)
    seg_loss_weight: float = 1.0
    cls_loss_weight: float = 1.0
    seg_only_epochs: int = 0          # train segmentation before enabling cls loss
    bg_seg_loss_weight: float = 0.5   # weight for background-only seg on normal samples
    boundary_loss_weight: float = 0.5 # weight for boundary BCE term (0 = disable)
    refine_up: bool = False           # enable SubPixelRefinement (stride-4 → stride-2)
    seg_use_adapters: bool = True     # SegAdapter modules in seg heads
    seg_use_aspp: bool = True         # ASPP fusion in seg heads
    seg_use_attention_gates: bool = True  # FPN attention gates (UPerNet only)
    seg_only: bool = False            # BUSI: skip classification head entirely
    tumor_only_train: bool = False    # BUSI: train on benign+malignant only
    eval_tumor_only: bool = False     # BUSI: primary Dice on tumour images only
    split_manifest: str = ""          # BUSI/BUS-BRA: path to split_manifest.json
    cv_fold: int = 0                    # BUS-BRA/TN3K: OpenUS cross-validation fold index
    lv_target: str = "lv_structures"  # CAMUS: binary mask definition (see camus_io)
    camus_quality_filter: bool = False  # CAMUS: Good+Medium only (TMI Table III)
    camus_variant: str = "lv_epi"       # lv_endo | lv_epi | la | multiclass
    camus_training_mode: str = "binary"   # binary | multiclass

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
        checkpoint_dir: str | None = None,
    ):
        self.data_root  = Path(data_root)
        self.output_dir = Path(output_dir)
        self.cfg        = cfg
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            from finetune.paths import mirror_checkpoint_dir
            self.checkpoint_dir = mirror_checkpoint_dir(self.output_dir)

        # Set by setup()
        self.encoder     = None          # BackboneEncoder (preferred)
        self.img_branch  = None          # kept for legacy subclass access
        self.vid_branch  = None          # kept for legacy subclass access
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

    def run_viz(self, results: dict, output_dir: Path | None = None) -> None:
        """Produce segmentation GT vs pred figures for segmentation tasks."""
        if "segmentation" not in self.TASK:
            return
        from finetune.segmentation_viz import save_segmentation_figure
        save_segmentation_figure(self)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(
        self,
        img_branch  = None,
        device:     str = "cuda",
        vid_branch  = None,
        encoder     = None,
    ):
        """
        Wire in the backbone and build the task head.

        New-style call (comparison mode):
            experiment.setup(encoder=encoder, device=device)

        Legacy call (backward compat, used by scripts/finetune.py and trainer):
            experiment.setup(img_branch, device=device, vid_branch=vid_branch)
        """
        from finetune.backbones.ultatron_encoder import UltatronBranchEncoder

        if encoder is not None:
            self.encoder    = encoder
            # Expose as img_branch for subclasses that still reference it
            self.img_branch = encoder
            # BackboneEncoder uses encode_video(); only legacy Ultatron VideoBranch needs vid_branch
            self.vid_branch = None
        elif img_branch is not None:
            self.encoder    = UltatronBranchEncoder(img_branch, vid_branch)
            self.img_branch = img_branch
            self.vid_branch = vid_branch
        else:
            raise ValueError("setup() requires either 'encoder' or 'img_branch'.")

        self.device = device
        if "segmentation" in self.TASK:
            head_dim = getattr(self.encoder, "patch_embed_dim", self.encoder.embed_dim)
        else:
            head_dim = self.encoder.embed_dim

        if self.cfg.freeze_backbone and img_branch is not None:
            for p in img_branch.parameters():
                p.requires_grad_(False)
            img_branch.eval()

        self._apply_backbone_freeze()
        try:
            backbone_dtype = next(
                p for mod in self.encoder._nn_modules()
                for p in mod.parameters()
            ).dtype
        except StopIteration:
            backbone_dtype = torch.bfloat16

        self.head = self.build_head(head_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[{self.EXPERIMENT_NAME}] Head: {self.head} (dtype={backbone_dtype})")
        if self.cfg.freeze_backbone:
            log.info(f"[{self.EXPERIMENT_NAME}] Backbone frozen — training task head(s) only")

    def _apply_backbone_freeze(self) -> None:
        """Freeze all encoder/backbone weights when freeze_backbone is set."""
        if not self.cfg.freeze_backbone or self.encoder is None:
            return
        for mod in self.encoder._nn_modules():
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)

    def _iter_trainable_params(self):
        """Parameters updated during finetune (heads + optional unfrozen encoder parts)."""
        yield from self.head.parameters()
        if hasattr(self, "head2") and self.head2 is not None:
            yield from self.head2.parameters()
        if not self.cfg.freeze_backbone and self.encoder is not None:
            yield from self.encoder.trainable_parameters()

    # ── Training loop ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Full finetune training loop with early stopping.
        Returns the training log (list of per-epoch dicts).
        """
        assert self.head is not None, "Call setup() before run()"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_loader = self.build_dataloader("train")
        val_loader   = self.build_dataloader("val")

        all_params = list(self._iter_trainable_params())
        optimiser = torch.optim.AdamW(
            all_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        # BF16 does not need loss scaling; GradScaler unscale is not implemented for BF16 on some CUDA builds.
        use_amp_scaler = next(self.head.parameters()).dtype != torch.bfloat16
        scaler = GradScaler(enabled=torch.cuda.is_available() and use_amp_scaler)

        log.info(f"[{self.EXPERIMENT_NAME}] Training for up to "
                 f"{self.cfg.max_epochs} epochs on {self.DATASET_ID}")

        for epoch in range(self.cfg.max_epochs):
            self._current_epoch = epoch
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
        best_path = self._head_checkpoint_path("best_head.pt")
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
                # encode_image() handles its own no_grad for frozen backbone;
                # TemporalAttentionPool (if any) runs outside no_grad for gradients.
                pmask = batch.get("padding_mask")
                if pmask is not None:
                    feats = self.encoder.encode_image(batch["image"], padding_mask=pmask)
                else:
                    feats = self.encoder.encode_image(batch["image"])
                if self._head_takes_patch_tokens():
                    from models.heads.finetune_seg import forward_seg_head
                    head_out = forward_seg_head(self.head, feats, padding_mask=pmask)
                else:
                    head_out = self.head(feats["cls"])

                loss = self.compute_loss(batch, feats, head_out)

            self._backward_step_with_scaler(
                loss, optimiser, scaler, list(self._iter_trainable_params())
            )
            optimiser.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def _head_takes_patch_tokens(self) -> bool:
        """
        Returns True if head.forward() takes patch_tokens (segmentation heads),
        False if it takes cls token (classification/regression heads).
        """
        from models.heads.finetune_seg import is_hierarchical_seg_head
        if is_hierarchical_seg_head(self.head):
            return True
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
                encoder=self.encoder,
                img_branch=self.img_branch,   # kept for legacy benchmarks
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
        self.run_viz(results, self.output_dir)
        return results

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _head_checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / name

    def _save_head(self, name: str = "best_head.pt"):
        path = self._head_checkpoint_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), path)
        log.info(f"[{self.EXPERIMENT_NAME}] Checkpoint → {path}")

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
        results = dict(results)
        results["results_dir"] = str(self.output_dir)
        results["checkpoint_dir"] = str(self.checkpoint_dir)
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
        parser.add_argument("--output-dir", default="dataset_exploration_outputs/finetune",
                            help="Where to write metrics, logs, and visualizations")
        parser.add_argument("--checkpoint-dir", default=None,
                            help="Where to save task-head weights (default: Capstor Finetune/)")
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
            data_root      = args.data_root,
            output_dir     = args.output_dir,
            cfg            = cfg,
            checkpoint_dir = args.checkpoint_dir,
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
            best = experiment._head_checkpoint_path("best_head.pt")
            if best.exists():
                experiment.load_head(str(best))
        else:
            experiment.run()

        results = experiment.evaluate("test")

        log.info("Test results:\n%s", json.dumps(results, indent=2))
