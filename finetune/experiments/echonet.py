"""
finetune/experiments/echonet.py  ·  EchoNet-Dynamic EF regression finetune
====================================================================

Task:    Predict ejection fraction (%) from apical 4-chamber cine clips.
Dataset: EchoNet-Dynamic — 10,030 labelled echocardiogram videos.
Branch:  Video — full cine loops (32 frames × 112×112).

This is a **video** task: EF depends on motion across the cardiac cycle.
Encoder handling differs by backbone type:

  video_native (Student, V-JEPA, Ultatron):
      encode_video(clips) → temporally fused clip_cls → VideoRegressionHead

  frame_based (ResNet, ViT, BioMed-CLIP, USFM, EchoCare, …):
      per-frame encode_image → frame_tokens (B, T, D)
      → trainable attention pool inside VideoRegressionHead → EF

Head:    VideoRegressionHead (mlp) or RegressionHead (linear legacy).
Loss:    MSE + 0.1 × MAE.
Metric:  MAE (primary), RMSE, R², Pearson r.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from finetune.video_regression import (
    encode_clip_for_regression,
    regression_head_forward,
    regression_head_params,
)
from models.heads import build_video_regression_head
from eval.metrics import mae, rmse, pearson_r, r2_score
from eval.benchmarks.echonet import EchoNetBenchmark

log = logging.getLogger(__name__)
N_FRAMES = 32
CLIP_SIZE = 112


class EchoNetFinetuneDataset(Dataset):
    """
    EchoNet-Dynamic finetune dataset.

    Loads clips from {root}/Videos/*.avi and EF labels from FileList.csv.

    Returns:
        clip      : (T, 3, 112, 112) float32 [0, 1]
        target    : scalar float32 (EF %)
        sample_id : str
    """

    def __init__(self, root: str, split: str = "TRAIN", n_frames: int = N_FRAMES):
        self.root     = Path(root)
        self.n_frames = n_frames
        self.samples  = self._load(split.upper())

    def _load(self, split: str) -> list[dict]:
        rows = []
        with open(self.root / "FileList.csv") as f:
            for row in csv.DictReader(f):
                if row.get("Split", "").upper() != split:
                    continue
                fname = row["FileName"]
                if not fname.endswith(".avi"):
                    fname += ".avi"
                vpath = self.root / "Videos" / fname
                if vpath.exists():
                    rows.append({"path": str(vpath), "ef": float(row.get("EF", 0)),
                                 "sample_id": fname.replace(".avi", "")})
        log.info(f"EchoNet {split}: {len(rows)} samples")
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        clip = self._load_clip(s["path"])
        return {"clip": clip,
                "target": torch.tensor(s["ef"], dtype=torch.float32),
                "sample_id": s["sample_id"]}

    def _load_clip(self, path: str) -> torch.Tensor:
        try:
            from decord import VideoReader, cpu
            vr      = VideoReader(path, ctx=cpu(0))
            indices = np.linspace(0, len(vr) - 1, self.n_frames, dtype=int)
            frames  = vr.get_batch(indices.tolist()).asnumpy()   # (T, H, W, 3)
        except Exception:
            frames  = np.zeros((self.n_frames, CLIP_SIZE, CLIP_SIZE, 3), dtype=np.uint8)

        clip = []
        for f in frames:
            t = torch.from_numpy(f).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            t = F.interpolate(t, size=(CLIP_SIZE, CLIP_SIZE),
                               mode="bilinear", align_corners=False)
            clip.append(t.squeeze(0))
        return torch.stack(clip, dim=0)   # (T, 3, 112, 112)


class EchoNetFinetune(FinetuneExperiment):
    """
    EchoNet-Dynamic EF regression finetune.

    Uses encoder-specific video encoding (native vs per-frame) and a
    literature-aligned regression head.
    """

    EXPERIMENT_NAME = "echonet_ef_regression"
    DATASET_ID      = "EchoNet-Dynamic"
    TASK            = "regression"
    BENCHMARK_CLS   = EchoNetBenchmark

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        video_native = getattr(self, "_video_native", True)
        return build_video_regression_head(
            head_type    = cfg.head_type,
            embed_dim    = embed_dim,
            video_native = video_native,
            hidden_dim   = 512,
            dropout      = 0.2,
            output_min   = 10.0,
            output_max   = 85.0,
        )

    def setup(self, img_branch=None, device="cuda", vid_branch=None, encoder=None):
        """Wire encoder and build head on video_embed_dim."""
        from finetune.backbones.ultatron_encoder import UltatronBranchEncoder
        if encoder is not None:
            self.encoder    = encoder
            self.img_branch = encoder
            self.vid_branch = None
        else:
            assert vid_branch is not None, "EchoNetFinetune requires vid_branch"
            self.encoder    = UltatronBranchEncoder(img_branch, vid_branch)
            self.img_branch = img_branch
            self.vid_branch = vid_branch
        self.device = device
        self._video_native = self.encoder.is_video_native

        if self.cfg.freeze_backbone and vid_branch is not None:
            for p in vid_branch.parameters():
                p.requires_grad_(False)
            vid_branch.eval()

        embed_dim = self.encoder.video_embed_dim
        try:
            backbone_dtype = next(
                p for mod in self.encoder._nn_modules()
                for p in mod.parameters()
            ).dtype
        except StopIteration:
            backbone_dtype = torch.bfloat16
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        mode = "video_native" if self._video_native else "frame_attention"
        log.info(f"[EchoNet] Regression head: {self.head}  encoding={mode}  dtype={backbone_dtype}")

    def build_dataloader(self, split: str) -> DataLoader:
        split_map = {"train": "TRAIN", "val": "VAL", "test": "TEST"}
        return DataLoader(
            EchoNetFinetuneDataset(str(self.data_root), split_map.get(split, split)),
            batch_size=self.cfg.batch_size, shuffle=(split == "train"),
            num_workers=self.cfg.num_workers, pin_memory=True,
        )

    def compute_loss(self, batch, feats, head_output) -> torch.Tensor:
        target = batch["target"]                              # (B,)
        pred   = head_output                                  # (B,)
        mse    = F.mse_loss(pred, target)
        mae_l  = (pred - target).abs().mean()
        return mse + 0.1 * mae_l

    def _predict_batch(self, clips: torch.Tensor) -> torch.Tensor:
        enc_out = encode_clip_for_regression(self.encoder, clips)
        return regression_head_forward(self.head, enc_out)

    def _head_params(self):
        return regression_head_params(self.head)

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train()
        self.encoder.eval()
        total_loss = 0.0
        n          = 0

        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
                pred = self._predict_batch(batch["clip"])
                loss = self.compute_loss(batch, {}, pred)

            self._backward_step_with_scaler(
                loss, optimiser, scaler, self._head_params()
            )
            optimiser.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n          += 1

        return total_loss / max(n, 1)

    def run(self) -> dict:
        """Train with head params that include frame attention when applicable."""
        assert self.head is not None, "Call setup() before run()"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_loader = self.build_dataloader("train")
        val_loader   = self.build_dataloader("val")

        optimiser = torch.optim.AdamW(
            self._head_params(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        use_amp_scaler = next(self.head.parameters()).dtype != torch.bfloat16
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=torch.cuda.is_available() and use_amp_scaler)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=self.cfg.max_epochs, eta_min=self.cfg.lr * 0.01,
        )

        log.info(f"[{self.EXPERIMENT_NAME}] Training for up to "
                 f"{self.cfg.max_epochs} epochs on {self.DATASET_ID}")

        best_metric = self._best_metric
        patience_ctr = self._patience_ctr

        for epoch in range(self.cfg.max_epochs):
            train_loss = self._train_epoch(train_loader, optimiser, scaler)
            val_metrics = self.compute_val_metrics(val_loader)
            scheduler.step()

            entry = {"epoch": epoch, "train_loss": round(train_loss, 4), **val_metrics}
            self._train_log.append(entry)

            if epoch % self.cfg.log_every == 0 or epoch == self.cfg.max_epochs - 1:
                log.info(f"  Epoch {epoch:3d}  " + "  ".join(
                    f"{k}={v}" for k, v in entry.items() if k != "epoch"
                ))

            monitor_val = val_metrics.get(self.cfg.monitor, val_metrics.get("val_loss"))
            improved = (
                monitor_val < best_metric if self.cfg.monitor_mode == "min"
                else monitor_val > best_metric
            )
            if improved:
                best_metric = monitor_val
                self._best_metric = monitor_val
                patience_ctr = 0
                self._patience_ctr = 0
                if self.cfg.checkpoint_best:
                    self._save_head("best_head.pt")
            else:
                patience_ctr += 1
                self._patience_ctr = patience_ctr
                if patience_ctr >= self.cfg.patience:
                    log.info(f"  Early stopping at epoch {epoch} (patience={self.cfg.patience})")
                    break

        best_path = self.output_dir / "best_head.pt"
        if best_path.exists():
            self.load_head(str(best_path))

        self._save_log()
        return {"train_log": self._train_log, "best_metric": self._best_metric}

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.encoder.eval()

        all_pred, all_true = [], []
        total_loss = 0.0
        n          = 0

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = self._predict_batch(batch["clip"])
            loss = self.compute_loss(batch, {}, pred)
            total_loss += loss.item()
            n          += 1
            all_pred.extend(pred.cpu().float().numpy().tolist())
            all_true.extend(batch["target"].cpu().float().numpy().tolist())

        pred_arr = np.array(all_pred)
        true_arr = np.array(all_true)

        return {
            "val_loss":  round(total_loss / max(n, 1), 4),
            "val_mae":   round(mae(pred_arr, true_arr), 3),
            "val_rmse":  round(rmse(pred_arr, true_arr), 3),
            "val_r2":    round(r2_score(pred_arr, true_arr), 4),
            "val_r":     round(pearson_r(pred_arr, true_arr), 4),
        }

    def evaluate(self, split: str = "test") -> dict:
        """Override to instantiate EchoNetBenchmark with the encoder."""
        assert self.head is not None, "Call setup() and run() first"
        self.head.eval()
        benchmark = EchoNetBenchmark(
            encoder=self.encoder,
            vid_branch=None if self.encoder is not None else self.vid_branch,
            reg_head=self.head,
            device=self.device,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )
        results = benchmark.run(str(self.data_root), split=split.upper())
        results["experiment"] = self.EXPERIMENT_NAME
        self._save_results(results)
        return results

    def run_viz(self, results: dict, output_dir: Path) -> None:
        try:
            from viz.regression import (
                plot_regression_scatter, plot_bland_altman, plot_error_distribution
            )
            from viz.core import save_figure
        except ImportError:
            return

        test_loader = self.build_dataloader("test")
        all_pred, all_true = [], []

        self.head.eval()
        self.encoder.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                pred = self._predict_batch(batch["clip"])
                all_pred.extend(pred.cpu().numpy().tolist())
                all_true.extend(batch["target"].cpu().numpy().tolist())

        p = np.array(all_pred)
        t = np.array(all_true)

        fig1 = plot_regression_scatter(p, t, title="EchoNet-Dynamic EF Regression")
        save_figure(fig1, output_dir / "echonet_scatter.png")

        fig2 = plot_bland_altman(p, t, title="EchoNet Bland-Altman Agreement")
        save_figure(fig2, output_dir / "echonet_bland_altman.png")

        fig3 = plot_error_distribution(p, t, title="EchoNet Absolute Error Distribution")
        save_figure(fig3, output_dir / "echonet_error_dist.png")


if __name__ == "__main__":
    EchoNetFinetune.main()
