"""
finetune/experiments/echonet.py  ·  EchoNet-Dynamic EF regression finetune
====================================================================

Task:    Predict ejection fraction (%) from apical 4-chamber cine clips.
Dataset: EchoNet-Dynamic — 10,030 labelled echocardiogram videos.
Branch:  Video branch (V-JEPA2 teacher — not the image branch).
Head:    RegressionHead on clip_cls token.
Loss:    MSE (primary) + MAE penalty.
Metric:  MAE (primary), RMSE, R², Pearson r.

EchoNet is the only finetune that uses the video branch rather than the
image branch.  The base class is designed for image; we override setup()
and _train_epoch() to use vid_branch instead.
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
from models.heads import RegressionHead
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

    Uses the video branch (vid_branch) not the image branch.
    Overrides setup() and _train_epoch() accordingly.
    """

    EXPERIMENT_NAME = "echonet_ef_regression"
    DATASET_ID      = "EchoNet-Dynamic"
    TASK            = "regression"
    BENCHMARK_CLS   = EchoNetBenchmark

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        """
        Regression head on the video clip_cls token.
        embed_dim is the video backbone's hidden size (e.g. 1024 for VJEPA2-L).
        output_min/max clamp to physiologically valid EF range.
        """
        return RegressionHead(
            embed_dim  = embed_dim,
            hidden_dim = 256,
            output_min = 10.0,   # physiologically: EF below 10% is non-viable
            output_max = 85.0,   # physiologically: EF above 85% is extremely high
        )

    def setup(self, img_branch, device="cuda", vid_branch=None):
        """Override to freeze vid_branch instead of img_branch."""
        self.img_branch = img_branch
        self.vid_branch = vid_branch
        self.device     = device

        assert vid_branch is not None, "EchoNetFinetune requires vid_branch"

        if self.cfg.freeze_backbone:
            for p in vid_branch.parameters():
                p.requires_grad_(False)
            vid_branch.eval()

        embed_dim = vid_branch.embed_dim
        backbone_dtype = next(vid_branch.parameters()).dtype
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[EchoNet] Regression head: {self.head} (dtype={backbone_dtype})")

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

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        """Override to use vid_branch instead of img_branch."""
        self.head.train()
        self.vid_branch.teacher.eval()   # frozen
        total_loss = 0.0
        n          = 0

        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    vid_out = self.vid_branch.forward_teacher(batch["clip"])
                pred = self.head(vid_out["clip_cls"])
                loss = self.compute_loss(batch, {}, pred)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n          += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.vid_branch.teacher.eval()

        all_pred, all_true = [], []
        total_loss = 0.0
        n          = 0

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            vid_out = self.vid_branch.forward_teacher(batch["clip"])
            pred    = self.head(vid_out["clip_cls"])
            loss    = self.compute_loss(batch, {}, pred)
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
        self.vid_branch.teacher.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                vid_out = self.vid_branch.forward_teacher(batch["clip"])
                pred    = self.head(vid_out["clip_cls"])
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
