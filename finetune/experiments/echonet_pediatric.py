"""
finetune/experiments/echonet_pediatric.py  ·  EchoNet-Pediatric EF regression
==============================================================================

Task:    Predict ejection fraction (%) from A4C and PSAX cine clips.
Dataset: EchoNet-Pediatric — 7,643 paediatric echocardiogram videos.
Branch:  Video branch (V-JEPA2 teacher).
Head:    RegressionHead on clip_cls token.
Loss:    MSE + 0.1 × MAE.
Metric:  MAE (primary), RMSE, R², Pearson r — reported globally and per view.

Layout:
  {root}/pediatric_echo_avi/pediatric_echo_avi/{A4C,PSAX}/
      Videos/       *.avi
      FileList.csv  FileName, EF, Sex, Age, Weight, Height, Split (0-9)
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from models.heads import RegressionHead
from eval.metrics import mae, rmse, pearson_r, r2_score

log = logging.getLogger(__name__)

N_FRAMES  = 32
CLIP_SIZE = 112
_DATA_PREFIX = Path("pediatric_echo_avi") / "pediatric_echo_avi"
_SPLIT_MAP   = {str(i): "train" for i in range(7)}
_SPLIT_MAP.update({"7": "val", "8": "test", "9": "test"})


class EchoNetPedDataset(Dataset):
    """
    Single-view EchoNet-Pediatric finetune dataset.

    Returns:
        clip      : (T, 3, 112, 112) float32 [0, 1]
        target    : scalar float32 (EF %)
        sample_id : str
        view      : "A4C" | "PSAX"
    """

    def __init__(self, root: str, view: str, split: str, n_frames: int = N_FRAMES):
        self.view_root = Path(root) / _DATA_PREFIX / view
        self.n_frames  = n_frames
        self.view      = view
        self.samples   = self._load(split)

    def _load(self, split: str) -> list[dict]:
        split_key = split.lower()   # "train" / "val" / "test"
        csv_path  = self.view_root / "FileList.csv"
        if not csv_path.exists():
            log.warning(f"EchoNet-Pediatric FileList.csv missing at {csv_path}")
            return []
        rows = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                row_split = _SPLIT_MAP.get(str(row.get("Split", "")), "train")
                if row_split != split_key:
                    continue
                fname = row["FileName"]
                if not fname.endswith(".avi"):
                    fname += ".avi"
                vpath = self.view_root / "Videos" / fname
                if vpath.exists():
                    try:
                        ef = float(row.get("EF", 0))
                    except (ValueError, TypeError):
                        ef = 0.0
                    rows.append({"path": str(vpath), "ef": ef,
                                 "sample_id": f"{self.view}_{fname.replace('.avi', '')}"})
        log.info(f"EchoNet-Pediatric {self.view} {split}: {len(rows)} samples")
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s    = self.samples[idx]
        clip = _load_clip(s["path"], self.n_frames)
        return {"clip": clip,
                "target": torch.tensor(s["ef"], dtype=torch.float32),
                "sample_id": s["sample_id"],
                "view": self.view}


def _load_clip(path: str, n_frames: int) -> torch.Tensor:
    try:
        from decord import VideoReader, cpu
        vr      = VideoReader(path, ctx=cpu(0))
        indices = np.linspace(0, len(vr) - 1, n_frames, dtype=int)
        frames  = vr.get_batch(indices.tolist()).asnumpy()
    except Exception:
        frames = np.zeros((n_frames, CLIP_SIZE, CLIP_SIZE, 3), dtype=np.uint8)

    clip = []
    for f in frames:
        t = torch.from_numpy(f).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, size=(CLIP_SIZE, CLIP_SIZE),
                          mode="bilinear", align_corners=False)
        clip.append(t.squeeze(0))
    return torch.stack(clip, dim=0)   # (T, 3, H, W)


class EchoNetPediatricFinetune(FinetuneExperiment):
    """
    EchoNet-Pediatric EF regression.

    Uses both A4C and PSAX views as one combined dataset.
    Overrides setup() to freeze vid_branch (same pattern as EchoNetFinetune).
    """

    EXPERIMENT_NAME = "echonet_pediatric_ef_regression"
    DATASET_ID      = "EchoNet-Pediatric"
    TASK            = "regression"
    BENCHMARK_CLS   = None  # no dedicated benchmark class yet

    def build_head(self, embed_dim: int, cfg: FinetuneConfig):
        return RegressionHead(
            embed_dim  = embed_dim,
            hidden_dim = 256,
            output_min = 10.0,
            output_max = 85.0,
        )

    def setup(self, img_branch, device="cuda", vid_branch=None):
        self.img_branch = img_branch
        self.vid_branch = vid_branch
        self.device     = device
        assert vid_branch is not None, "EchoNetPediatricFinetune requires vid_branch"
        if self.cfg.freeze_backbone:
            for p in vid_branch.parameters():
                p.requires_grad_(False)
            vid_branch.eval()
        embed_dim      = vid_branch.embed_dim
        backbone_dtype = next(vid_branch.parameters()).dtype
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[EchoNetPed] head={self.head}  dtype={backbone_dtype}")

    def build_dataloader(self, split: str) -> DataLoader:
        root = str(self.data_root)
        datasets = []
        for view in ("A4C", "PSAX"):
            ds = EchoNetPedDataset(root, view, split)
            if len(ds) > 0:
                datasets.append(ds)
        combined = ConcatDataset(datasets) if datasets else EchoNetPedDataset(root, "A4C", split)
        return DataLoader(combined, batch_size=self.cfg.batch_size,
                          shuffle=(split == "train"),
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def compute_loss(self, batch, feats, head_output):
        target = batch["target"]
        pred   = head_output
        return F.mse_loss(pred, target) + 0.1 * (pred - target).abs().mean()

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train()
        self.vid_branch.teacher.eval()
        total_loss, n = 0.0, 0
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
            total_loss += loss.item(); n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.vid_branch.teacher.eval()
        all_pred, all_true = [], []
        total_loss, n = 0.0, 0
        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            vid_out = self.vid_branch.forward_teacher(batch["clip"])
            pred    = self.head(vid_out["clip_cls"])
            loss    = self.compute_loss(batch, {}, pred)
            total_loss += loss.item(); n += 1
            all_pred.extend(pred.cpu().float().tolist())
            all_true.extend(batch["target"].cpu().float().tolist())
        p = np.array(all_pred); t = np.array(all_true)
        return {
            "val_loss": round(total_loss / max(n, 1), 4),
            "val_mae":  round(mae(p, t), 3),
            "val_rmse": round(rmse(p, t), 3),
            "val_r2":   round(r2_score(p, t), 4),
            "val_r":    round(pearson_r(p, t), 4),
        }

    def run_viz(self, results: dict, output_dir: Path) -> None:
        try:
            from viz.regression import plot_regression_scatter, plot_error_distribution
            from viz.core import save_figure
        except ImportError:
            return
        test_loader = self.build_dataloader("test")
        all_pred, all_true = [], []
        self.head.eval(); self.vid_branch.teacher.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                vid_out = self.vid_branch.forward_teacher(batch["clip"])
                pred    = self.head(vid_out["clip_cls"])
                all_pred.extend(pred.cpu().float().tolist())
                all_true.extend(batch["target"].cpu().float().tolist())
        p, t = np.array(all_pred), np.array(all_true)
        save_figure(plot_regression_scatter(p, t, title="EchoNet-Pediatric EF"),
                    output_dir / "echonet_ped_scatter.png")
        save_figure(plot_error_distribution(p, t, title="EchoNet-Pediatric Error"),
                    output_dir / "echonet_ped_error.png")


if __name__ == "__main__":
    EchoNetPediatricFinetune.main()
