"""
finetune/experiments/echonet_lvh.py  ·  EchoNet-LVH LV wall regression
========================================================================

Task:    Predict LV wall thickness measurements from PLAX cine clips.
Dataset: EchoNet-LVH — ~12,000 parasternal long-axis videos (Stanford).
Branch:  Video branch (V-JEPA2 teacher).
Head:    MeasurementHead — 3 outputs: IVSd, LVIDd, LVPWd (diastolic).
Loss:    MSE + 0.1 × MAE per output, averaged.
Metric:  MAE per measurement, overall MAE.

Layout:
  {root}/
    MeasurementsList.csv  HashedFileName, Calc, CalcValue, …, split
    Batch1/*.avi  Batch2/*.avi  Batch3/*.avi  Batch4/*.avi
"""
from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn

from finetune.base import FinetuneExperiment, FinetuneConfig
from eval.metrics import mae

log = logging.getLogger(__name__)

N_FRAMES    = 32
CLIP_SIZE   = 112
_BATCH_DIRS = ("Batch1", "Batch2", "Batch3", "Batch4")
# Diastolic measurements — most clinically stable and commonly reported
_TARGETS    = ("ivsd", "lvidd", "lvpwd")


def _build_index(root: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for b in _BATCH_DIRS:
        for p in (root / b).glob("*.avi"):
            idx[p.stem.upper()] = p
    return idx


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
        t = F.interpolate(t, (CLIP_SIZE, CLIP_SIZE), mode="bilinear", align_corners=False)
        clip.append(t.squeeze(0))
    return torch.stack(clip)


class EchoNetLVHDataset(Dataset):
    """
    EchoNet-LVH dataset.

    Aggregates multiple CSV rows per video into mean CalcValues.
    Only videos with all three diastolic measurements are included
    (IVSd, LVIDd, LVPWd).

    Returns:
        clip    : (T, 3, 112, 112)
        target  : (3,) float32  [IVSd_cm, LVIDd_cm, LVPWd_cm]
        sample_id: str
    """

    def __init__(self, root: str, split: str, n_frames: int = N_FRAMES):
        self.root     = Path(root)
        self.n_frames = n_frames
        self.samples  = self._load(split)

    def _load(self, split: str) -> list[dict]:
        csv_path = self.root / "MeasurementsList.csv"
        if not csv_path.exists():
            log.warning(f"EchoNet-LVH MeasurementsList.csv missing at {csv_path}")
            return []
        video_idx = _build_index(self.root)

        rows_by_file: dict[str, list[dict]] = defaultdict(list)
        split_by_file: dict[str, str]       = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                key = row.get("HashedFileName", "").strip().upper()
                if not key:
                    continue
                rows_by_file[key].append(row)
                if key not in split_by_file:
                    split_by_file[key] = row.get("split", "train").lower()

        samples = []
        for key, rows in rows_by_file.items():
            if split_by_file.get(key, "train") != split.lower():
                continue
            vpath = video_idx.get(key)
            if vpath is None:
                continue
            calc_vals: dict[str, list[float]] = defaultdict(list)
            for r in rows:
                c = r.get("Calc", "").strip().lower()
                v = r.get("CalcValue", "")
                if c and v:
                    try:
                        calc_vals[c].append(float(v))
                    except ValueError:
                        pass
            # Require all three diastolic measurements
            if not all(k in calc_vals for k in _TARGETS):
                continue
            target = torch.tensor(
                [float(np.mean(calc_vals[k])) for k in _TARGETS],
                dtype=torch.float32,
            )
            samples.append({"path": str(vpath), "target": target, "sample_id": key})

        log.info(f"EchoNet-LVH {split}: {len(samples)} samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {"clip": _load_clip(s["path"], self.n_frames),
                "target": s["target"],
                "sample_id": s["sample_id"]}


class EchoNetLVHFinetune(FinetuneExperiment):
    """EchoNet-LVH multi-output LV wall thickness regression."""

    EXPERIMENT_NAME = "echonet_lvh_wall_regression"
    DATASET_ID      = "EchoNet-LVH"
    TASK            = "regression"
    BENCHMARK_CLS   = None

    def build_head(self, embed_dim: int, cfg: FinetuneConfig):
        """3-output regression head for IVSd, LVIDd, LVPWd from clip_cls."""
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, len(_TARGETS)),
        )

    def setup(self, img_branch, device="cuda", vid_branch=None):
        self.img_branch = img_branch
        self.vid_branch = vid_branch
        self.device     = device
        assert vid_branch is not None, "EchoNetLVHFinetune requires vid_branch"
        if self.cfg.freeze_backbone:
            for p in vid_branch.parameters():
                p.requires_grad_(False)
            vid_branch.eval()
        embed_dim      = vid_branch.embed_dim
        backbone_dtype = next(vid_branch.parameters()).dtype
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[EchoNetLVH] head={self.head}  dtype={backbone_dtype}  outputs={_TARGETS}")

    def build_dataloader(self, split: str) -> DataLoader:
        ds = EchoNetLVHDataset(str(self.data_root), split)
        return DataLoader(ds, batch_size=self.cfg.batch_size,
                          shuffle=(split == "train"),
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def compute_loss(self, batch, feats, head_output):
        target = batch["target"]          # (B, 3)
        pred   = head_output              # (B, 3)
        return F.mse_loss(pred, target) + 0.1 * (pred - target).abs().mean()

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train(); self.vid_branch.teacher.eval()
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
            scaler.step(optimiser); scaler.update()
            optimiser.zero_grad(set_to_none=True)
            total_loss += loss.item(); n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval(); self.vid_branch.teacher.eval()
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
            all_pred.append(pred.cpu().float())
            all_true.append(batch["target"].cpu().float())

        p = torch.cat(all_pred).numpy()   # (N, 3)
        t = torch.cat(all_true).numpy()
        metrics = {"val_loss": round(total_loss / max(n, 1), 4)}
        for i, name in enumerate(_TARGETS):
            metrics[f"val_mae_{name}"] = round(mae(p[:, i], t[:, i]), 4)
        metrics["val_mae"] = round(float(np.mean([metrics[f"val_mae_{k}"] for k in _TARGETS])), 4)
        return metrics

    def run_viz(self, results: dict, output_dir: Path) -> None:
        pass   # scatter per measurement — can be added later


if __name__ == "__main__":
    EchoNetLVHFinetune.main()
