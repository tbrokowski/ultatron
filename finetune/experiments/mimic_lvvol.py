"""
finetune/experiments/mimic_lvvol.py  ·  MIMIC-IV-Echo-LVVol-A4C EF regression
===============================================================================

Task:    Predict LVEF from apical 4-chamber DICOM echocardiography clips.
Dataset: MIMIC-IV-Echo-LVVol-A4C — 1,007 clips with expert LV measurements.
Branch:  Video branch (V-JEPA2 teacher).
Head:    RegressionHead on clip_cls token (EF target).
Loss:    MSE + 0.1 × MAE.
Metric:  MAE, RMSE, R², Pearson r.

Layout:
  {root}/physionet.org/files/mimic-iv-echo-ext-lvvol-a4c/1.0.0/
      FileList.csv   study_id, LVEDV_A4C, LVESV_A4C, LVEF_A4C, …
      dicom/{study_id}.dcm
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from models.heads import RegressionHead
from eval.metrics import mae, rmse, pearson_r, r2_score

log = logging.getLogger(__name__)

N_FRAMES   = 32
CLIP_SIZE  = 112
_SUBPATH   = Path("physionet.org/files/mimic-iv-echo-ext-lvvol-a4c/1.0.0")


def _load_dicom_clip(path: str, n_frames: int) -> torch.Tensor:
    """Load a DICOM cine file as a float tensor (T, 3, H, W)."""
    try:
        import pydicom
        ds     = pydicom.dcmread(path)
        pixels = ds.pixel_array.astype(np.float32)   # (T, H, W) or (H, W)
        if pixels.ndim == 2:
            pixels = pixels[None]
        # Sample n_frames evenly
        indices = np.linspace(0, len(pixels) - 1, n_frames, dtype=int)
        frames  = pixels[indices]
        # Normalise per-clip
        pmin, pmax = frames.min(), frames.max()
        frames = (frames - pmin) / (pmax - pmin + 1e-8)
    except Exception:
        frames = np.zeros((n_frames, CLIP_SIZE, CLIP_SIZE), dtype=np.float32)

    clip = []
    for f in frames:
        t = torch.from_numpy(f).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
        t = F.interpolate(t, (CLIP_SIZE, CLIP_SIZE), mode="bilinear", align_corners=False)
        t = t.squeeze(0).repeat(3, 1, 1)                    # (3, H, W)
        clip.append(t)
    return torch.stack(clip)   # (T, 3, H, W)


class MIMICLVVolDataset(Dataset):
    """
    MIMIC-IV-Echo-LVVol-A4C dataset.

    Returns:
        clip      : (T, 3, 112, 112) float32
        target    : scalar float32 (LVEF_A4C %)
        sample_id : str
    """

    def __init__(self, root: str, split: str, n_frames: int = N_FRAMES):
        self.n_frames = n_frames
        data_root     = self._resolve_root(Path(root))
        self.samples  = self._load(data_root, split)

    @staticmethod
    def _resolve_root(root: Path) -> Path:
        deep = root / _SUBPATH
        return deep if deep.exists() else root

    def _load(self, data_root: Path, split: str) -> list[dict]:
        csv_path  = data_root / "FileList.csv"
        dicom_dir = data_root / "dicom"
        if not csv_path.exists():
            log.warning(f"MIMIC-LVVol FileList.csv missing at {csv_path}")
            return []
        rows = []
        with open(csv_path) as f:
            all_rows = list(csv.DictReader(f))
        n = len(all_rows)
        for i, row in enumerate(all_rows):
            # Infer split from index (no explicit split column)
            frac = i / max(n - 1, 1)
            row_split = "train" if frac < 0.80 else ("val" if frac < 0.90 else "test")
            if row_split != split.lower():
                continue
            study_id = row["study_id"]
            dcm      = dicom_dir / f"{study_id}.dcm"
            if not dcm.exists():
                continue
            try:
                ef = float(row.get("LVEF_A4C") or row.get("LVEF_BP") or "nan")
            except (ValueError, TypeError):
                ef = float("nan")
            if np.isnan(ef):
                continue
            rows.append({"path": str(dcm), "ef": ef, "sample_id": study_id})
        log.info(f"MIMIC-LVVol {split}: {len(rows)} samples")
        return rows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {"clip": _load_dicom_clip(s["path"], self.n_frames),
                "target": torch.tensor(s["ef"], dtype=torch.float32),
                "sample_id": s["sample_id"]}


class MIMICLVVolFinetune(FinetuneExperiment):
    """MIMIC-IV-Echo-LVVol-A4C LVEF regression from DICOM clips."""

    EXPERIMENT_NAME = "mimic_lvvol_ef_regression"
    DATASET_ID      = "MIMIC-IV-Echo-LVVol-A4C"
    TASK            = "regression"
    BENCHMARK_CLS   = None

    def build_head(self, embed_dim: int, cfg: FinetuneConfig):
        return RegressionHead(embed_dim=embed_dim, hidden_dim=256,
                              output_min=10.0, output_max=85.0)

    def setup(self, img_branch, device="cuda", vid_branch=None):
        self.img_branch = img_branch
        self.vid_branch = vid_branch
        self.device     = device
        assert vid_branch is not None, "MIMICLVVolFinetune requires vid_branch"
        if self.cfg.freeze_backbone:
            for p in vid_branch.parameters():
                p.requires_grad_(False)
            vid_branch.eval()
        embed_dim      = vid_branch.embed_dim
        backbone_dtype = next(vid_branch.parameters()).dtype
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[MIMICLVVol] head={self.head}  dtype={backbone_dtype}")

    def build_dataloader(self, split: str) -> DataLoader:
        ds = MIMICLVVolDataset(str(self.data_root), split)
        return DataLoader(ds, batch_size=self.cfg.batch_size,
                          shuffle=(split == "train"),
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def compute_loss(self, batch, feats, head_output):
        t = batch["target"]
        p = head_output
        return F.mse_loss(p, t) + 0.1 * (p - t).abs().mean()

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
            all_pred.extend(pred.cpu().float().tolist())
            all_true.extend(batch["target"].cpu().float().tolist())
        p, t = np.array(all_pred), np.array(all_true)
        return {"val_loss": round(total_loss / max(n, 1), 4),
                "val_mae":  round(mae(p, t), 3),
                "val_rmse": round(rmse(p, t), 3),
                "val_r2":   round(r2_score(p, t), 4),
                "val_r":    round(pearson_r(p, t), 4)}

    def run_viz(self, results: dict, output_dir: Path) -> None:
        pass


if __name__ == "__main__":
    MIMICLVVolFinetune.main()
