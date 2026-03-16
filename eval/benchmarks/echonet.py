"""
oura/eval/benchmarks/echonet.py  ·  EchoNet-Dynamic EF regression benchmark
============================================================================

EchoNet-Dynamic: 10,030 echocardiogram videos with expert EF labels.
Task:    Predict ejection fraction (%) from apical 4-chamber cine clips.
Metric:  MAE (primary), RMSE, R², Pearson r.
         EF ∈ [0, 100] — clinical threshold: EF < 50% = reduced function.

Protocol
--------
  1. Load test-split videos via EchoNetBenchmarkDataset.
  2. Run video branch (VJEPA2) teacher to get clip representation.
  3. Pass clip_cls through RegressionHead → EF prediction.
  4. Report MAE, RMSE, R², Pearson r on test split.

Reference: EchoNet-Dynamic paper reports MAE ≈ 4.1% with ResNet50+LSTM.
           A strong SSL baseline should achieve MAE < 5%.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval.benchmarks.base import BaseBenchmark
from eval.metrics import mae, rmse, pearson_r, r2_score, ef_mae, ef_r2

log = logging.getLogger(__name__)


class EchoNetBenchmarkDataset(Dataset):
    """
    EchoNet-Dynamic test set loader.

    Reads:
        {root}/Videos/*.avi
        {root}/FileList.csv   (FileName, Split, EF, ...)
    """

    def __init__(self, root: str, split: str = "TEST", n_frames: int = 32):
        self.root     = Path(root)
        self.split    = split.upper()
        self.n_frames = n_frames
        self.samples  = self._load_filelist()

    def _load_filelist(self) -> list[dict]:
        csv_path = self.root / "FileList.csv"
        samples  = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get("Split", "").upper() != self.split:
                    continue
                fname = row["FileName"]
                if not fname.endswith(".avi"):
                    fname += ".avi"
                vpath = self.root / "Videos" / fname
                if not vpath.exists():
                    continue
                ef = float(row.get("EF", 0.0))
                samples.append({"path": str(vpath), "ef": ef,
                                 "sample_id": fname.replace(".avi", "")})
        log.info(f"EchoNet {self.split}: {len(samples)} samples")
        return samples

    def _load_clip(self, path: str) -> torch.Tensor:
        """Load video clip → (T, 3, 112, 112) float32 [0, 1]."""
        try:
            from decord import VideoReader, cpu
            vr      = VideoReader(path, ctx=cpu(0))
            total   = len(vr)
            indices = np.linspace(0, total - 1, self.n_frames, dtype=int).tolist()
            frames  = vr.get_batch(indices).asnumpy()   # (T, H, W, 3)
        except Exception:
            # Fallback: return black clip if video unreadable
            frames = np.zeros((self.n_frames, 112, 112, 3), dtype=np.uint8)

        # Resize to 112×112 and normalise
        clip = []
        for f in frames:
            t = torch.from_numpy(f).float() / 255.0            # (H, W, 3)
            t = t.permute(2, 0, 1).unsqueeze(0)                # (1, 3, H, W)
            t = F.interpolate(t, size=(112, 112), mode="bilinear", align_corners=False)
            clip.append(t.squeeze(0))
        return torch.stack(clip, dim=0)   # (T, 3, 112, 112)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s    = self.samples[idx]
        clip = self._load_clip(s["path"])
        return {
            "clip":      clip,
            "target":    torch.tensor(s["ef"], dtype=torch.float32),
            "sample_id": s["sample_id"],
        }


class EchoNetBenchmark(BaseBenchmark):
    """
    EchoNet-Dynamic ejection fraction regression benchmark.

    Requires:
        vid_branch   : VideoBranch (video teacher used)
        reg_head     : RegressionHead (EF prediction head)
    """

    BENCHMARK_NAME = "echonet_ef_regression"
    DATASET_ID     = "EchoNet-Dynamic"
    ANATOMY_FAMILY = "cardiac"
    TASK           = "regression"

    def __init__(self, vid_branch, reg_head, device="cuda",
                 batch_size=8, num_workers=4, n_frames=32):
        # Note: passes None as img_branch — this benchmark uses vid_branch
        super().__init__(img_branch=None, head=reg_head,
                         device=device, batch_size=batch_size,
                         num_workers=num_workers)
        self.vid_branch = vid_branch
        self.n_frames   = n_frames

    def build_dataloader(self, root: str, split: str = "TEST") -> DataLoader:
        dataset = EchoNetBenchmarkDataset(root, split, self.n_frames)
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)

    def predict(self, batch: dict) -> dict:
        clips = batch["clip"].to(self.device)  # (B, T, 3, 112, 112)
        with torch.no_grad():
            vid_out = self.vid_branch.forward_teacher(clips)
        ef_pred = self.head(vid_out["clip_cls"])   # (B,) regression output
        return {"pred": ef_pred}

    def compute_metrics(self, pred, target, sample_ids) -> list[dict]:
        pred_np   = pred.cpu().float().numpy()
        target_np = target.cpu().float().numpy()
        return [
            {
                "sample_id": sid,
                "pred_ef":   float(pred_np[i]),
                "true_ef":   float(target_np[i]),
                "abs_error": float(abs(pred_np[i] - target_np[i])),
            }
            for i, sid in enumerate(sample_ids)
        ]

    def aggregate(self, per_sample: list[dict]) -> dict:
        preds   = np.array([s["pred_ef"] for s in per_sample])
        targets = np.array([s["true_ef"] for s in per_sample])

        # Clinical threshold: EF < 50% = reduced
        reduced_mask = targets < 50.0
        if reduced_mask.any():
            mae_reduced = mae(preds[reduced_mask], targets[reduced_mask])
        else:
            mae_reduced = float("nan")

        return {
            "mae":         round(ef_mae(preds, targets), 3),
            "rmse":        round(rmse(preds, targets), 3),
            "r2":          round(ef_r2(preds, targets), 4),
            "pearson_r":   round(pearson_r(preds, targets), 4),
            "mae_reduced": round(mae_reduced, 3),
        }

    def run(self, root: str, split: str = "TEST") -> dict:  # type: ignore[override]
        return super().run(root, split)
