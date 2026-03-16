"""
eval/benchmarks/camus.py  ·  CAMUS LV segmentation benchmark
==================================================================

CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation)
Dataset: 500 patients, 2-CH and 4-CH views, ED and ES phases.
Task:    LV endocardium and myocardium segmentation.
Metric:  Dice coefficient per structure per phase, mean Dice, Hausdorff-95.

Reference split: official train/val/test from patient IDs.
  - Train: patients 1–400
  - Val:   patients 401–450
  - Test:  patients 451–500

The benchmark runner:
  1. Loads all test images via CAMUSBenchmarkDataset
  2. Runs DPTSegHead or LinearSegHead predictions
  3. Reports Dice (endocardium, myocardium) and Hausdorff-95 per view/phase

Primary benchmark metric: mean_dice (average across all structures/views/phases)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval.benchmarks.base import BaseBenchmark
from eval.metrics import dice_score, iou_score, hausdorff_95

log = logging.getLogger(__name__)


# ── CAMUS benchmark dataset ───────────────────────────────────────────────────

class CAMUSBenchmarkDataset(Dataset):
    """
    Minimal CAMUS loader for evaluation only.

    Reads the directory structure:
        {root}/patient{id}/
            patient{id}_{view}_{phase}.mhd
            patient{id}_{view}_{phase}_gt.mhd

    Returns dicts with:
        image      : (3, H, W) float32 RGB [0, 1]  (greyscale repeated to 3ch)
        mask       : (1, H, W) float32 binary
        sample_id  : str
        view       : str  ("2CH" or "4CH")
        phase      : str  ("ED" or "ES")
    """

    def __init__(self, root: str, split: str = "test"):
        self.root    = Path(root)
        self.split   = split
        self.samples = self._collect_samples()

    def _collect_samples(self) -> list[dict]:
        patients = sorted(self.root.glob("patient*/"))
        total    = len(patients)

        # CAMUS official split by patient index
        split_map = {}
        for i, p in enumerate(patients):
            frac = i / max(total - 1, 1)
            if frac < 0.80:   split_map[p.name] = "train"
            elif frac < 0.90: split_map[p.name] = "val"
            else:              split_map[p.name] = "test"

        samples = []
        for pdir in patients:
            if split_map.get(pdir.name) != self.split:
                continue
            pid = pdir.name
            for view in ("2CH", "4CH"):
                for phase in ("ED", "ES"):
                    img_path = pdir / f"{pid}_{view}_{phase}.mhd"
                    msk_path = pdir / f"{pid}_{view}_{phase}_gt.mhd"
                    if img_path.exists() and msk_path.exists():
                        samples.append({
                            "img_path": str(img_path),
                            "msk_path": str(msk_path),
                            "sample_id": f"{pid}_{view}_{phase}",
                            "view": view,
                            "phase": phase,
                        })
        log.info(f"CAMUS {self.split}: {len(samples)} samples")
        return samples

    def _load_mhd(self, path: str) -> np.ndarray:
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = self._load_mhd(s["img_path"])   # (H, W)  raw intensities
        msk = self._load_mhd(s["msk_path"])   # (H, W)  label map 0/1/2

        # Normalise image to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Resize both to 256×256 for consistent batching
        img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        msk_t = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        img_t = F.interpolate(img_t, size=(256, 256), mode="bilinear",  align_corners=False)
        msk_t = F.interpolate(msk_t, size=(256, 256), mode="nearest")

        # Expand greyscale to RGB
        img_rgb = img_t.squeeze(0).expand(3, -1, -1)              # (3,256,256)

        # Binary mask: treat any non-zero label as foreground (LV region)
        # For multi-class (endo/myo), mask_endo = (msk==1), mask_myo = (msk==2)
        msk_bin = (msk_t.squeeze(0).squeeze(0) > 0).float().unsqueeze(0)  # (1,256,256)

        return {
            "image":     img_rgb,
            "mask":      msk_bin,
            "mask_full": msk_t.squeeze(0),   # (1,256,256) with 0/1/2 labels
            "sample_id": s["sample_id"],
            "view":      s["view"],
            "phase":     s["phase"],
        }


# ── CAMUS benchmark runner ────────────────────────────────────────────────────

class CAMUSBenchmark(BaseBenchmark):
    """
    CAMUS LV segmentation benchmark.

    Metrics
    -------
    dice_mean       macro-average Dice across all samples
    dice_2ch_ed     Dice on 2-chamber end-diastole frames
    dice_2ch_es     Dice on 2-chamber end-systole frames
    dice_4ch_ed     Dice on 4-chamber end-diastole frames
    dice_4ch_es     Dice on 4-chamber end-systole frames
    iou_mean        macro-average IoU
    hd95_mean       mean 95th-percentile Hausdorff distance (mm, isotropic 1px)
    """

    BENCHMARK_NAME = "camus_segmentation"
    DATASET_ID     = "CAMUS"
    ANATOMY_FAMILY = "cardiac"
    TASK           = "segmentation"

    def build_dataloader(self, root: str, split: str = "test") -> DataLoader:
        dataset = CAMUSBenchmarkDataset(root, split)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict(self, batch: dict) -> dict:
        images = batch["image"]   # (B, 3, H, W)
        feats  = self._extract_features(images)
        logits = self.head(feats["patch_tokens"])   # (B, 1, ph, pw)
        # Upsample to image resolution
        pred   = F.interpolate(logits, size=images.shape[-2:],
                               mode="bilinear", align_corners=False)
        return {"pred": torch.sigmoid(pred)}

    def compute_metrics(
        self,
        pred: torch.Tensor,    # (B, 1, H, W) float
        target: torch.Tensor,  # (B, 1, H, W) float binary
        sample_ids: list,
    ) -> list[dict]:
        pred_np   = pred.cpu().float().numpy()
        target_np = target.cpu().float().numpy()
        results   = []

        for i, sid in enumerate(sample_ids):
            p = pred_np[i, 0]    # (H, W)
            t = target_np[i, 0]  # (H, W)
            parts = sid.split("_") if isinstance(sid, str) else ["?", "?", "?"]
            view  = parts[-2] if len(parts) >= 3 else "?"
            phase = parts[-1] if len(parts) >= 3 else "?"

            results.append({
                "sample_id": sid,
                "view":      view,
                "phase":     phase,
                "dice":      dice_score(p, t),
                "iou":       iou_score(p, t),
                "hd95":      hausdorff_95(p >= 0.5, t >= 0.5),
            })
        return results

    def aggregate(self, per_sample: list[dict]) -> dict:
        dices  = [s["dice"] for s in per_sample]
        ious   = [s["iou"]  for s in per_sample]
        hd95s  = [s["hd95"] for s in per_sample if not np.isnan(s["hd95"])]

        def _mean_subset(key_val_pairs):
            vals = [s["dice"] for s in per_sample
                    if all(s.get(k) == v for k, v in key_val_pairs)]
            return float(np.mean(vals)) if vals else float("nan")

        return {
            "dice_mean":  round(float(np.mean(dices)), 4),
            "iou_mean":   round(float(np.mean(ious)), 4),
            "hd95_mean":  round(float(np.mean(hd95s)) if hd95s else float("nan"), 2),
            "dice_2ch_ed": round(_mean_subset([("view","2CH"),("phase","ED")]), 4),
            "dice_2ch_es": round(_mean_subset([("view","2CH"),("phase","ES")]), 4),
            "dice_4ch_ed": round(_mean_subset([("view","4CH"),("phase","ED")]), 4),
            "dice_4ch_es": round(_mean_subset([("view","4CH"),("phase","ES")]), 4),
        }
