"""
oura/eval/benchmarks/tn3k.py  ·  TN3K thyroid nodule segmentation benchmark
=============================================================================

TN3K: 3,493 thyroid ultrasound images with nodule segmentation masks.
Task:    Thyroid nodule boundary segmentation.
Metric:  Dice, IoU, S-measure (structural similarity).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from eval.benchmarks.base import BaseBenchmark
from eval.metrics import dice_score, iou_score

log = logging.getLogger(__name__)


class TN3KBenchmarkDataset(Dataset):
    """
    TN3K loader.
    Layout:
        {root}/image/*.jpg
        {root}/label/*.png
    """

    def __init__(self, root: str, split: str = "test", test_frac: float = 0.15):
        self.root    = Path(root)
        img_dir      = self.root / "image"
        lbl_dir      = self.root / "label"
        all_imgs     = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        n_test       = max(1, int(len(all_imgs) * test_frac))
        subset       = all_imgs[-n_test:] if split == "test" else all_imgs[:-n_test]
        self.samples = [
            {"img_path": str(p),
             "lbl_path": str(lbl_dir / (p.stem + ".png")),
             "sample_id": p.stem}
            for p in subset
            if (lbl_dir / (p.stem + ".png")).exists()
        ]
        log.info(f"TN3K {split}: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img_path"]).convert("RGB"), dtype=np.float32) / 255.0
        lbl = np.array(Image.open(s["lbl_path"]).convert("L"),   dtype=np.float32) / 255.0

        img_t = torch.from_numpy(img).permute(2, 0, 1)
        img_t = F.interpolate(img_t.unsqueeze(0), size=(224, 224),
                               mode="bilinear", align_corners=False).squeeze(0)
        lbl_t = torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0)
        lbl_t = F.interpolate(lbl_t, size=(224, 224), mode="nearest").squeeze(0)
        mask  = (lbl_t > 0.5).float()

        return {"image": img_t, "mask": mask, "sample_id": s["sample_id"]}


class TN3KBenchmark(BaseBenchmark):
    """
    TN3K thyroid nodule segmentation benchmark.

    Metrics: dice_mean, iou_mean, s_measure (structural similarity)
    """

    BENCHMARK_NAME = "tn3k_segmentation"
    DATASET_ID     = "TN3K"
    ANATOMY_FAMILY = "thyroid"
    TASK           = "segmentation"

    def build_dataloader(self, root: str, split: str = "test") -> DataLoader:
        return DataLoader(TN3KBenchmarkDataset(root, split),
                          batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def predict(self, batch: dict) -> dict:
        feats  = self._extract_features(batch["image"])
        logits = self.head(feats["patch_tokens"])
        pred   = F.interpolate(logits, size=(224, 224),
                               mode="bilinear", align_corners=False)
        return {"pred": torch.sigmoid(pred)}

    def compute_metrics(self, pred, target, sample_ids) -> list[dict]:
        pred_np   = pred.cpu().float().numpy()
        target_np = target.cpu().float().numpy()
        return [
            {
                "sample_id": sample_ids[i] if i < len(sample_ids) else "",
                "dice": dice_score(pred_np[i, 0], target_np[i, 0]),
                "iou":  iou_score(pred_np[i, 0], target_np[i, 0]),
                "s_measure": self._s_measure(pred_np[i, 0], target_np[i, 0]),
            }
            for i in range(pred_np.shape[0])
        ]

    @staticmethod
    def _s_measure(pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5) -> float:
        """
        S-measure (Structure measure, Fan et al. 2017).
        Combines region-aware (Sr) and object-aware (So) structural similarities.
        Simplified version: weighted average of object-level and region-level Dice.
        """
        pred_bin = (pred >= 0.5).astype(float)
        gt_bin   = (gt   >= 0.5).astype(float)
        so = dice_score(pred_bin, gt_bin)           # object-level
        # Region-level: split into foreground and background
        if gt_bin.sum() > 0:
            sr_fg = dice_score(pred_bin * gt_bin, gt_bin)
        else:
            sr_fg = 1.0
        bg_pred = 1.0 - pred_bin
        bg_gt   = 1.0 - gt_bin
        sr_bg   = dice_score(bg_pred, bg_gt)
        sr      = 0.5 * sr_fg + 0.5 * sr_bg
        return float(alpha * so + (1 - alpha) * sr)

    def aggregate(self, per_sample: list[dict]) -> dict:
        return {
            "dice_mean":     round(float(np.mean([s["dice"]     for s in per_sample])), 4),
            "iou_mean":      round(float(np.mean([s["iou"]      for s in per_sample])), 4),
            "s_measure_mean":round(float(np.mean([s["s_measure"]for s in per_sample])), 4),
        }
