"""
eval/benchmarks/busi.py  ·  BUSI breast ultrasound segmentation benchmark
================================================================================

BUSI (Breast Ultrasound Images Dataset):
  780 images across 3 classes: benign, malignant, normal.
  Task: tumour segmentation (benign vs malignant) + classification.
  Primary metric: Dice per class.
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


class BUSIBenchmarkDataset(Dataset):
    """
    BUSI loader.
    Layout:
        {root}/benign/
            benign (1).png
            benign (1)_mask.png
        {root}/malignant/
        {root}/normal/
    """

    CLASSES = ("benign", "malignant", "normal")

    def __init__(self, root: str, split: str = "test", test_frac: float = 0.2):
        self.root = Path(root)
        self.samples = self._collect(split, test_frac)

    def _collect(self, split: str, test_frac: float) -> list[dict]:
        samples = []
        for cls_idx, cls_name in enumerate(self.CLASSES):
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            imgs = sorted(f for f in cls_dir.glob("*.png")
                          if "_mask" not in f.name)
            n_test = max(1, int(len(imgs) * test_frac))
            if split == "test":
                subset = imgs[-n_test:]
            else:
                subset = imgs[:-n_test]
            for img_path in subset:
                mask_path = img_path.parent / img_path.name.replace(".png", "_mask.png")
                has_mask  = mask_path.exists() and cls_name != "normal"
                samples.append({
                    "img_path":  str(img_path),
                    "mask_path": str(mask_path) if has_mask else None,
                    "cls_name":  cls_name,
                    "cls_label": cls_idx,
                    "sample_id": f"{cls_name}_{img_path.stem}",
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img_path"]).convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)
        img_t = F.interpolate(img_t.unsqueeze(0), size=(224, 224),
                               mode="bilinear", align_corners=False).squeeze(0)

        mask = torch.zeros(1, 224, 224)
        if s["mask_path"]:
            m   = np.array(Image.open(s["mask_path"]).convert("L"), dtype=np.float32) / 255.0
            m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            m_t = F.interpolate(m_t, size=(224, 224), mode="nearest").squeeze(0)
            mask = (m_t > 0.5).float()

        return {
            "image":     img_t,
            "mask":      mask,
            "cls_label": torch.tensor(s["cls_label"], dtype=torch.long),
            "cls_name":  s["cls_name"],
            "sample_id": s["sample_id"],
        }


class BUSIBenchmark(BaseBenchmark):
    """
    BUSI segmentation + classification benchmark.

    Metrics:
        dice_benign, dice_malignant    per-class segmentation Dice
        dice_mean                      macro-average Dice
        cls_accuracy                   3-class classification accuracy
    """

    BENCHMARK_NAME = "busi_segmentation"
    DATASET_ID     = "BUSI"
    ANATOMY_FAMILY = "breast"
    TASK           = "segmentation"

    def build_dataloader(self, root: str, split: str = "test") -> DataLoader:
        return DataLoader(BUSIBenchmarkDataset(root, split),
                          batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def predict(self, batch: dict) -> dict:
        feats  = self._extract_features(batch["image"])
        logits = self.head(feats["patch_tokens"])
        pred   = F.interpolate(logits, size=(224, 224),
                               mode="bilinear", align_corners=False)
        return {"pred": torch.sigmoid(pred), "cls_feat": feats["cls"]}

    def compute_metrics(self, pred, target, sample_ids) -> list[dict]:
        pred_np   = pred.cpu().float().numpy()
        target_np = target.cpu().float().numpy()
        return [
            {
                "sample_id": sample_ids[i] if i < len(sample_ids) else "",
                "dice":      dice_score(pred_np[i, 0], target_np[i, 0]),
                "iou":       iou_score(pred_np[i, 0], target_np[i, 0]),
            }
            for i in range(pred_np.shape[0])
        ]

    def aggregate(self, per_sample: list[dict]) -> dict:
        dices = [s["dice"] for s in per_sample]
        return {
            "dice_mean": round(float(np.mean(dices)), 4),
            "iou_mean":  round(float(np.mean([s["iou"] for s in per_sample])), 4),
        }
