"""
finetune/experiments/tn3k.py  ·  TN3K thyroid nodule segmentation finetune
====================================================================

Task:    Segment thyroid nodule boundaries from ultrasound images.
Dataset: TN3K — 3,493 images with expert boundary annotations.
Head:    DPTSegHead (default) — boundary-sensitive structures benefit from DPT.
Loss:    BCE + Dice.
Metric:  Dice, IoU, S-measure (structural similarity).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.adapters.thyroid.tn3k_layout import list_tn3k_samples
from finetune.base import FinetuneExperiment, FinetuneConfig
from finetune.seg_common import build_seg_finetune_head, compute_binary_seg_loss
from models.heads import forward_seg_head
from eval.metrics import dice_score, iou_score
from eval.benchmarks.tn3k import TN3KBenchmark

log = logging.getLogger(__name__)
IMG_SIZE = 224


class TN3KFinetuneDataset(Dataset):
    def __init__(self, root: str, split: str = "train", fold: int = 0):
        self.root    = Path(root)
        self.samples = list_tn3k_samples(self.root, split, fold=fold)
        log.info(f"TN3K {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img"]).convert("RGB"), dtype=np.float32) / 255.0
        lbl = np.array(Image.open(s["lbl"]).convert("L"),   dtype=np.float32) / 255.0

        img_t = F.interpolate(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
                               size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                               align_corners=False).squeeze(0)
        lbl_t = F.interpolate(torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0),
                               size=(IMG_SIZE, IMG_SIZE), mode="nearest").squeeze(0)
        return {"image": img_t, "mask": (lbl_t > 0.5).float(),
                "sample_id": s["sample_id"]}


class TN3KFinetune(FinetuneExperiment):
    EXPERIMENT_NAME = "tn3k_thyroid_segmentation"
    DATASET_ID      = "TN3K"
    TASK            = "segmentation"
    BENCHMARK_CLS   = TN3KBenchmark

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        return build_seg_finetune_head(self.encoder, cfg, n_classes=1)

    def build_dataloader(self, split: str) -> DataLoader:
        return DataLoader(TN3KFinetuneDataset(str(self.data_root), split, fold=self.cfg.cv_fold),
                          batch_size=self.cfg.batch_size, shuffle=(split == "train"),
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def compute_loss(self, batch, feats, head_output) -> torch.Tensor:
        return compute_binary_seg_loss(
            head_output, batch["mask"], self.cfg, use_pos_weight=False,
        )

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.encoder.eval()
        per_sample = []
        total_loss = 0.0
        n          = 0

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            feats   = self.encoder.encode_image(batch["image"])
            logits  = forward_seg_head(self.head, feats)
            pred    = F.interpolate(logits, size=batch["mask"].shape[-2:],
                                    mode="bilinear", align_corners=False)
            loss    = self.compute_loss(batch, feats, logits)
            total_loss += loss.item(); n += 1

            pred_b = (torch.sigmoid(pred) > 0.5).cpu().numpy()
            gt_b   = (batch["mask"] > 0.5).cpu().numpy()
            for i in range(len(pred_b)):
                per_sample.append({
                    "dice": dice_score(pred_b[i, 0], gt_b[i, 0]),
                    "iou":  iou_score(pred_b[i, 0], gt_b[i, 0]),
                })

        return {
            "val_loss": round(total_loss / max(n, 1), 4),
            "val_dice": round(float(np.mean([s["dice"] for s in per_sample])), 4),
            "val_iou":  round(float(np.mean([s["iou"]  for s in per_sample])), 4),
        }


if __name__ == "__main__":
    TN3KFinetune.main()
