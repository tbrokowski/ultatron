"""
finetune/experiments/busbra.py  ·  BUS-BRA breast lesion segmentation finetune
================================================================================

Task:    Segment breast tumour boundaries from ultrasound images.
Dataset: BUS-BRA — 1,875 images with expert lesion masks.
Protocol: OpenUS split (1200/299/376); training follows BUSI head recipe
          (UPerNet + refine_up + augmentations) at native-friendly 500×500.
Metrics: Dice (DSC), IoU.
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

from data.adapters.breast.bus_bra_layout import list_bus_bra_samples
from finetune.base import FinetuneExperiment, FinetuneConfig
from finetune.datasets.seg_augment import augment_breast_us
from finetune.datasets.native_batch import finetune_img_size
from finetune.seg_common import build_seg_finetune_head, compute_binary_seg_loss
from models.heads import forward_seg_head
from eval.metrics import dice_score, iou_score
from eval.benchmarks.busbra import BUSBRABenchmark

log = logging.getLogger(__name__)


class BUSBRAFinetuneDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        fold: int = 0,
        split_manifest: str = "",
        img_size: int = 500,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.augment = augment and split == "train"
        self.img_size = img_size
        self.samples = list_bus_bra_samples(
            self.root, split, fold=fold,
            split_manifest=split_manifest or None,
        )
        log.info("BUS-BRA %s: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img"]).convert("RGB"), dtype=np.float32) / 255.0
        lbl = np.array(Image.open(s["lbl"]).convert("L"),   dtype=np.float32) / 255.0

        img_t = F.interpolate(
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        lbl_t = F.interpolate(
            torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode="nearest",
        ).squeeze(0)
        mask = (lbl_t > 0.5).float()

        if self.augment:
            img_t, mask = augment_breast_us(img_t, mask)

        return {
            "image":     img_t,
            "mask":      mask,
            "sample_id": s["sample_id"],
        }


class BUSBRAFinetune(FinetuneExperiment):
    EXPERIMENT_NAME = "busbra_breast_segmentation"
    DATASET_ID      = "BUS-BRA"
    TASK            = "segmentation"
    BENCHMARK_CLS   = BUSBRABenchmark

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        return build_seg_finetune_head(self.encoder, cfg, n_classes=1)

    def build_dataloader(self, split: str) -> DataLoader:
        sz = finetune_img_size(self.cfg)
        return DataLoader(
            BUSBRAFinetuneDataset(
                str(self.data_root), split,
                fold=self.cfg.cv_fold,
                split_manifest=self.cfg.split_manifest,
                img_size=sz,
                augment=(split == "train"),
            ),
            batch_size=self.cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

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
            loss    = self.compute_loss(batch, feats, logits)
            total_loss += loss.item()
            n += 1

            pred = F.interpolate(logits, size=batch["mask"].shape[-2:],
                                 mode="bilinear", align_corners=False)
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

    def evaluate(self, split: str = "test") -> dict:
        """Evaluate with OpenUS split manifest and matched train resolution."""
        assert self.head is not None, "Call setup() and run() first"
        self.head.eval()

        benchmark = self.BENCHMARK_CLS(
            encoder=self.encoder,
            img_branch=self.img_branch,
            head=self.head,
            device=self.device,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            split_manifest=self.cfg.split_manifest or None,
            cv_fold=self.cfg.cv_fold,
            img_size=finetune_img_size(self.cfg),
        )
        results = benchmark.run(str(self.data_root), split=split)
        results["experiment"] = self.EXPERIMENT_NAME
        if self.cfg.split_manifest:
            results["split_manifest"] = self.cfg.split_manifest
        self._save_results(results)
        self.run_viz(results, self.output_dir)
        return results


if __name__ == "__main__":
    BUSBRAFinetune.main()
