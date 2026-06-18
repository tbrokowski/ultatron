"""
eval/benchmarks/busbra.py  ·  BUS-BRA breast lesion segmentation benchmark
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.adapters.breast.bus_bra_layout import list_bus_bra_samples
from eval.benchmarks.base import BaseBenchmark
from eval.metrics import dice_score, iou_score

log = logging.getLogger(__name__)


class BUSBRABenchmarkDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "test",
        fold: int = 0,
        split_manifest: str | Path | None = None,
        img_size: int = 500,
    ):
        self.root     = Path(root)
        self.img_size = img_size
        self.samples  = list_bus_bra_samples(
            self.root, split, fold=fold,
            split_manifest=split_manifest,
        )
        log.info("BUS-BRA %s: %d samples", split, len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img_path"]).convert("RGB"), dtype=np.float32) / 255.0
        lbl = np.array(Image.open(s["lbl_path"]).convert("L"),   dtype=np.float32) / 255.0

        size = (self.img_size, self.img_size)
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        img_t = F.interpolate(
            img_t.unsqueeze(0), size=size,
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        lbl_t = torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0)
        lbl_t = F.interpolate(lbl_t, size=size, mode="nearest").squeeze(0)
        mask  = (lbl_t > 0.5).float()

        return {"image": img_t, "mask": mask, "sample_id": s["sample_id"]}


class BUSBRABenchmark(BaseBenchmark):
    BENCHMARK_NAME = "busbra_segmentation"
    DATASET_ID     = "BUS-BRA"
    ANATOMY_FAMILY = "breast"
    TASK           = "segmentation"

    def __init__(
        self,
        img_branch=None,
        head: Optional[torch.nn.Module] = None,
        device: str = "cuda",
        batch_size: int = 16,
        num_workers: int = 4,
        encoder=None,
        split_manifest: str | Path | None = None,
        cv_fold: int = 0,
        img_size: int = 500,
        **kwargs,
    ):
        super().__init__(
            img_branch=img_branch,
            head=head,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            encoder=encoder,
            **kwargs,
        )
        self.split_manifest = split_manifest
        self.cv_fold        = cv_fold
        self.img_size       = img_size

    def build_dataloader(self, root: str, split: str = "test") -> DataLoader:
        return DataLoader(
            BUSBRABenchmarkDataset(
                root, split,
                fold=self.cv_fold,
                split_manifest=self.split_manifest,
                img_size=self.img_size,
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict(self, batch: dict) -> dict:
        feats  = self._extract_features(batch["image"])
        logits = self._predict_seg_logits(feats)
        size   = (self.img_size, self.img_size)
        pred   = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return {"pred": torch.sigmoid(pred)}

    def compute_metrics(self, pred, target, sample_ids) -> list[dict]:
        pred_np   = pred.cpu().float().numpy()
        target_np = target.cpu().float().numpy()
        return [
            {
                "sample_id": sample_ids[i] if i < len(sample_ids) else "",
                "dice": dice_score(pred_np[i, 0], target_np[i, 0]),
                "iou":  iou_score(pred_np[i, 0], target_np[i, 0]),
            }
            for i in range(pred_np.shape[0])
        ]

    def aggregate(self, per_sample: list[dict]) -> dict:
        return {
            "dice_mean": round(float(np.mean([s["dice"] for s in per_sample])), 4),
            "iou_mean":  round(float(np.mean([s["iou"]  for s in per_sample])), 4),
        }
