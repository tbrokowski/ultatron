"""
eval/benchmarks/busi.py  ·  BUSI breast ultrasound segmentation benchmark
================================================================================

BUSI (Breast Ultrasound Images Dataset):
  780 images across 3 classes: benign, malignant, normal.
  Task: tumour segmentation (benign vs malignant) + classification.
  Primary metric: dice_tumor_mean (benign + malignant only).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from eval.benchmarks.base import BaseBenchmark
from eval.metrics import dice_score, iou_score, pixel_accuracy, pixel_precision, pixel_recall
from finetune.datasets.busi_splits import (
    collect_busi_samples,
    default_repo_manifest_path,
    load_split_manifest,
    split_busi_samples,
)
from finetune.datasets.native_batch import (
    extract_native_region,
    finetune_native_collate,
    native_crop_shape,
    padding_mask_for_shape,
)

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

    def __init__(
        self,
        root: str,
        split: str = "test",
        test_frac: float = 0.2,
        val_frac: float = 0.2,
        split_manifest: str | Path | None = None,
        img_size: int = 224,
        native: bool = False,
        native_max_px: int = 512,
    ):
        self.root = Path(root)
        self.img_size = img_size
        self.native = native
        self.native_max_px = native_max_px
        manifest = load_split_manifest(split_manifest) if split_manifest else None
        all_samples = collect_busi_samples(self.root)
        self.samples = split_busi_samples(
            all_samples,
            split,
            test_frac=test_frac,
            val_frac=val_frac,
            manifest=manifest,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img"]).convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)   # (3, H, W)

        if self.native:
            _, h0, w0 = img_t.shape
            th, tw = native_crop_shape(h0, w0, max_px=self.native_max_px)
            img_t = extract_native_region(img_t, th, tw)
            if s["mask"]:
                m = np.array(Image.open(s["mask"]).convert("L"), dtype=np.float32) / 255.0
                m_t = torch.from_numpy(m).unsqueeze(0)
                mask = (extract_native_region(m_t, th, tw) > 0.5).float()
            else:
                mask = torch.zeros(1, th, tw)
            return {
                "image": img_t,
                "mask": mask,
                "padding_mask": padding_mask_for_shape(th, tw, valid_h=th, valid_w=tw),
                "cls_label": torch.tensor(s["cls_label"], dtype=torch.long),
                "cls_name": s["cls_name"],
                "sample_id": s["sample_id"],
            }

        img_t = F.interpolate(img_t.unsqueeze(0), size=(self.img_size, self.img_size),
                               mode="bilinear", align_corners=False).squeeze(0)

        mask = torch.zeros(1, self.img_size, self.img_size)
        if s["mask"]:
            m   = np.array(Image.open(s["mask"]).convert("L"), dtype=np.float32) / 255.0
            m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            m_t = F.interpolate(m_t, size=(self.img_size, self.img_size), mode="nearest").squeeze(0)
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
        dice_tumor_mean                macro-average Dice on tumour images
        dice_mean                      all images (includes normal zeros)
        iou_mean                       macro-average IoU
        acc_mean                       pixel-level accuracy
        precision_mean                 pixel-level precision
        recall_mean                    pixel-level recall
    """

    BENCHMARK_NAME = "busi_segmentation"
    DATASET_ID     = "BUSI"
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
        eval_tumor_only: bool = True,
        img_size: int = 224,
        native: bool = False,
        native_max_px: int = 512,
        head2: Optional[torch.nn.Module] = None,
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
        if split_manifest is None:
            default = default_repo_manifest_path()
            split_manifest = default if default.is_file() else None
        self.split_manifest = split_manifest
        self.eval_tumor_only = eval_tumor_only
        self.img_size = img_size
        self.native = native
        self.native_max_px = native_max_px
        self.head2 = head2

    def build_dataloader(self, root: str, split: str = "test") -> DataLoader:
        dataset = BUSIBenchmarkDataset(
            root,
            split,
            split_manifest=self.split_manifest,
            img_size=self.img_size,
            native=self.native,
            native_max_px=self.native_max_px,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=finetune_native_collate if self.native else None,
        )

    def predict(self, batch: dict) -> dict:
        pmask = batch.get("padding_mask")
        feats  = self._extract_features(batch["image"], padding_mask=pmask)
        logits = self._predict_seg_logits(feats, padding_mask=pmask)
        pred   = F.interpolate(
            logits, size=batch["mask"].shape[-2:],
            mode="bilinear", align_corners=False,
        )
        out = {"pred": torch.sigmoid(pred)}
        if self.head2 is not None:
            out["cls_logits"] = self.head2(feats["cls"])
        return out

    def compute_metrics(self, pred, target, sample_ids, *, cls_logits=None, cls_labels=None):
        pred_np   = pred.cpu().float().numpy()
        target_np = target.cpu().float().numpy()
        rows = []
        cls_pred_np = None
        if cls_logits is not None:
            cls_pred_np = cls_logits.argmax(dim=-1).cpu().numpy()
        cls_label_np = cls_labels.cpu().numpy() if cls_labels is not None else None

        for i, sid in enumerate(sample_ids):
            row = {
                "sample_id": sid,
                "cls_name":  sid.split("_", 1)[0] if "_" in sid else "?",
                "dice":      dice_score(pred_np[i, 0], target_np[i, 0]),
                "iou":       iou_score(pred_np[i, 0], target_np[i, 0]),
                "acc":       pixel_accuracy(pred_np[i, 0], target_np[i, 0]),
                "precision": pixel_precision(pred_np[i, 0], target_np[i, 0]),
                "recall":    pixel_recall(pred_np[i, 0], target_np[i, 0]),
            }
            if cls_pred_np is not None and cls_label_np is not None:
                row["cls_correct"] = int(cls_pred_np[i] == cls_label_np[i])
            rows.append(row)
        return rows

    def run(self, root: str, split: str = "test") -> dict:
        """Run seg metrics; add cls_accuracy when head2 is present."""
        if self.head2 is None:
            return super().run(root, split=split)

        import time
        t0 = time.time()
        log.info(f"[{self.BENCHMARK_NAME}] Running multitask on split={split!r} ...")

        loader = self.build_dataloader(root, split)
        per_sample = []

        if self.encoder is not None:
            self.encoder.eval()
        self.head.eval()
        self.head2.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self.predict(batch)
                metrics = self.compute_metrics(
                    outputs["pred"],
                    batch["mask"],
                    batch.get("sample_id", [""] * outputs["pred"].shape[0]),
                    cls_logits=outputs.get("cls_logits"),
                    cls_labels=batch.get("cls_label"),
                )
                per_sample.extend(metrics)

        results = self.aggregate(per_sample)
        cls_hits = [s["cls_correct"] for s in per_sample if "cls_correct" in s]
        if cls_hits:
            results["cls_accuracy"] = round(float(np.mean(cls_hits)), 4)
        results.update({
            "benchmark":   self.BENCHMARK_NAME,
            "dataset_id":  self.DATASET_ID,
            "task":        self.TASK,
            "split":       split,
            "n_samples":   len(per_sample),
            "elapsed_sec": round(time.time() - t0, 1),
        })
        return results

    def aggregate(self, per_sample: list[dict]) -> dict:
        tumor = [s for s in per_sample if s.get("cls_name") in ("benign", "malignant")]
        scored = tumor if self.eval_tumor_only else per_sample
        return {
            "dice_mean":       round(float(np.mean([s["dice"] for s in scored])), 4) if scored else float("nan"),
            "dice_tumor_mean": round(float(np.mean([s["dice"] for s in tumor])), 4) if tumor else float("nan"),
            "iou_mean":        round(float(np.mean([s["iou"]  for s in scored])), 4) if scored else float("nan"),
            "iou_tumor_mean":  round(float(np.mean([s["iou"]  for s in tumor])), 4) if tumor else float("nan"),
            "acc_mean":        round(float(np.mean([s["acc"]  for s in scored])), 4) if scored else float("nan"),
            "precision_mean":  round(float(np.mean([s["precision"] for s in scored])), 4) if scored else float("nan"),
            "recall_mean":     round(float(np.mean([s["recall"]    for s in scored])), 4) if scored else float("nan"),
            "eval_tumor_only": self.eval_tumor_only,
        }
