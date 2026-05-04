"""
eval/benchmarks/base.py  ·  Shared benchmark infrastructure
=================================================================

BaseBenchmark
-------------
Abstract class all benchmark runners inherit from.

Provides:
  - Standard DataLoader construction from a manifest split
  - Batch inference loop with gradient disabled
  - Results dict construction and JSON output
  - Consistent device handling

Every benchmark subclass implements:
  _run_batch(batch, model_outputs) → dict of per-sample metrics
  _aggregate(per_sample_results)   → summary dict

The trainer integration (called at end of Phase 3 / end of training):
    benchmark = CAMUSBenchmark(img_branch, seg_head, dm, device)
    results   = benchmark.run(split="test")
    benchmark.save(results, "results/camus_test.json", step=global_step)
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """
    Abstract benchmark runner.

    Parameters
    ----------
    img_branch : ImageBranch  (teacher used for inference)
    head       : nn.Module or None  (task-specific head)
    device     : str
    batch_size : int
    num_workers: int
    """

    # Subclasses set these
    BENCHMARK_NAME: str = "base"
    DATASET_ID:     str = "unknown"
    ANATOMY_FAMILY: str = "other"
    TASK:           str = "segmentation"   # "segmentation" | "regression" | "classification"

    def __init__(
        self,
        img_branch,
        head: Optional[nn.Module],
        device: str = "cuda",
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        self.img_branch  = img_branch
        self.head        = head
        self.device      = device
        self.batch_size  = batch_size
        self.num_workers = num_workers

    # ── Abstract interface ─────────────────────────────────────────────────────

    @abstractmethod
    def build_dataloader(self, root: str, split: str = "test") -> DataLoader:
        """
        Build a DataLoader for this dataset.

        Must return batches with at minimum:
          "image"      : (B, 3, H, W) float32 [0, 1]
          "sample_id"  : list of str
        Plus task-specific keys:
          For segmentation: "mask" (B, 1, H, W) float32 binary
          For regression:   "target" (B,) float32
          For classification: "label" (B,) int64
        """
        ...

    @abstractmethod
    def predict(self, batch: dict) -> dict:
        """
        Run model inference on one batch.
        Returns dict with "pred" key (task-specific tensor).
        """
        ...

    @abstractmethod
    def compute_metrics(self, pred: Any, target: Any, sample_ids: list) -> list[dict]:
        """
        Compute per-sample metrics.
        Returns list of dicts, one per sample.
        """
        ...

    @abstractmethod
    def aggregate(self, per_sample: list[dict]) -> dict:
        """Aggregate per-sample metrics to summary dict."""
        ...

    # ── Shared run loop ───────────────────────────────────────────────────────

    def run(self, root: str, split: str = "test") -> dict:
        """
        Full benchmark run.

        Returns a results dict that includes:
          - All task-specific metrics
          - "n_samples": int
          - "elapsed_sec": float
          - "benchmark": str
          - "split": str
        """
        t0 = time.time()
        log.info(f"[{self.BENCHMARK_NAME}] Running on split={split!r} ...")

        loader       = self.build_dataloader(root, split)
        per_sample   = []

        if self.img_branch is not None:
            self.img_branch.teacher.eval()
        if getattr(self, "vid_branch", None) is not None:
            self.vid_branch.teacher.eval()
        if self.head is not None:
            self.head.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs     = self.predict(batch)
                pred        = outputs["pred"]
                target      = next((batch[k] for k in ("mask", "target", "label")
                                    if batch.get(k) is not None), None)
                sample_ids  = batch.get("sample_id", [""] * pred.shape[0])
                metrics     = self.compute_metrics(pred, target, sample_ids)
                per_sample.extend(metrics)

        results = self.aggregate(per_sample)
        results.update({
            "benchmark":   self.BENCHMARK_NAME,
            "dataset_id":  self.DATASET_ID,
            "task":        self.TASK,
            "split":       split,
            "n_samples":   len(per_sample),
            "elapsed_sec": round(time.time() - t0, 1),
        })

        log.info(f"[{self.BENCHMARK_NAME}] Done. "
                 + "  ".join(f"{k}={v:.4f}" for k, v in results.items()
                              if isinstance(v, float) and k != "elapsed_sec"))
        return results

    @staticmethod
    def save(results: dict, output_path: str | Path, step: int = 0):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"step": step, **results}, indent=2))
        log.info(f"Benchmark results → {path}")

    # ── Shared backbone inference ─────────────────────────────────────────────

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor, padding_mask=None) -> dict:
        """Run image teacher and return {cls, patch_tokens}."""
        return self.img_branch.forward_teacher(images, padding_mask=padding_mask)
