"""
vlm/grpo/data.py  ·  VLMDataModule
=====================================

Data pipeline for VLM GRPO training.

Reads the existing Ultatron JSONL manifest and converts each entry into a
VLMSample with:
  - A clinical prompt (generated from task_type + anatomy_family)
  - Ground-truth label (extracted from instances)
  - PIL Image (loaded from image_paths[0] or first video frame)
  - Optional pre-extracted Ultatron tokens (if a backbone is pre-run offline)

The prompt templates live in vlm/utils/prompt.py and cover:
  - ef_regression     (EchoNet, MIMIC-LVVol, EchoNet-Pediatric)
  - view_classification (CAMUS, MIMIC-Echo)
  - binary_classification (CardiacUDC, EchoCP, LUS, BUSI)
  - multilabel (LUS patient)
  - segmentation (CAMUS LV, BUSI tumor)
  - weak_label / open-ended (general VQA using anatomy + source_meta)

Manifest entries without a valid ground-truth label are still included as
ssl_only (unsupervised) samples with R_acc = 0.

Stage-aware sampling
--------------------
In Stage 1 (rule-based): only entries with numeric labels (regression/cls) are
sampled; ssl_only entries are skipped.
In Stage 2+: segmentation entries with mask_path are included.

Cross-dataset weighting uses the anatomy_weights from the config (same keys as
the SSL data module).
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

log = logging.getLogger(__name__)


# ── VLMSample ─────────────────────────────────────────────────────────────────

@dataclass
class VLMSample:
    """A single training sample for VLM GRPO."""
    sample_id:      str
    dataset_id:     str
    anatomy_family: str
    task_type:      str
    split:          str
    image_paths:    List[str]
    prompt:         str
    ground_truth:   Any                    # numeric / str / mask_path / None
    image:          Optional[Any] = None   # PIL Image (loaded lazily)
    image_tensor:   Optional[torch.Tensor] = None
    video_tensor:   Optional[torch.Tensor] = None
    patch_tokens:   Optional[torch.Tensor] = None
    tube_tokens:    Optional[torch.Tensor] = None
    meta:           Dict[str, Any] = field(default_factory=dict)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VLMManifestDataset(Dataset):
    """
    PyTorch Dataset wrapping the Ultatron JSONL manifest.

    Parameters
    ----------
    manifest_path   : path to JSONL manifest
    prompt_builder  : PromptBuilder instance from vlm.utils.prompt
    split           : "train" | "val" | "test"
    task_types      : if set, only include these task types
    max_samples     : cap total dataset size (useful for quick sanity runs)
    img_size        : resize PIL images to (img_size, img_size)
    """

    def __init__(
        self,
        manifest_path: str,
        prompt_builder: Any,
        split:         str = "train",
        task_types:    Optional[List[str]] = None,
        max_samples:   Optional[int] = None,
        img_size:      int = 224,
    ):
        self.prompt_builder = prompt_builder
        self.img_size       = img_size

        self.entries = self._load(manifest_path, split, task_types, max_samples)
        log.info(f"VLMManifestDataset: {len(self.entries)} samples "
                 f"(split={split!r}, task_types={task_types!r})")

    @staticmethod
    def _load(path: str, split: str, task_types, max_samples) -> List[dict]:
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get("split") != split:
                    continue
                if task_types and e.get("task_type") not in task_types:
                    continue
                entries.append(e)
                if max_samples and len(entries) >= max_samples:
                    break
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> VLMSample:
        raw = self.entries[idx]

        # Build ground truth from instances
        ground_truth = _extract_ground_truth(raw)

        # Build prompt
        prompt = self.prompt_builder.build(
            task_type      = raw.get("task_type", "ssl_only"),
            anatomy_family = raw.get("anatomy_family", "other"),
            source_meta    = raw.get("source_meta", {}),
            instances      = raw.get("instances", []),
            view_type      = raw.get("view_type"),
        )

        # Load image (first path only; video = first frame)
        image = _load_image(raw.get("image_paths", []), self.img_size)

        return VLMSample(
            sample_id      = raw["sample_id"],
            dataset_id     = raw["dataset_id"],
            anatomy_family = raw.get("anatomy_family", "other"),
            task_type      = raw.get("task_type", "ssl_only"),
            split          = raw.get("split", "train"),
            image_paths    = raw.get("image_paths", []),
            prompt         = prompt,
            ground_truth   = ground_truth,
            image          = image,
            meta           = raw.get("source_meta", {}),
        )


# ── VLMDataModule ─────────────────────────────────────────────────────────────

class VLMDataModule:
    """
    DataModule that provides a DataLoader of VLMSamples.

    Parameters
    ----------
    manifest_path    : path to run1_train.jsonl
    batch_size       : number of samples per step (each spawns G rollouts)
    num_workers      : DataLoader worker count
    split            : "train" | "val"
    img_size         : image resize
    anatomy_weights  : per-anatomy sampling weights (same schema as SSL config)
    task_types       : subset of task types to include
    max_samples      : cap dataset size
    """

    def __init__(
        self,
        manifest_path:  str,
        batch_size:     int = 4,
        num_workers:    int = 4,
        split:          str = "train",
        img_size:       int = 224,
        anatomy_weights: Optional[Dict[str, float]] = None,
        task_types:     Optional[List[str]] = None,
        max_samples:    Optional[int] = None,
    ):
        from vlm.utils.prompt import PromptBuilder
        self.prompt_builder  = PromptBuilder()
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.anatomy_weights = anatomy_weights or {}

        self.dataset = VLMManifestDataset(
            manifest_path  = manifest_path,
            prompt_builder = self.prompt_builder,
            split          = split,
            task_types     = task_types,
            max_samples    = max_samples,
            img_size       = img_size,
        )

    def loader(self) -> DataLoader:
        """Return a DataLoader (infinite via sampler)."""
        sampler = self._build_sampler()
        return DataLoader(
            self.dataset,
            batch_size  = self.batch_size,
            sampler     = sampler,
            num_workers = self.num_workers,
            collate_fn  = _collate_fn,
            pin_memory  = True,
            drop_last   = True,
            persistent_workers = self.num_workers > 0,
        )

    def _build_sampler(self) -> WeightedRandomSampler:
        """Weight samples by anatomy family (mirrors SSL data module)."""
        weights = []
        for e in self.dataset.entries:
            anatomy = e.get("anatomy_family", "other")
            w = self.anatomy_weights.get(anatomy, 1.0)
            weights.append(w)
        return WeightedRandomSampler(
            weights     = weights,
            num_samples = len(weights),
            replacement = True,
        )

    # ── Stage-aware task type filtering ──────────────────────────────────────

    @staticmethod
    def for_stage(
        manifest_path: str,
        stage:         int,
        **kwargs,
    ) -> "VLMDataModule":
        """
        Convenience factory that sets task_types based on training stage.

        Stage 1: regression + classification (no segmentation/open-ended)
        Stage 2: add segmentation (mask_path required)
        Stage 3: all task types
        """
        if stage == 1:
            task_types = ["regression", "measurement", "classification", "sequence"]
        elif stage == 2:
            task_types = ["regression", "measurement", "classification", "sequence",
                          "segmentation"]
        else:
            task_types = None  # all types
        return VLMDataModule(manifest_path=manifest_path, task_types=task_types, **kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_ground_truth(raw: dict) -> Any:
    """
    Extract the primary ground-truth label from a manifest entry.

    Priority: measurement_mm (regression) > classification_label > label_raw
    """
    instances = raw.get("instances", [])
    task_type = raw.get("task_type", "ssl_only")

    if task_type in ("regression", "measurement"):
        # Return the first measurement value
        for inst in instances:
            if inst.get("measurement_mm") is not None:
                return float(inst["measurement_mm"])
        # Try source_meta for EF
        meta = raw.get("source_meta", {})
        for key in ("ef", "ejection_fraction", "EF", "ef_percent"):
            if key in meta:
                return float(meta[key])

    if task_type in ("classification", "sequence", "binary_classification"):
        for inst in instances:
            if inst.get("classification_label") is not None:
                return inst["classification_label"]
        if raw.get("label_ids_raw"):
            return raw["label_ids_raw"][0]
        if raw.get("label_raw"):
            return raw["label_raw"][0]

    if task_type == "segmentation":
        for inst in instances:
            if inst.get("mask_path"):
                return {"mask_path": inst["mask_path"]}

    # Fallback: return raw label string if available
    if raw.get("label_raw"):
        return raw["label_raw"][0]

    return None


def _load_image(image_paths: List[str], img_size: int = 224) -> Optional[Any]:
    """Load first valid image path as PIL Image, resized to img_size."""
    if not image_paths:
        return None
    try:
        from PIL import Image
        p = Path(image_paths[0])
        if not p.exists():
            return None
        img = Image.open(p).convert("RGB")
        img = img.resize((img_size, img_size))
        return img
    except Exception as e:
        log.debug(f"Image load failed ({image_paths[0]}): {e}")
        return None


def _collate_fn(batch: List[VLMSample]) -> List[VLMSample]:
    """
    Simple collate: return list (not tensor stack) since samples have
    variable-length trajectories and PIL images.
    The GRPO trainer iterates this list directly.
    """
    return batch
