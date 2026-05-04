"""
data/adapters/breast/s1_adapter.py  ·  S1 breast ultrasound adapter
====================================================================
S1 (Breast Tumour Dataset): 201 images / 204 masks.
  Labels:  binary tumour segmentation masks (no benign/malignant distinction)
  Source:  US-43d
  SonoDQS: silver

Dataset layout on disk
-----------------------
S1/
├── TrainingDataSet/
│   ├── BreastTumourImages/         ← .jpg images  (0.jpg, 1.jpg, ...)
│   ├── Expanded-3-channel-Labels/  ← 3-channel PNG masks (ignored)
│   └── General-1-channel-Labels/   ← binary PNG masks  ← used
├── TestingDataSet/
│   ├── Test-Expanded-BreastTumourImages/   ← .jpg images
│   ├── Test-Expanded-3-channel-Labels/     ← 3-channel PNG masks (ignored)
│   ├── Test-General-1-BreastTumourImages/  ← .jpg images
│   └── Test-General-1-channel-Labels/      ← binary PNG masks  ← used
└── Segmentation_Results/   ← model outputs, ignored

Key observations
----------------
- No benign/malignant label — segmentation only (breast_lesion).
- Split is defined by folder: TrainingDataSet → train, TestingDataSet → test.
- Pairing by stem number: BreastTumourImages/0.jpg ↔ General-1-channel-Labels/0.png
- Two mask types: we use General-1-channel-Labels (binary) as primary mask.
- TestingDataSet has two image subfolders (Expanded + General); we deduplicate
  by loading from Test-General-1-BreastTumourImages only (or Expanded if General
  is absent).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class S1Adapter(BaseAdapter):
    """
    Adapter for the S1 breast tumour ultrasound dataset.

    Parameters
    ----------
    root : str | Path
        Root directory containing TrainingDataSet/ and TestingDataSet/.
    split_override : str, optional
        If set, all entries get this split label.
    """

    DATASET_ID     = "S1"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = ""   # no DOI found — US-43d sourced

    def iter_entries(self) -> Iterator[USManifestEntry]:
        """Yield one USManifestEntry per image across train and test splits."""

        all_samples = []

        # ── Training split ────────────────────────────────────────────────
        train_dir  = self.root / "TrainingDataSet"
        train_imgs = train_dir / "BreastTumourImages"
        train_masks = train_dir / "General-1-channel-Labels"

        if train_imgs.exists():
            for img_path in sorted(train_imgs.glob("*.jpg")):
                mask_path = train_masks / f"{img_path.stem}.png"
                all_samples.append((img_path, mask_path, "train"))

        # ── Testing split ─────────────────────────────────────────────────
        test_dir = self.root / "TestingDataSet"

        # Prefer Test-General-1 images; fall back to Test-Expanded
        test_imgs = test_dir / "Test-General-1-BreastTumourImages"
        test_masks = test_dir / "Test-General-1-channel-Labels"

        if not test_imgs.exists():
            test_imgs  = test_dir / "Test-Expanded-BreastTumourImages"
            test_masks = test_dir / "Test-Expanded-3-channel-Labels"

        if test_imgs.exists():
            for img_path in sorted(test_imgs.glob("*.jpg")):
                mask_path = test_masks / f"{img_path.stem}.png"
                all_samples.append((img_path, mask_path, "test"))

        # ── Yield entries ─────────────────────────────────────────────────
        for img_path, mask_path, split in all_samples:
            if self.split_override:
                split = self.split_override

            has_mask = mask_path.exists()

            instance = self._make_instance(
                instance_id    = img_path.stem,
                label_raw      = "tumor",
                label_ontology = "breast_lesion",
                mask_path      = str(mask_path) if has_mask else None,
                is_promptable  = has_mask,
            )

            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = [instance],
                has_mask      = has_mask,
                task_type     = "segmentation" if has_mask else "ssl_only",
                ssl_stream    = "image",
                is_promptable = has_mask,
                probe_type    = "linear",
                source_meta   = {
                    "original_split": split,
                    "doi":            self.DOI,
                },
            )
