"""
data/adapters/busi.py  ·  BUSI breast ultrasound adapter
=============================================================

BUSI (Breast Ultrasound Images Dataset): 780 images.
  Classes: benign (437), malignant (210), normal (133)
  Labels:  binary tumour segmentation masks
  Format:  PNG images + PNG masks (suffix _mask.png)

  {root}/benign/benign (1).png
  {root}/benign/benign (1)_mask.png
  {root}/malignant/...
  {root}/normal/...
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class BUSIAdapter(BaseAdapter):
    DATASET_ID     = "BUSI"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1016/j.dib.2019.104863"

    CLASSES = {
        "benign":    ("benign_lesion",    "breast_lesion_benign"),
        "malignant": ("malignant_lesion", "breast_lesion_malignant"),
        "normal":    ("normal",           "breast_normal"),
    }

    def iter_entries(self) -> Iterator[USManifestEntry]:
        all_samples = []
        for cls_name in self.CLASSES:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            imgs = sorted(
                p for p in cls_dir.glob("*.png") if "_mask" not in p.name
            )
            all_samples.extend((img, cls_name) for img in imgs)

        n = len(all_samples)
        for i, (img_path, cls_name) in enumerate(all_samples):
            mask_path = img_path.parent / img_path.name.replace(".png", "_mask.png")
            has_mask  = mask_path.exists() and cls_name != "normal"
            split     = self._infer_split(img_path.stem, i, n)

            label_raw, label_ontology = self.CLASSES[cls_name]
            # Always create a classification instance so that all three classes
            # (benign, malignant, normal) are represented in the manifest.
            instances = [
                self._make_instance(
                    instance_id    = img_path.stem,
                    label_raw      = label_raw,
                    label_ontology = label_ontology,
                    mask_path      = str(mask_path) if has_mask else None,
                    is_promptable  = has_mask,
                )
            ]

            yield self._make_entry(
                str(img_path), split,
                modality      = "image",
                instances     = instances,
                has_mask      = has_mask,
                task_type     = "segmentation" if has_mask else "classification",
                ssl_stream    = "image",
                is_promptable = has_mask,
                source_meta   = {
                    "root":     str(self.root),
                    "doi":      self.DOI,
                    "cls_name": cls_name,
                },
            )
