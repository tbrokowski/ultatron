"""
data/adapters/tn3k.py  ·  TN3K thyroid nodule adapter
===========================================================

TN3K: 3,493 thyroid ultrasound images with expert nodule segmentation.
  Format: JPG images + PNG masks
  Layout:
    {root}/image/*.jpg
    {root}/label/*.png  (binary mask, same stem as image)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class TN3KAdapter(BaseAdapter):
    DATASET_ID     = "TN3K"
    ANATOMY_FAMILY = "thyroid"
    SONODQS        = "silver"
    DOI            = "https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self.root / "image"
        lbl_dir = self.root / "label"
        imgs    = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        n       = len(imgs)

        for i, img_path in enumerate(imgs):
            lbl_path = lbl_dir / (img_path.stem + ".png")
            has_mask = lbl_path.exists()
            split    = self._infer_split(img_path.stem, i, n)

            instances = []
            if has_mask:
                instances.append(self._make_instance(
                    instance_id    = img_path.stem,
                    label_raw      = "thyroid_nodule",
                    label_ontology = "thyroid_nodule_boundary",
                    mask_path      = str(lbl_path),
                    is_promptable  = True,
                ))

            yield self._make_entry(
                str(img_path), split,
                modality      = "image",
                instances     = instances,
                has_mask      = has_mask,
                task_type     = "segmentation" if has_mask else "ssl_only",
                ssl_stream    = "image",
                is_promptable = has_mask,
            )
