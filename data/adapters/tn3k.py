"""
data/adapters/tn3k.py  ·  TN3K thyroid nodule adapter
===========================================================

TN3K: 3,493 thyroid ultrasound images with expert nodule segmentation.
  Format: JPG images + PNG masks
  Layout on disk (store):
    {root}/trainval-image/*.jpg   +  {root}/trainval-mask/*.png
    {root}/test-image/*.jpg       +  {root}/test-mask/*.png

  Legacy / flat layout also supported:
    {root}/image/*.jpg            +  {root}/label/*.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class TN3KAdapter(BaseAdapter):
    DATASET_ID     = "TN3K"
    ANATOMY_FAMILY = "thyroid"
    SONODQS        = "silver"
    DOI            = "https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation"

    def _split_dirs(self) -> List[Tuple[Path, Path, str]]:
        """
        Return a list of (image_dir, mask_dir, split_name) tuples.

        Supports both the store layout (trainval-image / test-image)
        and the legacy flat layout (image / label).
        """
        pairs: List[Tuple[Path, Path, str]] = []

        # Store layout: {prefix}-image / {prefix}-mask
        for img_dir in sorted(self.root.glob("*-image")):
            prefix   = img_dir.name[: -len("-image")]   # e.g. "trainval", "test"
            mask_dir = self.root / f"{prefix}-mask"
            split    = "test" if "test" in prefix else "train"
            pairs.append((img_dir, mask_dir, split))

        # Legacy flat layout fallback
        if not pairs:
            img_dir  = self.root / "image"
            mask_dir = self.root / "label"
            if img_dir.exists():
                pairs.append((img_dir, mask_dir, "train"))

        return pairs

    def iter_entries(self) -> Iterator[USManifestEntry]:
        split_dirs = self._split_dirs()

        # Collect all images first so total n is correct for split inference
        all_images: List[Tuple[Path, Path, str]] = []
        for img_dir, mask_dir, split_name in split_dirs:
            for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
                all_images.append((img_path, mask_dir, split_name))
        n = len(all_images)

        for i, (img_path, mask_dir, split_name) in enumerate(all_images):
            lbl_path = mask_dir / (img_path.stem + ".png")
            has_mask = lbl_path.exists()
            # Honour explicit split from directory name; fall back to ratio-based
            split    = split_name if split_name else self._infer_split(img_path.stem, i, n)

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
