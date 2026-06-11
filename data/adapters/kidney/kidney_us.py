"""
data/adapters/kidney/kidney_us.py  - Open Kidney Ultrasound Data Set adapter
=============================================================================

Layout (images only, as staged on CSCS store):
  {root}/kidneyUS_images_*/{patient_id}_IM-{study}-{frame}_anon.png

Optional labels (from https://github.com/rsingla92/kidneyUS):
  {root}/labels/reviewed_masks_{1,2}/capsule/{image_name}.png
  {root}/labels/reviewed_labels_{1,2}.csv
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)


class KidneyUSAdapter(BaseAdapter):
    DATASET_ID     = "KidneyUS"
    ANATOMY_FAMILY = "kidney"
    SONODQS        = "gold"
    DOI            = "https://rsingla92.github.io/kidneyUS/"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._images_dir = self._find_images_dir(self.root)
        self._mask_dirs = self._resolve_mask_dirs()
        self._metadata = self._load_csv_metadata()

    @staticmethod
    def _find_images_dir(root: Path) -> Optional[Path]:
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and sub.name.startswith("kidneyUS_images"):
                return sub
        if any(root.glob("*.png")):
            return root
        return None

    def _resolve_mask_dirs(self) -> list[Path]:
        labels_root = self.root / "labels"
        if not labels_root.exists():
            return []
        return sorted(
            d for d in labels_root.glob("reviewed_masks_*/capsule") if d.is_dir()
        )

    def _load_csv_metadata(self) -> Dict[str, dict]:
        """Parse VIA-style CSVs for per-image quality/view metadata."""
        meta: Dict[str, dict] = {}
        labels_root = self.root / "labels"
        if not labels_root.exists():
            return meta

        for csv_path in sorted(labels_root.glob("reviewed_labels_*.csv")):
            with csv_path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename", "")
                    if not fname or fname in meta:
                        continue
                    attrs = {}
                    raw_attrs = row.get("file_attributes", "")
                    if raw_attrs:
                        try:
                            attrs = json.loads(raw_attrs)
                        except json.JSONDecodeError:
                            pass
                    meta[fname] = {
                        "quality": attrs.get("Quality"),
                        "view": attrs.get("View"),
                        "comments": attrs.get("Comments"),
                    }
        return meta

    def _find_capsule_mask(self, image_name: str) -> Optional[Path]:
        for mask_dir in self._mask_dirs:
            mask_path = mask_dir / image_name
            if mask_path.exists():
                return mask_path
        return None

    @staticmethod
    def _patient_id(stem: str) -> str:
        return stem.split("_IM-", 1)[0]

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if self._images_dir is None:
            log.warning("KidneyUS: no image directory found under %s", self.root)
            return

        images = sorted(self._images_dir.glob("*.png"))
        if not images:
            log.warning("KidneyUS: no PNG images found under %s", self._images_dir)
            return

        if not self._mask_dirs:
            log.info(
                "KidneyUS: %d images found; no capsule masks under labels/ — "
                "emitting ssl_only entries",
                len(images),
            )

        patient_splits = self._group_split_map(
            self._patient_id(img.stem) for img in images
        )

        for img_path in images:
            stem = img_path.stem
            patient_id = self._patient_id(stem)
            split = self.split_override or patient_splits[patient_id]
            mask_path = self._find_capsule_mask(img_path.name)
            has_mask = mask_path is not None
            row_meta = self._metadata.get(img_path.name, {})

            if has_mask:
                instances = [
                    self._make_instance(
                        instance_id=f"{stem}_capsule",
                        label_raw="kidney_capsule",
                        label_ontology="kidney",
                        mask_path=str(mask_path),
                        is_promptable=True,
                    )
                ]
                task_type = "segmentation"
            else:
                instances = [
                    self._make_instance(
                        instance_id=stem,
                        label_raw="kidney",
                        label_ontology="kidney",
                        is_promptable=False,
                    )
                ]
                task_type = "ssl_only"

            view = row_meta.get("view")
            view_type = view.lower().replace(" ", "_") if view else None

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                study_id=patient_id,
                view_type=view_type,
                has_mask=has_mask,
                task_type=task_type,
                ssl_stream="image",
                is_promptable=has_mask,
                source_meta={
                    "patient_id": patient_id,
                    "quality": row_meta.get("quality"),
                    "view": row_meta.get("view"),
                    "comments": row_meta.get("comments"),
                },
            )
