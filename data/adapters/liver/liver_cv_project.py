"""
data/adapters/liver/liver_cv_project.py  ·  Liver CV Project adapter
=====================================================================

Liver Ultrasound CV Project — Klepich, Roboflow 2023 (CC BY 4.0).
  ~220 training JPG images (299×299) in COCO detection format.
  10 anatomical / view categories; bounding-box annotations only
  (segmentation field is empty throughout).

Layout on disk:
  {root}/
  ├── train/
  │   ├── *.jpg
  │   └── _annotations.coco.json
  ├── valid/
  │   ├── *.jpg
  │   └── _annotations.coco.json
  └── test/
      ├── *.jpg
      └── _annotations.coco.json

COCO annotation format:
  bbox: [x, y, width, height]  (top-left origin, pixel coordinates)
  Converted to bbox_xyxy = [x1, y1, x2, y2] on the Instance.

Category → ontology mapping:
  LVR    → liver_parenchyma
  HCC    → liver_lesion
  NO HCC → liver_parenchyma
  HV     → hepatic_vein
  PV     → portal_vein
  IVC    → inferior_vena_cava
  K      → kidney
  K-M    → kidney
  SAG    → liver_parenchyma   (sagittal view label, not a structure)
  TRV    → liver_parenchyma   (transverse view label, not a structure)

Entries emitted — one per image:
  modality_type  = "image"
  anatomy_family = "liver"
  task_type      = "detection"  if the image has ≥1 annotation
                 = "ssl_only"   otherwise
  has_box        = True / False accordingly
  has_mask       = False        (no pixel masks in this dataset)
  instances      = one per COCO annotation  (bbox_xyxy set, mask_path None)
  split          = predefined folder name (train/valid/test)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


_SPLITS: Dict[str, str] = {
    "train": "train",
    "valid": "val",
    "test":  "test",
}

_ONTOLOGY: Dict[str, str] = {
    "LVR":    "liver_parenchyma",
    "HCC":    "liver_lesion",
    "NO HCC": "liver_parenchyma",
    "HV":     "hepatic_vein",
    "PV":     "portal_vein",
    "IVC":    "inferior_vena_cava",
    "K":      "kidney",
    "K-M":    "kidney",
    "SAG":    "liver_parenchyma",
    "TRV":    "liver_parenchyma",
}


class LiverCVProjectAdapter(BaseAdapter):
    """
    Adapter for the Liver Ultrasound CV Project (Roboflow COCO export).
    Yields one image entry per JPG across all three predefined splits.
    """

    DATASET_ID     = "liver-CV-project"
    ANATOMY_FAMILY = "liver"
    SONODQS        = "silver"
    DOI            = "https://public.roboflow.ai/object-detection/undefined"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if any((root / d).is_dir() for d in _SPLITS):
            return root
        candidate = root / "liver_ultrasound.v11i.coco"
        if any((candidate / d).is_dir() for d in _SPLITS):
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected train/ valid/ test/ under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for dir_name, split_label in _SPLITS.items():
            split_dir = self.root / dir_name
            if not split_dir.exists():
                continue

            ann_path = split_dir / "_annotations.coco.json"
            if not ann_path.exists():
                continue

            with ann_path.open(encoding="utf-8") as f:
                coco = json.load(f)

            cat_names: Dict[int, str] = {
                c["id"]: c["name"] for c in coco.get("categories", [])
            }
            anns_by_image: Dict[int, List[dict]] = {}
            for ann in coco.get("annotations", []):
                anns_by_image.setdefault(ann["image_id"], []).append(ann)

            split = self.split_override or split_label

            for img_info in sorted(coco.get("images", []), key=lambda x: x["id"]):
                img_id    = img_info["id"]
                file_name = img_info["file_name"]
                img_path  = split_dir / file_name

                if not img_path.exists():
                    continue

                stem     = Path(file_name).stem
                img_anns = sorted(
                    anns_by_image.get(img_id, []),
                    key=lambda a: a["id"],
                )

                instances = []
                for ann in img_anns:
                    cat_name = cat_names.get(ann["category_id"], "unknown")
                    ontology = _ONTOLOGY.get(cat_name, "liver_parenchyma")
                    x, y, w, h = ann["bbox"]
                    instances.append(self._make_instance(
                        instance_id    = f"{stem}_{ann['id']}",
                        label_raw      = cat_name,
                        label_ontology = ontology,
                        is_promptable  = True,
                        bbox_xyxy      = [x, y, x + w, y + h],
                    ))

                has_box = bool(instances)

                yield self._make_entry(
                    str(img_path),
                    split         = split,
                    modality      = "image",
                    instances     = instances,
                    study_id      = stem,
                    series_id     = stem,
                    view_type     = "liver_bmode",
                    width         = img_info.get("width", 0),
                    height        = img_info.get("height", 0),
                    has_mask      = False,
                    has_box       = has_box,
                    task_type     = "detection" if has_box else "ssl_only",
                    ssl_stream    = "image",
                    is_promptable = has_box,
                    source_meta   = {
                        "coco_image_id": img_id,
                        "file_name":     file_name,
                        "n_annotations": len(instances),
                    },
                )
