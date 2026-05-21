"""
data/adapters/gallbladder/gist514_db.py  ·  GIST514-DB adapter
===============================================================

GIST514-DB — "Query2: Query over queries for improving gastrointestinal
stromal tumour detection in an endoscopic ultrasound"
Howard et al., Computers in Biology and Medicine, 2022.

  514 endoscopic ultrasound (EUS) images from 514 cases.
  Task: object detection + instance segmentation.
  Classes: GIST + other subepithelial lesions (leiomyoma, schwannoma, ...)
  Annotations: COCO JSON format (bboxes + optional masks).
  Extra: anatomical location per image (stored in annotation JSON).

DOI     : https://doi.org/10.1016/j.compbiomed.2022.106382
SonoDQS : gold (multi-class, expert-labelled, 514 cases)
Probe   : radial EUS

Dataset layout (standard COCO format)
--------------------------------------
  {root}/
    images/
      train/   (or train2017/)
        *.jpg
      val/     (or val2017/)
        *.jpg
      test/    (or test2017/)
        *.jpg
    annotations/
      instances_train.json    (or instances_train2017.json)
      instances_val.json
      instances_test.json

COCO JSON has extra field per image: "anatomical_location" (optional).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Map raw COCO category names → label_ontology
_LABEL_MAP: dict[str, str] = {
    "gist":         "gist",
    "leiomyoma":    "leiomyoma",
    "schwannoma":   "schwannoma",
    "lipoma":       "lipoma",
    "ectopic_pancreas": "ectopic_pancreas",
    "carcinoid":    "carcinoid",
    "granular_cell_tumor": "granular_cell_tumor",
    "duplication_cyst": "duplication_cyst",
}

# Possible annotation file names per split
_ANN_CANDIDATES = {
    "train": ["instances_train.json", "instances_train2017.json", "train.json"],
    "val":   ["instances_val.json",   "instances_val2017.json",   "val.json"],
    "test":  ["instances_test.json",  "instances_test2017.json",  "test.json"],
}

# Possible image directory names per split
_IMG_DIR_CANDIDATES = {
    "train": ["train", "train2017"],
    "val":   ["val",   "val2017"],
    "test":  ["test",  "test2017"],
}


def _find_ann(ann_dir: Path, split: str) -> Path | None:
    for name in _ANN_CANDIDATES.get(split, []):
        p = ann_dir / name
        if p.exists():
            return p
    return None


def _find_img_dir(images_dir: Path, split: str) -> Path | None:
    for name in _IMG_DIR_CANDIDATES.get(split, []):
        p = images_dir / name
        if p.is_dir():
            return p
    return None


class GIST514DBAdapter(BaseAdapter):
    """
    Adapter for the GIST514-DB endoscopic ultrasound dataset.

    Yields one USManifestEntry per image. Each entry has:
    - modality_type = "image"
    - anatomy_family = "gallbladder" (GI tract / EUS)
    - task_type = "detection" (bbox present) or "segmentation" (mask present)
    - instances: one per annotated lesion with bbox_xyxy + optional mask

    Parameters
    ----------
    root : str | Path
        Root directory containing images/ and annotations/.
    split_override : str, optional
        Force all entries to a single split.
    """

    DATASET_ID     = "GIST514-DB"
    ANATOMY_FAMILY = "gallbladder"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.1016/j.compbiomed.2022.106382"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        ann_dir    = self.root / "annotations"
        images_dir = self.root / "images"

        for split_name in ("train", "val", "test"):
            ann_path = _find_ann(ann_dir, split_name)
            if ann_path is None:
                continue

            img_dir = _find_img_dir(images_dir, split_name)

            split = self.split_override or split_name
            yield from self._iter_split(ann_path, img_dir, split)

    def _iter_split(
        self, ann_path: Path, img_dir: Path | None, split: str
    ) -> Iterator[USManifestEntry]:
        with open(ann_path) as f:
            coco = json.load(f)

        # Build category index
        cat_map: dict[int, str] = {
            c["id"]: c["name"] for c in coco.get("categories", [])
        }

        # Build annotations index: image_id → list of annotations
        ann_index: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            ann_index.setdefault(ann["image_id"], []).append(ann)

        for img_info in coco.get("images", []):
            img_id   = img_info["id"]
            filename = img_info["file_name"]

            # Resolve image path
            img_path: Path | None = None
            if img_dir is not None:
                candidate = img_dir / Path(filename).name
                if candidate.exists():
                    img_path = candidate
            if img_path is None:
                # Try root fallback
                for ext in [""]:
                    p = self.root / filename
                    if p.exists():
                        img_path = p
                        break
            if img_path is None:
                img_path = (img_dir or self.root) / Path(filename).name

            anns      = ann_index.get(img_id, [])
            has_mask  = any("segmentation" in a and a["segmentation"] for a in anns)
            has_box   = bool(anns)
            task_type = "segmentation" if has_mask else ("detection" if has_box else "ssl_only")

            instances = []
            for a in anns:
                cat_name   = cat_map.get(a["category_id"], "gist")
                label_onto = _LABEL_MAP.get(cat_name.lower(), "gi_lesion")
                inst = self._make_instance(
                    instance_id    = str(a["id"]),
                    label_raw      = cat_name,
                    label_ontology = label_onto,
                    mask_path      = None,
                    is_promptable  = True,
                )
                # Bbox: COCO format [x, y, w, h] → [xmin, ymin, xmax, ymax]
                if "bbox" in a and a["bbox"]:
                    x, y, w, h = a["bbox"]
                    inst.bbox_xyxy = (x, y, x + w, y + h)
                instances.append(inst)

            # Optional anatomical location stored in image metadata
            anatomy_loc = img_info.get("anatomical_location") or img_info.get("location")

            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = instances,
                has_mask      = has_mask,
                has_box       = has_box,
                task_type     = task_type,
                ssl_stream    = "image",
                is_promptable = has_box,
                probe_type    = "radial",
                source_meta   = {
                    "image_id":       img_id,
                    "filename":       filename,
                    "anatomy_loc":    anatomy_loc,
                    "doi":            self.DOI,
                },
            )
