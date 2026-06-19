"""
data/adapters/kidney/normal_kidney_cv.py  ·  Normal Kidney CV adapter
=======================================================================

Roboflow COCO-segmentation export of the Normal Kidney CV dataset.

Layout on Store:

    {root}/.../train/
      *.jpg                   ← 1080 images
      _annotations.coco.json

COCO categories: Normal-Kidney (0), Kidney (1), Liver (2), Spleen (3).
Multiple annotations per image possible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class NormalKidneyCVAdapter(BaseAdapter):
    DATASET_ID     = "Normal-Kidney-CV"
    ANATOMY_FAMILY = "kidney"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._coco_dir, self._coco = self._load_coco()

    def _load_coco(self) -> tuple[Path, dict]:
        for sub in self.root.rglob("train"):
            ann = sub / "_annotations.coco.json"
            if ann.exists():
                with ann.open() as f:
                    return sub, json.load(f)
        ann = self.root / "train" / "_annotations.coco.json"
        if ann.exists():
            with ann.open() as f:
                return ann.parent, json.load(f)
        return self.root / "train", {}

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._coco:
            return

        cat_names: Dict[int, str] = {
            c["id"]: c["name"]
            for c in self._coco.get("categories", [])
        }

        img_by_id: Dict[int, dict] = {
            img["id"]: img for img in self._coco.get("images", [])
        }
        anns_by_img: Dict[int, list] = {}
        for ann in self._coco.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        split = self.split_override or "train"

        for img_id, img_info in sorted(img_by_id.items(),
                                       key=lambda x: x[1].get("file_name", "")):
            fname = img_info.get("file_name", "")
            img_path = self._coco_dir / fname
            if not img_path.exists():
                continue

            # study_id from extra.name stem (original filename without extension)
            extra = img_info.get("extra", {})
            orig_name = extra.get("name", "") if isinstance(extra, dict) else ""
            study_id = Path(orig_name).stem if orig_name else Path(fname).stem

            anns = anns_by_img.get(img_id, [])

            # unique category names for this image
            unique_cats = list(dict.fromkeys(
                cat_names.get(a["category_id"], "unknown")
                for a in anns
            ))

            instances: List[Instance] = []
            for j, ann in enumerate(anns):
                cat_name = cat_names.get(ann["category_id"], "unknown")
                inst = self._make_instance(
                    instance_id=f"{Path(fname).stem}_{j}",
                    label_raw=cat_name,
                    label_ontology=cat_name.lower().replace("-", "_"),
                    is_promptable=bool(ann.get("segmentation")),
                )
                if ann.get("bbox"):
                    x, y, w, h = ann["bbox"]
                    inst.bbox_xyxy = [x, y, x + w, y + h]
                if ann.get("segmentation"):
                    inst.polygon = ann["segmentation"][0] if ann["segmentation"] else None
                instances.append(inst)

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                study_id=study_id,
                label_raw=unique_cats if unique_cats else None,
                height=img_info.get("height", 0),
                width=img_info.get("width", 0),
                has_mask=True,
                has_box=True,
                has_temporal_order=False,
                num_frames=1,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=bool(instances),
                source_meta={
                    "coco_image_id":    img_id,
                    "coco_annotations": anns,
                    "coco_segmentation": [
                        a.get("segmentation") for a in anns
                    ],
                },
            )
