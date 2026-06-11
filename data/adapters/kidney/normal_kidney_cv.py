"""
data/adapters/kidney/normal_kidney_cv.py  - Normal Kidney CV (Roboflow COCO) adapter
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

        img_by_id: Dict[int, dict] = {img["id"]: img for img in self._coco.get("images", [])}
        anns_by_img: Dict[int, list] = {}
        for ann in self._coco.get("annotations", []):
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        items = sorted(img_by_id.items(), key=lambda x: x[1].get("file_name", ""))
        n = len(items)
        for i, (img_id, img_info) in enumerate(items):
            fname = img_info.get("file_name", "")
            img_path = self._coco_dir / fname
            if not img_path.exists():
                continue

            split = self._infer_split(Path(fname).stem, i, n)
            anns = anns_by_img.get(img_id, [])
            instances: List[Instance] = []
            for j, ann in enumerate(anns):
                instances.append(
                    self._make_instance(
                        instance_id=f"{Path(fname).stem}_{j}",
                        label_raw="kidney",
                        label_ontology="kidney",
                        is_promptable=bool(ann.get("segmentation")),
                    )
                )

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                has_mask=len(anns) > 0,
                task_type="segmentation" if anns else "ssl_only",
                ssl_stream="image",
                is_promptable=len(anns) > 0,
                source_meta={"coco_image_id": img_id, "n_annotations": len(anns)},
            )
