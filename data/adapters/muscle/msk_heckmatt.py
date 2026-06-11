"""
data/adapters/muscle/msk_heckmatt.py  - MSK Heckmatt grading adapter

Layout:
  <root>/.../images/{patientID}_{muscleCode}_{side}_{grade}.png
  <root>/.../masks/  (same filenames)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class MSKHeckmattAdapter(BaseAdapter):
    DATASET_ID     = "msk-heckmatt-radboud"
    ANATOMY_FAMILY = "musculoskeletal"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._img_dir, self._mask_dir = self._find_image_mask_dirs(self.root)

    @staticmethod
    def _find_image_mask_dirs(root: Path) -> tuple[Path, Path]:
        candidates = [root]
        if root.is_dir():
            candidates.extend(p for p in root.iterdir() if p.is_dir())
        for candidate in candidates:
            img = candidate / "images"
            msk = candidate / "masks"
            if img.is_dir() and msk.is_dir():
                return img, msk
        return root / "images", root / "masks"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._img_dir.exists():
            return

        images = sorted(self._img_dir.glob("*.png"))
        n = len(images)
        for i, img_path in enumerate(images):
            mask_path = self._mask_dir / img_path.name
            if not mask_path.exists():
                continue

            parts = img_path.stem.split("_")
            grade = parts[-1] if parts else "unknown"
            split = self._infer_split(img_path.stem, i, n)

            instances: List[Instance] = [
                self._make_instance(
                    instance_id=img_path.stem,
                    label_raw=f"heckmatt_grade_{grade}",
                    label_ontology="muscle",
                    mask_path=str(mask_path),
                    is_promptable=True,
                )
            ]

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                has_mask=True,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
                source_meta={"heckmatt_grade": grade},
            )
