"""
data/adapters/stu_hospital.py  ·  STU-Hospital segmentation adapter
=====================================================================

STU-Hospital-master contains PNG ultrasound images with paired binary masks.
The dataset lives under `Hospital/` and is treated as a small multi-organ
segmentation set.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class STUHospitalAdapter(BaseAdapter):
    DATASET_ID = "STU-Hospital-master"
    ANATOMY_FAMILY = "multi"
    SONODQS = "silver"
    DOI = ""

    def __init__(self, root, split_override=None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "Hospital").exists():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "Hospital").exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected 'Hospital/' under {root}")

    def iter_entries(self) -> Iterator[USManifestEntry]:
        image_paths = sorted((self.root / "Hospital").glob("Test_Image_*.png"))
        n = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            sample_id = self._sample_id(img_path.stem)
            if sample_id is None:
                continue
            mask_path = img_path.parent / f"mask_{sample_id}.png"
            if not mask_path.exists():
                continue

            instances = [
                self._make_instance(
                    instance_id=f"stu_hospital_{sample_id}",
                    label_raw="target_region",
                    label_ontology="target_region",
                    mask_path=str(mask_path),
                    is_promptable=True,
                )
            ]

            yield self._make_entry(
                str(img_path),
                split=self._infer_split(str(sample_id), idx, n),
                modality="image",
                instances=instances,
                study_id=f"stu_hospital_{sample_id}",
                view_type="hospital_segmentation",
                has_mask=True,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
                source_meta={
                    "sample_index": sample_id,
                },
            )

    @staticmethod
    def _sample_id(stem: str) -> str | None:
        match = re.search(r"(\d+)$", stem)
        return match.group(1) if match else None
