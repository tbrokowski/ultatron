"""
data/adapters/lung/lung_database.py  - Lung Database SSL adapter (~325k images)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class LungDatabaseAdapter(BaseAdapter):
    DATASET_ID     = "Lung-Database"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "bronze"
    DOI            = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        patient_dirs: List[Path] = []

        for p in sorted(self.root.glob("Pt*")):
            img_dir = p / "images"
            if img_dir.is_dir():
                patient_dirs.append(img_dir)

        for ed in sorted(self.root.glob("ED*")):
            img_dir = ed / "processed_N_images_batch" / "images"
            if img_dir.is_dir():
                patient_dirs.append(img_dir)

        n_patients = len(patient_dirs)
        for i, img_dir in enumerate(patient_dirs):
            if self.split_override:
                split = self.split_override
            elif i < int(0.8 * n_patients):
                split = "train"
            elif i < int(0.9 * n_patients):
                split = "val"
            else:
                split = "test"

            for img_path in sorted(img_dir.glob("*.jpg")):
                yield self._make_entry(
                    str(img_path),
                    split=split,
                    modality="image",
                    task_type="ssl_only",
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={"patient_dir": img_dir.parent.name},
                )
