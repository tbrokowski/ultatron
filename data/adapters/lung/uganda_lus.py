"""
data/adapters/lung/uganda_lus.py  ·  Uganda LUS adapter
=========================================================

Dataset layout (Store):

    /capstor/.../lung/Uganda LUS/A Dataset of Lung Ultrasound Images for Automated/
      dataset/
        train/
          covid/        *.png
          healthy/      *.png
          other/        *.png
        test/
          covid/ healthy/ other/
        validation/
          covid/ healthy/ other/

One entry per image. Split is encoded in the folder name (validation → val).
Label comes from the class subfolder name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_SPLIT_MAP = {
    "train": "train",
    "test": "test",
    "validation": "val",
}

_CLASSES = ("covid", "healthy", "other")
_IMG_EXTS = {".png", ".jpg", ".jpeg"}


class UgandaLUSAdapter(BaseAdapter):
    DATASET_ID     = "uganda-lus"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "silver"
    DOI            = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        dataset_root = self.root / "dataset"
        if not dataset_root.is_dir():
            raise FileNotFoundError(
                f"uganda-lus: expected 'dataset/' subdirectory under {self.root}"
            )

        for split_folder, split in _SPLIT_MAP.items():
            if self.split_override:
                split = self.split_override
            split_dir = dataset_root / split_folder
            if not split_dir.is_dir():
                continue

            for cls in _CLASSES:
                cls_dir = split_dir / cls
                if not cls_dir.is_dir():
                    continue

                images = sorted(
                    p for p in cls_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in _IMG_EXTS
                )
                for img_path in images:
                    instances: List[Instance] = [
                        self._make_instance(
                            instance_id=img_path.stem,
                            label_raw=cls,
                            label_ontology=cls,
                            is_promptable=False,
                        )
                    ]
                    yield self._make_entry(
                        str(img_path),
                        split=split,
                        modality="image",
                        instances=instances,
                        study_id=img_path.stem,
                        has_mask=False,
                        has_temporal_order=False,
                        num_frames=1,
                        task_type="multiclass_cls",
                        ssl_stream="image",
                        is_promptable=False,
                        source_meta={"class": cls},
                    )
