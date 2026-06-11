"""
data/adapters/maternal_fetal/ultrasound_fetus.py  - Ultrasound fetus dataset adapter
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_SPLIT_MAP = {
    "train":      "train",
    "validation": "val",
    "test":       "test",
}
_CLASSES = {
    "benign":    ("benign",    0),
    "malignant": ("malignant", 1),
    "normal":    ("normal",    2),
}


class UltrasoundFetusAdapter(BaseAdapter):
    DATASET_ID     = "ultrasound-fetus-dataset"
    ANATOMY_FAMILY = "fetal"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_root = self._find_data_root(self.root)

    @staticmethod
    def _find_data_root(root: Path) -> Path:
        for sub in root.rglob("Data"):
            if (sub / "train").is_dir() or (sub / "validation").is_dir():
                return sub
        nested = root / "Ultrasound Fetus Dataset" / "Ultrasound Fetus Dataset" / "Data" / "Data"
        if nested.is_dir():
            return nested
        return root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for split_dir_name, split in _SPLIT_MAP.items():
            split_root = self._data_root / split_dir_name
            if not split_root.is_dir():
                continue
            for cls_name, (label_raw, cls_id) in _CLASSES.items():
                cls_dir = split_root / cls_name
                if not cls_dir.is_dir():
                    continue
                for img_path in sorted(cls_dir.glob("*.png")):
                    if self.split_override:
                        split = self.split_override

                    instances: List[Instance] = [
                        self._make_instance(
                            instance_id=img_path.stem,
                            label_raw=label_raw,
                            label_ontology="fetal_health",
                            is_promptable=False,
                        )
                    ]

                    yield self._make_entry(
                        str(img_path),
                        split=split,
                        modality="image",
                        instances=instances,
                        task_type="multiclass_cls",
                        ssl_stream="image",
                        is_promptable=False,
                        source_meta={"class_id": cls_id, "class_name": cls_name},
                    )
