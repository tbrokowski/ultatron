"""
data/adapters/ovarian/polycystic_ovary_telkom.py  - PCO Telkom binary cls adapter
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_SPLIT_MAP = {
    "train copy": "train",
    "testing":    "test",
    "validation": "val",
}
_CLASSES = {"pco": 1, "normal": 0}


class PolycysticOvaryTelkomAdapter(BaseAdapter):
    DATASET_ID     = "Polycystic-Ovary-US-Telkom"
    ANATOMY_FAMILY = "ovarian"
    SONODQS        = "bronze"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        root = Path(root)
        self._base = root / "dataverse_files" if (root / "dataverse_files").is_dir() else root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for split_dir_name, split in _SPLIT_MAP.items():
            split_root = self._base / split_dir_name
            if not split_root.is_dir():
                continue
            for cls_name, cls_id in _CLASSES.items():
                cls_dir = split_root / cls_name
                if not cls_dir.is_dir():
                    continue
                for img_path in sorted(cls_dir.glob("*.jpg")):
                    if self.split_override:
                        split = self.split_override

                    instances: List[Instance] = [
                        self._make_instance(
                            instance_id=img_path.stem,
                            label_raw=cls_name,
                            label_ontology="pco" if cls_id == 1 else "normal_ovary",
                            is_promptable=False,
                        )
                    ]

                    yield self._make_entry(
                        str(img_path),
                        split=split,
                        modality="image",
                        instances=instances,
                        task_type="binary_cls",
                        ssl_stream="image",
                        is_promptable=False,
                        source_meta={"class_id": cls_id, "class_name": cls_name},
                    )
