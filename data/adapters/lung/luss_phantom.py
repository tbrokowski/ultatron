"""
data/adapters/lung/luss_phantom.py  - LUSS PHANTOM lung segmentation adapter
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class LUSSPhantomAdapter(BaseAdapter):
    DATASET_ID     = "LUSS-PHANTOM"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "silver"
    DOI            = ""

    _SPLIT_DIRS = {"train": "train", "test": "test"}

    def iter_entries(self) -> Iterator[USManifestEntry]:
        data_root = self.root / "data-3"
        if not data_root.is_dir():
            data_root = self.root

        for split_name, split_label in self._SPLIT_DIRS.items():
            img_dir = data_root / split_name / "images"
            msk_dir = data_root / split_name / "masks"
            if not img_dir.is_dir():
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                mask_path = msk_dir / img_path.name
                if not mask_path.exists():
                    continue

                split = self.split_override or split_label
                instances: List[Instance] = [
                    self._make_instance(
                        instance_id=img_path.stem,
                        label_raw="lung_phantom",
                        label_ontology="lung",
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
                )
