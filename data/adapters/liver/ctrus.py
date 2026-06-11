"""
data/adapters/liver/ctrus.py  - C-TRUS colon wall segmentation adapter
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class CTRUSAdapter(BaseAdapter):
    DATASET_ID     = "C-TRUS"
    ANATOMY_FAMILY = "liver"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._base = self._find_base(self.root)
        self._meta = self._load_csv()

    @staticmethod
    def _find_base(root: Path) -> Path:
        nested = root / "c-trus-main"
        return nested if nested.is_dir() else root

    def _load_csv(self) -> Dict[str, dict]:
        csv_path = self._base / "c-trus.csv"
        meta: Dict[str, dict] = {}
        if not csv_path.exists():
            return meta
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                fname = row.get("file", "").strip()
                if fname:
                    meta[fname] = row
        return meta

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self._base / "original"
        lbl_dir = self._base / "labels"
        if not img_dir.is_dir():
            return

        images = sorted(img_dir.glob("*.jpg"))
        for img_path in images:
            mask_path = lbl_dir / img_path.name
            if not mask_path.exists():
                continue

            row = self._meta.get(img_path.name, {})
            fold = int(row.get("testitem_in_fold", 0) or 0)
            if self.split_override:
                split = self.split_override
            elif fold == 0:
                split = "test"
            else:
                split = "train"

            instances: List[Instance] = [
                self._make_instance(
                    instance_id=img_path.stem,
                    label_raw="colon_wall",
                    label_ontology="colon",
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
                source_meta={
                    "patient": row.get("patient"),
                    "quality": row.get("quality_name"),
                    "fold": fold,
                },
            )
