"""
data/adapters/liver/ctrus.py  ·  C-TRUS colon wall segmentation adapter
=========================================================================

C-TRUS: 827 transrectal ultrasound images of the colon wall from 13 patients,
with manual segmentation masks and annotation quality scores.

Layout on Store:

    {root}/c-trus-main/
      original/   *.jpg          ← 827 images
      labels/     *.jpg          ← corresponding masks, same filename
      c-trus.csv                 ← file, quality, quality_name, patient,
                                    testitem_in_fold

Split by cross-validation fold:
  fold 0 → test
  fold 1 → val
  fold 2-4 → train
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)


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
            log.warning("C-TRUS: c-trus.csv not found under %s", self._base)
            return meta
        with csv_path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                fname = row.get("file", "").strip()
                if fname:
                    meta[fname] = row
        return meta

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self._base / "original"
        lbl_dir = self._base / "labels"
        if not img_dir.is_dir():
            log.warning("C-TRUS: original/ not found under %s", self._base)
            return

        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = lbl_dir / img_path.name
            if not mask_path.exists():
                log.warning("C-TRUS: mask not found for %s — skipping", img_path.name)
                continue

            row = self._meta.get(img_path.name, {})
            patient = str(row.get("patient", "")).strip()
            quality_name = str(row.get("quality_name", "")).strip()

            try:
                fold = int(row.get("testitem_in_fold", 2))
            except (ValueError, TypeError):
                fold = 2

            if self.split_override:
                split = self.split_override
            elif fold == 0:
                split = "test"
            elif fold == 1:
                split = "val"
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
                study_id=f"patient_{patient}" if patient else img_path.stem,
                label_raw=["colon_wall"],
                has_mask=True,
                has_temporal_order=False,
                num_frames=1,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
                source_meta={
                    "patient":      patient,
                    "quality_name": quality_name,
                    "fold":         fold,
                },
            )
