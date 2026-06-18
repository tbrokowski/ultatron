"""
data/adapters/muscle/knee_us_jocohS.py  - KneeUS JoCoHS adapter

Dataset:  Knee Ultrasound JoCoHS dataset
Task:     SSL (knee images with optional OA grading labels)
Layout:

    <root>/
        data/
            image/
                ultrasound/
                    imageArchive.{E03USIMGT}/
                        {subject_id}_{scan_idx}.png
            reference/
                dataTable.IMAGE_REF.csv   # E03SUBJECTID, E03USIMGT, E03USIMGF, ...
                dataTable.SUBJECT.csv     # E03SUBJECTID, E03PASKR, E03PASKL, ...

PASK scores (E03PASKR / E03PASKL): 0=normal, 1-5=increasing OA severity.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)


class KneeUSJoCoHSAdapter(BaseAdapter):
    """
    Adapter for the knee ultrasound JoCoHS dataset.
    """

    DATASET_ID     = "KneeUSJoCoHS"
    ANATOMY_FAMILY = "joint"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.7910/DVN/SKP9IB"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_root = self.root / "data"
        self._ref_root  = self._data_root / "reference"
        self._img_root  = self._data_root / "image" / "ultrasound"
        self._image_rows: List[dict] = []
        self._subject_info: Dict[str, dict] = {}
        self._splits: Dict[str, str] = {}
        if self._ref_root.exists():
            self._load_metadata()

    def _load_metadata(self) -> None:
        img_ref_path = self._ref_root / "dataTable.IMAGE_REF.csv"
        subj_path    = self._ref_root / "dataTable.SUBJECT.csv"

        if not img_ref_path.exists():
            log.warning("KneeUSJoCoHS: IMAGE_REF.csv not found at %s", img_ref_path)
            return
        with img_ref_path.open() as f:
            self._image_rows = list(csv.DictReader(f))

        if subj_path.exists():
            with subj_path.open() as f:
                for row in csv.DictReader(f):
                    sid = row.get("E03SUBJECTID", "").strip('"')
                    if sid:
                        self._subject_info[sid] = row

        subjects = sorted({r["E03SUBJECTID"].strip('"') for r in self._image_rows if r.get("E03SUBJECTID")})
        n     = len(subjects)
        n_tr  = int(0.80 * n)
        n_val = int(0.10 * n)
        for i, sid in enumerate(subjects):
            if self.split_override:
                self._splits[sid] = self.split_override
            elif i < n_tr:
                self._splits[sid] = "train"
            elif i < n_tr + n_val:
                self._splits[sid] = "val"
            else:
                self._splits[sid] = "test"

    def _resolve_image_path(self, row: dict) -> Path | None:
        """Resolve image path from IMAGE_REF row (archive keyed by E03USIMGT)."""
        filename = row.get("E03USIMGF", "").strip('"')
        if not filename:
            return None

        archive_type = row.get("E03USIMGT", "").strip('"')
        if archive_type:
            ipath = self._img_root / f"imageArchive.{archive_type}" / filename
            if ipath.exists():
                return ipath

        # Fallback: search all archives
        for archive_dir in sorted(self._img_root.glob("imageArchive.*")):
            ipath = archive_dir / filename
            if ipath.exists():
                return ipath
        return None

    def _build_pask_instances(self, sid: str, subj: dict) -> List[Instance]:
        instances: List[Instance] = []
        for side_key, suffix in (("E03PASKR", "R"), ("E03PASKL", "L")):
            raw = subj.get(side_key, "").strip('"')
            try:
                grade = int(raw)
            except (ValueError, TypeError):
                continue
            if grade < 0:
                continue
            instances.append(self._make_instance(
                instance_id=f"{sid}_{suffix}",
                label_raw=f"knee_oa_grade_{grade}",
                label_ontology="knee_oa_feature",
                is_promptable=False,
            ))
        return instances

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._image_rows:
            log.warning("KneeUSJoCoHS: no metadata loaded — root may be missing or empty.")
            return

        for row in self._image_rows:
            sid = row.get("E03SUBJECTID", "").strip('"')
            if not sid:
                continue

            ipath = self._resolve_image_path(row)
            if ipath is None:
                continue

            split = self._splits.get(sid, "train")
            subj  = self._subject_info.get(sid, {})
            instances = self._build_pask_instances(sid, subj)

            archive_type = row.get("E03USIMGT", "").strip('"')
            depth        = row.get("E03USIMGD", "").strip('"')

            yield self._make_entry(
                str(ipath),
                split=split,
                modality="image",
                instances=instances,
                view_type="knee_longitudinal",
                task_type="weak_label" if instances else "ssl_only",
                ssl_stream="image",
                is_promptable=False,
                probe_type="linear",
                source_meta={
                    "subject_id": sid,
                    "archive_type": archive_type,
                    "depth": depth,
                },
            )
