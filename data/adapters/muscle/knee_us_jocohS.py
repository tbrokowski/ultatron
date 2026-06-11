"""
data/adapters/muscle/knee_us_jocohS.py  - KneeUS JoCoHS adapter

Dataset:  Knee Ultrasound JoCoHS dataset
Task:     SSL (knee images with optional OA grading labels)
Layout:

    <root>/
        data/
            image/
                ultrasound/
                    imageArchive.{depth}/
                        {subject_id}_{depth}.png
            reference/
                dataTable.IMAGE_REF.csv   # SUBJECTID, DEPTH_NUM, FILENAME, SIZE, DEPTH
                dataTable.SUBJECT.csv     # SUBJECTID, GENDER, PASKR, PASKL, AGE, ...

PASK scores (E03PASKR / E03PASKL): 0=normal, 1-5=increasing OA severity.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)


class KneeUSJoCoHSAdapter(BaseAdapter):
    """
    Adapter for the knee ultrasound JoCoHS dataset.
    """

    DATASET_ID     = "KneeUSJoCoHS"
    ANATOMY_FAMILY = "musculoskeletal"
    SONODQS        = "silver"
    DOI            = ""

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

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._image_rows:
            log.warning("KneeUSJoCoHS: no metadata loaded — root may be missing or empty.")
            return

        for row in self._image_rows:
            sid      = row.get("E03SUBJECTID", "").strip('"')
            filename = row.get("E03USIMGF", "").strip('"')
            depth    = row.get("E03USIMGD", "").strip('"')
            if not sid or not filename:
                continue

            archive_dir = self._img_root / f"imageArchive.{depth}"
            ipath = archive_dir / filename
            if not ipath.exists():
                continue

            split = self._splits.get(sid, "train")
            subj  = self._subject_info.get(sid, {})

            # Optional OA grading label (right/left knee, PASK scale 0-5)
            paskr_str = subj.get("E03PASKR", "").strip('"')
            paskl_str = subj.get("E03PASKL", "").strip('"')
            try:
                paskr = int(paskr_str)
                paskl = int(paskl_str)
            except (ValueError, TypeError):
                paskr = paskl = -1

            instances: List[Instance] = []
            if paskr >= 0:
                instances.append(self._make_instance(
                    instance_id=f"{sid}_R", label_raw=f"OA_grade_{paskr}",
                    label_ontology="osteoarthritis", is_promptable=False,
                ))

            yield self._make_entry(
                str(ipath),
                split=split, modality="image", instances=instances,
                view_type="knee_longitudinal",
                task_type="multiclass_cls" if instances else "ssl_only",
                ssl_stream="image", is_promptable=False,
                source_meta={"subject_id": sid, "depth": depth},
            )
