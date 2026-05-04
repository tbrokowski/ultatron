"""
data/adapters/maternal_fetal/pbf_us1.py  ·  PBF-US1 adapter
=============================================================

PBF-US1 (NatalIA Phantom Blind-sweep Fetal Ultrasound):
19,407 JPEG frames from 90 freehand sweep exams of a 23-week phantom,
labelled per-frame with a 6-class fetal standard plane taxonomy.

Layout on disk:
  {root}/
  ├── Obstetrics Exam - <DD-Mon-YYYY>_<HH>_<AM/PM>/   # 90 folders, names with spaces
  │   └── cineframe_<N>_<ISO_timestamp>.jpeg
  ├── resume.csv    — per-frame labels  (19,407 rows)
  └── metadata.csv  — per-exam metadata (90 rows)

resume.csv (comma-delimited):
  file_name   JPEG filename (no path)
  studie      Exam folder name — join key to locate file
  class       Human-readable plane label
  value       Integer class 0–5
  image       Empty column (ignored)

metadata.csv (comma-delimited):
  Study Name  Exam folder name (join key)
  protocol    Sweep type: Vertical / Horizontal / Diagonal / Diagonal \
  position    Fetal pose: OP / SP / OA / SA

Class taxonomy:
  0  Biparietal standard plane  (   42 frames)
  1  Abdominal standard plane   (   63 frames)
  2  Heart standard plane       (   61 frames)
  3  Spine standard plane       (  134 frames)
  4  Femur standard plane       (   46 frames)
  5  No plane                   (19,061 frames — 98.2 %)

Split strategy:
  No predefined split. Assign by exam folder (studie) to prevent temporal
  leakage within sweeps. Sorted studie names → 80/10/10 by index.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

PLANE_CLASSES: List[str] = [
    "Biparietal standard plane",  # 0
    "Abdominal standard plane",   # 1
    "Heart standard plane",       # 2
    "Spine standard plane",       # 3
    "Femur standard plane",       # 4
    "No plane",                   # 5
]


class PBFUS1Adapter(BaseAdapter):
    DATASET_ID     = "PBF-US1"
    ANATOMY_FAMILY = "fetal_planes"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "resume.csv").exists():
            return root
        candidate = root / "PBF-US1"
        if (candidate / "resume.csv").exists():
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected resume.csv under {root}"
        )

    def _load_resume(self) -> List[dict]:
        path = self.root / "resume.csv"
        with path.open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def _load_metadata(self) -> Dict[str, dict]:
        path = self.root / "metadata.csv"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8-sig") as f:
            return {row["Study Name"].strip(): row for row in csv.DictReader(f)}

    def iter_entries(self) -> Iterator[USManifestEntry]:
        rows = self._load_resume()
        meta = self._load_metadata()

        studies: List[str] = sorted({
            (r.get("studie") or "").strip()
            for r in rows
            if (r.get("studie") or "").strip()
        })
        n = len(studies)
        study_split: Dict[str, str] = {
            s: self._infer_split(s, i, n)
            for i, s in enumerate(studies)
        }

        for row in rows:
            studie    = (row.get("studie")    or "").strip()
            file_name = (row.get("file_name") or "").strip()
            cls_name  = (row.get("class")     or "").strip()
            value_str = (row.get("value")     or "").strip()

            if not studie or not file_name:
                continue

            img_path = self.root / studie / file_name
            if not img_path.exists():
                continue

            try:
                cls_idx = int(value_str)
            except (ValueError, TypeError):
                continue

            if cls_idx < 0 or cls_idx >= len(PLANE_CLASSES):
                continue

            split     = self.split_override or study_split.get(studie, "train")
            exam_meta = meta.get(studie, {})
            stem      = img_path.stem

            instance = self._make_instance(
                instance_id          = stem,
                label_raw            = cls_name or PLANE_CLASSES[cls_idx],
                label_ontology       = "pbf_us1_planes",
                is_promptable        = False,
                classification_label = cls_idx,
            )

            yield self._make_entry(
                str(img_path),
                split         = split,
                modality      = "image",
                instances     = [instance],
                study_id      = studie,
                series_id     = stem,
                height        = 500,
                width         = 700,
                has_mask      = False,
                task_type     = "classification",
                ssl_stream    = "image",
                is_promptable = False,
                source_meta   = {
                    "studie":   studie,
                    "protocol": (exam_meta.get("protocol") or "").strip() or None,
                    "position": (exam_meta.get("position") or "").strip() or None,
                },
            )
