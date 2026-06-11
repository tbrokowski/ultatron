"""
data/adapters/lung/covid_blues.py  ·  COVID-BLUES lung ultrasound adapter
===========================================================================

COVID Bluepoint Lung Ultrasound (BLUES) dataset — Maastricht UMC+, 2025.
362 standardized BLUE-protocol LUS videos from 63 patients with:
  * severity.csv          — per-video severity (0–3), A-lines, B-lines
  * clinical_variables.csv — per-patient PCR (cov_test), vitals, comorbidities

Expected layout on Store (after download_covid_blues.sh):
  COVID-BLUES/
    lus_videos/              patient_<ID>_<BLUEPOINT>.mp4
    severity.csv
    clinical_variables.csv
    metadata/
      video_labels.jsonl     — merged labels (preferred)
      patient_splits.json

Splits are patient-level (do not leak videos from the same patient).
Source: https://github.com/NinaWie/COVID-BLUES
DOI paper: 10.1109/JBHI.2025.3543686
"""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_VIDEO_STEM_RE = re.compile(r"^patient_(\d+)_([LR]\d+)(?:_(\d+))?$", re.IGNORECASE)
_VIDEO_EXTS = {".mp4", ".avi", ".mov"}


def _parse_video_stem(stem: str) -> tuple[str, str, Optional[str]]:
    m = _VIDEO_STEM_RE.match(stem.strip())
    if not m:
        raise ValueError(f"Unrecognized COVID-BLUES video stem: {stem!r}")
    return m.group(1), m.group(2).upper(), m.group(3)


def _yes_no(val) -> Optional[bool]:
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in ("yes", "y", "1", "true"):
        return True
    if v in ("no", "n", "0", "false"):
        return False
    return None


class COVIDBLUESAdapter(BaseAdapter):
    DATASET_ID = "COVID-BLUES"
    ANATOMY_FAMILY = "lung"
    SONODQS = "gold"
    DOI = "https://doi.org/10.1109/JBHI.2025.3543686"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self.video_dir = self.root / "lus_videos"
        self.labels_jsonl = self.root / "metadata" / "video_labels.jsonl"

    def _iter_from_jsonl(self) -> Iterator[USManifestEntry]:
        with self.labels_jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                video_path = Path(rec["video_path"])
                if not video_path.is_file():
                    alt = self.video_dir / rec.get("video_file", video_path.name)
                    if alt.is_file():
                        video_path = alt
                    else:
                        continue
                yield self._entry_from_record(rec, video_path)

    def _iter_from_csv(self) -> Iterator[USManifestEntry]:
        severity_path = self.root / "severity.csv"
        clinical_path = self.root / "clinical_variables.csv"
        if not severity_path.exists() or not clinical_path.exists():
            log.warning("COVID-BLUES: missing severity.csv or clinical_variables.csv")
            return

        clinical: dict[str, dict] = {}
        with clinical_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                pid = str(row.get("patient_id", "")).strip()
                if pid:
                    clinical[pid] = row

        patients = sorted(clinical.keys(), key=lambda x: int(x))
        n = len(patients)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        patient_splits = {}
        for i, pid in enumerate(patients):
            if i < n_train:
                patient_splits[pid] = "train"
            elif i < n_train + n_val:
                patient_splits[pid] = "val"
            else:
                patient_splits[pid] = "test"

        with severity_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                stem = Path(str(row.get("video_file", "")).strip()).stem
                if not stem:
                    continue
                try:
                    patient_id, blue_point, duplicate_idx = _parse_video_stem(stem)
                except ValueError:
                    continue
                video_path = self.video_dir / f"{stem}.mp4"
                if not video_path.is_file():
                    continue

                clin = clinical.get(patient_id, {})
                cov_raw = clin.get("cov_test", "")
                try:
                    cov_test = int(float(cov_raw)) if str(cov_raw).strip() else None
                except (TypeError, ValueError):
                    cov_test = None
                try:
                    severity_score = float(row.get("Severity Score", ""))
                except (TypeError, ValueError):
                    severity_score = None

                rec = {
                    "video_file": video_path.name,
                    "video_path": str(video_path),
                    "patient_id": patient_id,
                    "blue_point": blue_point,
                    "lung_side": "left" if blue_point.startswith("L") else "right",
                    "duplicate_idx": int(duplicate_idx) if duplicate_idx else None,
                    "severity_score": severity_score,
                    "a_lines": _yes_no(row.get("A-lines")),
                    "b_lines": _yes_no(row.get("B-lines")),
                    "comments": (row.get("comments") or "").strip(),
                    "cov_test": cov_test,
                    "covid_positive": cov_test == 1 if cov_test is not None else None,
                    "split": patient_splits.get(patient_id, "train"),
                }
                yield self._entry_from_record(rec, video_path)

    def _entry_from_record(self, rec: dict, video_path: Path) -> USManifestEntry:
        split = self.split_override or rec.get("split", "train")
        patient_id = str(rec["patient_id"])
        severity = rec.get("severity_score")
        cov_pos = rec.get("covid_positive")

        label_raw = "covid_positive" if cov_pos else "covid_negative" if cov_pos is False else "unknown"
        label_ontology = "lung_covid_positive" if cov_pos else "lung_covid_negative" if cov_pos is False else "lung_other"

        instances = []
        if cov_pos is not None:
            instances.append(self._make_instance(
                instance_id=video_path.stem,
                label_raw=label_raw,
                label_ontology=label_ontology,
                mask_path=None,
                is_promptable=False,
            ))

        comments = rec.get("comments") or ""
        text_parts = []
        if severity is not None:
            text_parts.append(f"LUS severity score {severity:g}/3")
        if rec.get("a_lines") is not None:
            text_parts.append("A-lines present" if rec["a_lines"] else "no A-lines")
        if rec.get("b_lines") is not None:
            text_parts.append("B-lines present" if rec["b_lines"] else "no B-lines")
        if cov_pos is not None:
            text_parts.append("COVID-19 PCR positive" if cov_pos else "COVID-19 PCR negative")
        if comments:
            text_parts.append(comments)
        report_text = "; ".join(text_parts) if text_parts else None

        entry = self._make_entry(
            str(video_path),
            split,
            modality="video",
            instances=instances,
            has_mask=False,
            task_type="classification" if cov_pos is not None else "ssl_only",
            ssl_stream="video",
            is_promptable=False,
            probe_type="curvilinear",
            study_id=patient_id,
            source_meta={
                "patient_id": patient_id,
                "blue_point": rec.get("blue_point"),
                "lung_side": rec.get("lung_side"),
                "duplicate_idx": rec.get("duplicate_idx"),
                "severity_score": severity,
                "a_lines": rec.get("a_lines"),
                "b_lines": rec.get("b_lines"),
                "cov_test": rec.get("cov_test"),
                "covid_positive": cov_pos,
                "comments": comments,
                "report_text": report_text,
                "doi": self.DOI,
            },
        )
        entry.has_temporal_order = True
        return entry

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            log.warning("COVID-BLUES: root missing at %s", self.root)
            return

        if self.labels_jsonl.is_file():
            yield from self._iter_from_jsonl()
        else:
            yield from self._iter_from_csv()
