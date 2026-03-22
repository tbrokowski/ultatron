"""
data/adapters/rsa_lus.py  ·  RSA lung ultrasound adapter
================================================================

Dataset layout (Store):

    /capstor/store/cscs/swissai/a127/ultrasound/raw/lung/RSA_Videos/
      cleaned/
        videos/                 # {PatientID}_{Site}_{Depth}_{Count}.mp4
        processed_files.csv
        rsa_pathology_labels.csv

rsa_pathology_labels.csv has the same one-hot site-level findings as Benin,
plus per-site severity scores:

    record_id, TB Label, Pneumonia, Covid, data_source,
    APXD_severity, QASD_severity, ..., QPIG_severity,
    APXD_A-line, APXD_B-lines, ...

We interpret:
  * TB Label, Pneumonia, Covid   → patient-level labels (per record_id)
  * {SITE}_{finding}             → video-level multilabel vector
  * {SITE}_severity              → ordinal severity (1–7, -1 = not measured)

This adapter mirrors BeninLUSAdapter but adds a per-video `severity` field
in source_meta for a regression-style head.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

from .benin_lus import (
    LUS_CANONICAL_FINDINGS,
    _SUFFIX_TO_INDEX,
)


class RSALUSAdapter(BaseAdapter):
    """
    Adapter for the RSA lung ultrasound video dataset.
    """

    DATASET_ID = "RSA-LUS"
    ANATOMY_FAMILY = "lung"
    SONODQS = "silver"
    DOI = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self.cleaned_root = self.root / "cleaned"
        self._processed_rows: List[dict] = []
        self._labels_by_patient: Dict[str, dict] = {}
        self._patient_splits: Dict[str, str] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        processed_path = self.cleaned_root / "processed_files.csv"
        labels_path = self.cleaned_root / "rsa_pathology_labels.csv"

        if not processed_path.exists():
            raise FileNotFoundError(f"RSA-LUS processed_files.csv not found at {processed_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"RSA-LUS rsa_pathology_labels.csv not found at {labels_path}")

        with processed_path.open() as f:
            reader = csv.DictReader(f)
            self._processed_rows = [row for row in reader if row.get("Type", "").lower() == "video"]

        with labels_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("record_id")
                if not pid:
                    continue
                self._labels_by_patient[pid] = row

        patients = sorted({row["Patient ID"] for row in self._processed_rows if row.get("Patient ID")})
        n = len(patients)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        for i, pid in enumerate(patients):
            if self.split_override:
                split = self.split_override
            elif i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            self._patient_splits[pid] = split

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_video_labels_and_severity(patient_row: dict, site: str) -> Tuple[List[float], int]:
        vec = [0.0] * len(LUS_CANONICAL_FINDINGS)
        site_prefix = site.strip()

        # One-hot findings
        for suffix, idx in _SUFFIX_TO_INDEX.items():
            col = f"{site_prefix}_{suffix}"
            if col in patient_row:
                try:
                    val = int(patient_row[col])
                except ValueError:
                    val = 0
                if val == 1:
                    vec[idx] = 1.0

        # Severity (if available)
        sev_col = f"{site_prefix}_severity"
        severity = -1
        if sev_col in patient_row:
            try:
                severity = int(patient_row[sev_col])
            except ValueError:
                severity = -1

        return vec, severity

    # ── Main iterator ─────────────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[USManifestEntry]:
        videos_root = self.cleaned_root / "videos"

        for row in self._processed_rows:
            patient_id = row.get("Patient ID")
            if not patient_id:
                continue
            site = row.get("Site", "").strip()
            depth = row.get("Depth", "").strip()
            new_name = row.get("New File Name")
            if not new_name:
                continue

            vpath = videos_root / new_name
            if not vpath.exists():
                continue

            split = self._patient_splits.get(patient_id, "train")

            label_row = self._labels_by_patient.get(patient_id)
            patient_labels = None
            video_labels: List[float] = []
            severity = -1
            instances: List[Instance] = []

            if label_row is not None:
                def _safe_int(key: str) -> int:
                    try:
                        return int(label_row.get(key, 0))
                    except ValueError:
                        return 0

                tb = _safe_int("TB Label")
                pneumonia = _safe_int("Pneumonia")
                covid = _safe_int("Covid")
                patient_labels = {
                    "tb": int(tb > 0),
                    "pneumonia": int(pneumonia > 0),
                    "covid": int(covid > 0),
                }

                video_labels, severity = self._build_video_labels_and_severity(label_row, site)

                for idx, val in enumerate(video_labels):
                    if val <= 0.0:
                        continue
                    canon = LUS_CANONICAL_FINDINGS[idx]
                    inst_id = f"{patient_id}_{site}_{canon}"
                    instances.append(
                        self._make_instance(
                            instance_id=inst_id,
                            label_raw=f"{site}_{canon}",
                            label_ontology=canon,
                            is_promptable=False,
                        )
                    )

            has_labels = patient_labels is not None or video_labels
            task_type = "multilabel_cls" if has_labels else "ssl_only"

            source_meta = {
                "patient_id": patient_id,
                "site": site,
                "depth": depth,
            }
            if patient_labels is not None:
                source_meta["patient_labels"] = patient_labels
            if video_labels:
                source_meta["video_labels"] = video_labels
            if severity >= 0:
                source_meta["severity"] = severity

            yield self._make_entry(
                str(vpath),
                split=split,
                modality="video",
                instances=instances,
                study_id=patient_id,
                view_type=site,
                is_cine=True,
                has_temporal_order=True,
                task_type=task_type,
                # Include RSA videos in BOTH streams so Phase 3 can sample
                # frames (image stream) paired with clips (video stream).
                ssl_stream="both",
                is_promptable=False,
                source_meta=source_meta,
            )

