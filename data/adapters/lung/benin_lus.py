"""
data/adapters/benin_lus.py  ·  Benin lung ultrasound adapter
================================================================

Dataset layout (Store):

    /capstor/store/cscs/swissai/a127/ultrasound/raw/lung/Benin_Videos/
      cleaned/
        videos/                 # {PatientID}_{Site}_{Depth}_{Count}.mp4
        images/                 # {PatientID}_{Site}_{Depth}_{Count}.png (still frames)
        processed_files.csv     # Patient ID, Site, Depth, Count, New File Name, type, ...
        labels_multidiagnosis.csv

labels_multidiagnosis.csv is patient-level:

    record_id, TB Label, Pneumonia, Covid, APXD_A-line, APXD_B-lines, ...

We treat:
  * TB Label, Pneumonia, Covid  → patient-level labels
  * {SITE}_{finding}            → video-level multilabel for that site

This adapter emits one USManifestEntry per media file (video OR image), with:
  * dataset_id      = "Benin-LUS"
  * anatomy_family  = "lung"
  * modality_type   = "video" (for .mp4 clips) | "image" (for still frames)
  * ssl_stream      = "both" (video) | "image" (still frame)
  * task_type       = "multilabel_cls"
  * study_id        = patient_id  (enables PatientLevelDataset grouping)
  * source_meta:
        {
          "patient_id":      str,
          "site":            str,
          "depth":           str,
          "patient_labels":  {"tb": 0|1, "pneumonia": 0|1, "covid": 0|1},
          "video_labels":    [float] * 7,   # a_line, b_line, confluent_b_line,
                                           # pleural_effusion, large_consolidation,
                                           # small_consolidation, pneumothorax
        }

Per-finding instances are added for bookkeeping (one per positive finding),
but classification heads primarily consume source_meta["video_labels"].
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


LUS_CANONICAL_FINDINGS: Tuple[str, ...] = (
    "a_line",
    "b_line",
    "confluent_b_line",
    "pleural_effusion",
    "large_consolidation",
    "small_consolidation",
    "pneumothorax",
)

# Mapping from CSV column suffix → index in LUS_CANONICAL_FINDINGS
_SUFFIX_TO_INDEX: Dict[str, int] = {
    "A-line": 0,
    "B-lines": 1,
    "Confluent B-lines": 2,
    "Pleural effusion": 3,
    "large Consolidations": 4,
    "small Consolidations or Nodules": 5,
    "Pattern A' (pneumothorax)": 6,
}


def row_media_type(row: dict) -> str:
    """Return normalized media type from processed_files.csv (Type or type column)."""
    raw = row.get("Type") or row.get("type") or "video"
    return str(raw).strip().lower()


def resolve_lus_media(
    row: dict,
    videos_root: Path,
    images_root: Path,
) -> Optional[Tuple[Path, str, str, bool, Dict[str, object]]]:
    """
    Resolve a processed_files.csv row to an on-disk media file.

    Returns (path, modality, ssl_stream, is_cine, extra_source_meta) or None
    when no file can be resolved. Image rows whose PNG is missing fall back to
    a sibling MP4 with the same stem (Benin store often has videos only).
    """
    new_name = (row.get("New File Name") or "").strip()
    if not new_name:
        return None

    row_type = row_media_type(row)
    extra_meta: Dict[str, object] = {}

    if row_type == "video":
        media_path = videos_root / new_name
        if not media_path.exists():
            return None
        return media_path, "video", "both", True, extra_meta

    media_path = images_root / new_name
    if media_path.exists():
        return media_path, "image", "image", False, extra_meta

    video_fallback = videos_root / f"{Path(new_name).stem}.mp4"
    if video_fallback.exists():
        extra_meta["image_from_video"] = True
        extra_meta["still_image_name"] = new_name
        return video_fallback, "image", "image", False, extra_meta

    return None


class BeninLUSAdapter(BaseAdapter):
    """
    Adapter for the Benin lung ultrasound video dataset.
    """

    DATASET_ID = "Benin-LUS"
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

    # ── Metadata loading ───────────────────────────────────────────────────────

    def _load_metadata(self) -> None:
        processed_path = self.cleaned_root / "processed_files.csv"
        labels_path = self.cleaned_root / "labels_multidiagnosis.csv"

        if not processed_path.exists():
            raise FileNotFoundError(f"Benin-LUS processed_files.csv not found at {processed_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Benin-LUS labels_multidiagnosis.csv not found at {labels_path}")

        # Load processed_files rows — both video and image rows
        with processed_path.open() as f:
            reader = csv.DictReader(f)
            self._processed_rows = list(reader)

        # Load labels by patient (record_id)
        with labels_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("record_id")
                if not pid:
                    continue
                self._labels_by_patient[pid] = row

        # Pre-compute patient-level split (80/10/10 by sorted patient id)
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

    # ── Helper: build video-level label vector ────────────────────────────────

    @staticmethod
    def _build_video_labels(patient_row: dict, site: str) -> List[float]:
        """
        Build a length-7 multilabel vector for a given (patient, site).
        """
        vec = [0.0] * len(LUS_CANONICAL_FINDINGS)
        site_prefix = site.strip()
        for suffix, idx in _SUFFIX_TO_INDEX.items():
            col = f"{site_prefix}_{suffix}"
            if col in patient_row:
                try:
                    val = int(patient_row[col])
                except ValueError:
                    val = 0
                if val == 1:
                    vec[idx] = 1.0
        return vec

    # ── Main iterator ─────────────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[USManifestEntry]:
        videos_root = self.cleaned_root / "videos"
        images_root = self.cleaned_root / "images"

        for row in self._processed_rows:
            patient_id = row.get("Patient ID")
            if not patient_id:
                continue
            site = row.get("Site", "").strip()
            depth = row.get("Depth", "").strip()
            new_name = row.get("New File Name")
            if not new_name:
                continue

            resolved = resolve_lus_media(row, videos_root, images_root)
            if resolved is None:
                continue
            media_path, modality, ssl_stream, is_cine, media_meta = resolved

            split = self._patient_splits.get(patient_id, "train")

            # Patient-level labels
            label_row = self._labels_by_patient.get(patient_id)
            patient_labels = None
            video_labels: List[float] = []
            instances: List[Instance] = []

            if label_row is not None:
                try:
                    tb = int(label_row.get("TB Label", 0))
                except ValueError:
                    tb = 0
                try:
                    pneumonia = int(label_row.get("Pneumonia", 0))
                except ValueError:
                    pneumonia = 0
                try:
                    covid = int(label_row.get("Covid", 0))
                except ValueError:
                    covid = 0
                patient_labels = {
                    "tb": int(tb > 0),
                    "pneumonia": int(pneumonia > 0),
                    "covid": int(covid > 0),
                }

                # Video-level labels for this site
                video_labels = self._build_video_labels(label_row, site)

                # Create one instance per positive canonical finding
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

            # If we have no labels at all, treat as ssl_only
            has_labels = patient_labels is not None or video_labels
            task_type = "multilabel_cls" if has_labels else "ssl_only"

            source_meta = {
                "patient_id": patient_id,
                "site": site,
                "depth": depth,
            }
            source_meta.update(media_meta)
            if patient_labels is not None:
                source_meta["patient_labels"] = patient_labels
            if video_labels:
                source_meta["video_labels"] = video_labels

            yield self._make_entry(
                str(media_path),
                split=split,
                modality=modality,
                instances=instances,
                study_id=patient_id,
                view_type=site,
                is_cine=is_cine,
                has_temporal_order=(modality == "video"),
                task_type=task_type,
                ssl_stream=ssl_stream,
                is_promptable=False,
                source_meta=source_meta,
            )

