"""
data/adapters/cardiac/mimic_echo.py  ·  MIMIC-IV-ECHO adapter
==============================================================

MIMIC-IV-ECHO: ~525,000 echocardiogram DICOM clips from Beth Israel Deaconess.
  Labels: None (pure SSL — for labelled EF subsets use MIMICLVVolA4CAdapter).
  Format: .dcm files under a PhysioNet directory hierarchy.
  Access: PhysioNet credentialled access required.

Actual layout after wget download:
  {root}/physionet.org/files/mimic-iv-echo/1.0/
    echo-record-list.csv          ← one row per .dcm (dicom_filepath, study_id, subject_id)
    echo-study-list.csv           ← study_id → measurement_id linkage (~99% coverage)
    structured_measurement.csv.gz ← long-format echo measurements keyed on measurement_id
    files/
      p{prefix}/p{subject_id}/s{study_id}/{study_id}_{series}.dcm

Each .dcm is a single multi-frame cine (one cardiac view = one temporal sequence).
The adapter is driven from echo-record-list.csv (not a disk scan) and emits only
DICOM files that are present on disk.  A partial download produces a partial
training manifest rather than runtime missing-file failures in the dataloader.

measurement_id is joined from echo-study-list.csv so that downstream tasks can
link manifest entries to structured echo measurements without re-reading CSVs.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_PHYSIONET_SUBPATH = Path("physionet.org") / "files" / "mimic-iv-echo" / "1.0"


class MIMICEchoAdapter(BaseAdapter):
    """
    MIMIC-IV-ECHO adapter — CSV-driven, available-file entries.

    Reads echo-record-list.csv to enumerate every DICOM record.  The
    dicom_filepath column is a path relative to the 1.0 base directory,
    e.g. files/p10/p10002221/s94106955/94106955_0001.dcm.  Missing files are
    skipped so generated training manifests are directly trainable.

    echo-study-list.csv is joined on study_id to populate measurement_id in
    source_meta, enabling downstream linkage to structured_measurement.csv.gz.
    """

    DATASET_ID     = "MIMIC-IV-ECHO"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.13026/nrjh-5r77"

    def _base_dir(self) -> Path:
        """Locate the 1.0 base directory regardless of how root was provided."""
        deep = self.root / _PHYSIONET_SUBPATH
        if deep.exists():
            return deep
        # User may have pointed directly at 1.0/
        if (self.root / "echo-record-list.csv").exists():
            return self.root
        return deep  # will raise FileNotFoundError below

    @staticmethod
    def _load_study_measurements(base: Path) -> Dict[str, str]:
        """
        Load echo-study-list.csv and return a study_id → measurement_id mapping.

        ~99% of DICOM studies in MIMIC-IV-ECHO have a corresponding structured
        measurement record within two days.  Studies without one map to "".
        """
        study_csv = base / "echo-study-list.csv"
        if not study_csv.exists():
            log.warning(
                "MIMIC-IV-ECHO: echo-study-list.csv not found at %s; "
                "measurement_id will be empty for all entries.",
                study_csv,
            )
            return {}
        mapping: Dict[str, str] = {}
        with open(study_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                sid = row.get("study_id", "").strip()
                mid = row.get("measurement_id", "").strip()
                if sid:
                    mapping[sid] = mid
        log.info(
            "MIMIC-IV-ECHO: loaded %d study→measurement links from echo-study-list.csv "
            "(%d with measurement_id)",
            len(mapping),
            sum(1 for v in mapping.values() if v),
        )
        return mapping

    def iter_entries(self) -> Iterator[USManifestEntry]:
        base = self._base_dir()
        record_csv = base / "echo-record-list.csv"

        if not record_csv.exists():
            raise FileNotFoundError(
                f"MIMIC-IV-ECHO: echo-record-list.csv not found at {record_csv}.\n"
                "Expected layout: {root}/physionet.org/files/mimic-iv-echo/1.0/echo-record-list.csv"
            )

        study_measurements = self._load_study_measurements(base)

        # Read all rows first to get total count for split assignment
        with open(record_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))

        n = len(rows)
        log.info("MIMIC-IV-ECHO: %d records in echo-record-list.csv", n)

        emitted = 0
        missing = 0
        for i, row in enumerate(rows):
            rel_path   = row["dicom_filepath"]          # e.g. files/p10/p10002221/s94106955/…
            abs_path   = base / rel_path
            if not abs_path.exists():
                missing += 1
                continue

            study_id      = row["study_id"]
            subject_id    = row["subject_id"]
            measurement_id = study_measurements.get(study_id, "")
            split         = self._infer_split(f"{subject_id}_{study_id}", i, n)

            emitted += 1
            yield self._make_entry(
                str(abs_path), split,
                modality           = "video",
                study_id           = study_id,
                is_cine            = True,
                has_temporal_order = True,
                fps                = 30.0,
                task_type          = "ssl_only",
                ssl_stream         = "video",
                is_promptable      = False,
                source_meta        = {
                    "subject_id":           subject_id,
                    "study_id":             study_id,
                    "measurement_id":       measurement_id,
                    "acquisition_datetime": row.get("acquisition_datetime", ""),
                    "doi":                  self.DOI,
                },
            )

        if missing:
            log.warning(
                "MIMIC-IV-ECHO: skipped %d missing DICOM records; emitted %d available records.",
                missing,
                emitted,
            )
