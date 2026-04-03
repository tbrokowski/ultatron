"""
data/adapters/cardiac/mimic_echo.py  ·  MIMIC-IV-ECHO adapter
==============================================================

MIMIC-IV-ECHO: ~525,000 echocardiogram DICOM clips from Beth Israel Deaconess.
  Labels: None (pure SSL — for labelled EF subsets use MIMICLVVolA4CAdapter).
  Format: .dcm files under a PhysioNet directory hierarchy.
  Access: PhysioNet credentialled access required.

Actual layout after wget download:
  {root}/physionet.org/files/mimic-iv-echo/1.0/
    echo-record-list.csv          ← authoritative index of all 525k records
    echo-study-list.csv
    structured-measurement.csv.gz
    files/
      p{prefix}/p{subject_id}/s{study_id}/{study_id}_{series}.dcm

The adapter is driven from echo-record-list.csv (not a disk scan) so the
manifest reflects the full intended dataset even when the download is
partial.  Files that do not yet exist on disk are still emitted — the
dataloader should handle missing-file errors gracefully at read time.
"""
from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_PHYSIONET_SUBPATH = Path("physionet.org") / "files" / "mimic-iv-echo" / "1.0"


class MIMICEchoAdapter(BaseAdapter):
    """
    MIMIC-IV-ECHO adapter — CSV-driven, full 525k entries.

    Reads echo-record-list.csv to enumerate every DICOM record.  The
    dicom_filepath column is a path relative to the 1.0 base directory,
    e.g. files/p10/p10002221/s94106955/94106955_0001.dcm.
    """

    DATASET_ID     = "MIMIC-IV-ECHO"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.13026/7rbq-q661"

    def _base_dir(self) -> Path:
        """Locate the 1.0 base directory regardless of how root was provided."""
        deep = self.root / _PHYSIONET_SUBPATH
        if deep.exists():
            return deep
        # User may have pointed directly at 1.0/
        if (self.root / "echo-record-list.csv").exists():
            return self.root
        return deep  # will raise FileNotFoundError below

    def iter_entries(self) -> Iterator[USManifestEntry]:
        base = self._base_dir()
        record_csv = base / "echo-record-list.csv"

        if not record_csv.exists():
            raise FileNotFoundError(
                f"MIMIC-IV-ECHO: echo-record-list.csv not found at {record_csv}.\n"
                "Expected layout: {root}/physionet.org/files/mimic-iv-echo/1.0/echo-record-list.csv"
            )

        # Read all rows first to get total count for split assignment
        with open(record_csv, newline="") as fh:
            rows = list(csv.DictReader(fh))

        n = len(rows)
        log.info(f"MIMIC-IV-ECHO: {n:,} records in echo-record-list.csv")

        for i, row in enumerate(rows):
            rel_path   = row["dicom_filepath"]          # e.g. files/p10/p10002221/s94106955/…
            abs_path   = base / rel_path
            study_id   = row["study_id"]
            subject_id = row["subject_id"]
            split      = self._infer_split(f"{subject_id}_{study_id}", i, n)

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
                    "acquisition_datetime": row.get("acquisition_datetime", ""),
                    "doi":                  self.DOI,
                },
            )
