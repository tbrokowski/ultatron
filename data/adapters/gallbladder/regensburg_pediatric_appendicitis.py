"""
data/adapters/gallbladder/regensburg_pediatric_appendicitis.py
==============================================================

Regensburg Pediatric Appendicitis Dataset
Marcinkevičs et al., Medical Image Analysis, 2024.

  579 pediatric patients, 1–15 B-mode US views each (~2097 images total).
  Three target variables per subject:
    - diagnosis  : appendicitis | no_appendicitis
    - management : surgical | conservative
    - severity   : complicated | uncomplicated | no_appendicitis
  Accompanying: lab values, clinical scores (Alvarado, PAS), US findings.

Files
-----
  US_Pictures/
    {subject_id}.{view_idx}.bmp   e.g. 23.7.bmp
  app_data.xlsx                   per-subject labels + clinical data
  test_set_codes.csv              subject IDs for the test split

DOI     : https://doi.org/10.5281/zenodo.7711412
SonoDQS : gold (multi-centre, expert-labelled, three label levels)
Probe   : curvilinear (abdominal)
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif"}

# Filename prefix: {subject_id}.{view_idx} (optionally followed by a view label)
_FNAME_RE = re.compile(r"^(?P<subject_id>\d+)\.(?P<view_idx>\d+)")

_ID_COL_NAMES = ("us_number", "subject_id", "id", "subjectid", "patient_id")
_SHEET_CANDIDATES = ("All cases", "all cases")

# Diagnosis label → label_ontology
_DIAGNOSIS_MAP = {
    "appendicitis":      "appendicitis",
    "no_appendicitis":   "no_appendicitis",
    "no appendicitis":   "no_appendicitis",
    "1":                 "appendicitis",
    "0":                 "no_appendicitis",
}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMG_EXTS


def _parse_stem(stem: str) -> tuple[str | None, str | None]:
    """Return (subject_id, view_idx) from filename stem."""
    prefix = stem.split()[0]
    m = _FNAME_RE.match(prefix)
    if m:
        return m.group("subject_id"), m.group("view_idx")
    return None, None


def _iter_image_paths(root: Path) -> list[Path]:
    """Collect US images from US_Pictures/, including nested archive layouts."""
    us_dir = root / "US_Pictures"
    if not us_dir.is_dir():
        return sorted(f for f in root.iterdir() if _is_image(f))

    nested = us_dir / "US_Pictures"
    for candidate in (nested, us_dir):
        if candidate.is_dir():
            direct = sorted(f for f in candidate.iterdir() if _is_image(f))
            if direct:
                return direct

    return sorted(p for p in us_dir.rglob("*") if p.is_file() and _is_image(p))


def _load_test_codes(root: Path) -> set[str]:
    """Load test subject IDs from test_set_codes.csv."""
    test_codes: set[str] = set()
    csv_path = root / "test_set_codes.csv"
    if not csv_path.exists():
        return test_codes
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                cell = cell.strip()
                if cell.isdigit():
                    test_codes.add(cell)
    return test_codes


def _load_app_data(root: Path) -> dict[str, dict]:
    """
    Load app_data.xlsx into {subject_id_str: {col: value}}.
    Falls back to app_data.csv if openpyxl not available.
    """
    meta: dict[str, dict] = {}

    xlsx_path = root / "app_data.xlsx"
    csv_path  = root / "app_data.csv"

    if xlsx_path.exists():
        try:
            import openpyxl
            wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
            ws = next(
                (wb[name] for name in _SHEET_CANDIDATES if name in wb.sheetnames),
                wb.active,
            )
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                wb.close()
                return meta
            headers = [str(h).strip() if h is not None else f"col_{i}"
                       for i, h in enumerate(rows[0])]
            id_col = next(
                (i for i, h in enumerate(headers)
                 if h.lower().replace(" ", "_") in _ID_COL_NAMES),
                0,
            )
            for row in rows[1:]:
                if row[id_col] is None:
                    continue
                sid = (
                    str(int(row[id_col]))
                    if isinstance(row[id_col], float)
                    else str(row[id_col]).strip()
                )
                meta[sid] = {headers[i]: row[i] for i in range(len(headers))}
            wb.close()
            return meta
        except Exception as exc:
            log.warning("Failed to read %s via openpyxl, falling back to CSV: %s", xlsx_path, exc)

    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid_col = next(
                    (k for k in row
                     if k.lower().replace(" ", "_") in _ID_COL_NAMES),
                    None,
                )
                if sid_col:
                    sid = row[sid_col].strip()
                    meta[sid] = dict(row)

    return meta


def _extract_label(subject_meta: dict, col_names: list[str]) -> str | None:
    """Try a list of column name variants and return the first non-null value."""
    for col in col_names:
        for key in subject_meta:
            if key.lower().replace(" ", "_").replace("-", "_") == col.lower():
                val = subject_meta[key]
                if val is not None and str(val).strip():
                    return str(val).strip().lower()
    return None


class RegensburgPediatricAppendicitisAdapter(BaseAdapter):
    """
    Adapter for the Regensburg Pediatric Appendicitis dataset.

    Yields one USManifestEntry per US image (view). Labels from app_data.xlsx
    are joined by subject_id. Each entry carries:
    - task_type = "classification"
    - instances: one Instance per subject with the diagnosis label
    - source_meta: diagnosis, management, severity, subject_id, view_idx

    Parameters
    ----------
    root : str | Path
        Root directory containing US_Pictures/, app_data.xlsx,
        test_set_codes.csv.
    split_override : str, optional
        Force all entries to a single split.
    """

    DATASET_ID     = "RegensburgPedAppend"
    ANATOMY_FAMILY = "abdomen"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.5281/zenodo.7711412"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        test_codes = _load_test_codes(self.root)
        app_meta   = _load_app_data(self.root)

        imgs = _iter_image_paths(self.root)
        n    = len(imgs)

        for i, img_path in enumerate(imgs):
            subject_id, view_idx = _parse_stem(img_path.stem)

            # Split assignment
            if self.split_override:
                split = self.split_override
            elif subject_id and subject_id in test_codes:
                split = "test"
            else:
                # Hash-based train/val from non-test subjects
                split = self._infer_split(img_path.stem, i, n)

            # Clinical labels from app_data
            smeta = app_meta.get(subject_id or "", {})

            diagnosis  = _extract_label(smeta, ["diagnosis", "Diagnosis"]) or "unknown"
            management = _extract_label(smeta, ["management", "Management"])
            severity   = _extract_label(smeta, ["severity", "Severity"])

            label_raw  = _DIAGNOSIS_MAP.get(diagnosis.lower(), diagnosis)
            label_onto = (
                "appendicitis"
                if label_raw == "appendicitis"
                else "no_appendicitis"
            )

            instance = self._make_instance(
                instance_id    = f"{subject_id}_{view_idx}",
                label_raw      = label_raw,
                label_ontology = label_onto,
                mask_path      = None,
                is_promptable  = False,
            )

            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = [instance],
                has_mask      = False,
                task_type     = "classification",
                ssl_stream    = "image",
                is_promptable = False,
                probe_type    = "curvilinear",
                source_meta   = {
                    "subject_id":  subject_id,
                    "view_idx":    view_idx,
                    "diagnosis":   diagnosis,
                    "management":  management,
                    "severity":    severity,
                    "doi":         self.DOI,
                },
            )
