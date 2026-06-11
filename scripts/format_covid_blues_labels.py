#!/usr/bin/env python3
"""
format_covid_blues_labels.py  ·  Normalize COVID-BLUES CSV labels for Ultatron
===============================================================================

Merges severity.csv (per-video) and clinical_variables.csv (per-patient) into:
  {root}/metadata/video_labels.jsonl
  {root}/metadata/patient_splits.json

Splits are assigned at the **patient** level (80/10/10) per dataset authors'
recommendation not to mix frames/videos from the same patient across splits.

Source: https://github.com/NinaWie/COVID-BLUES
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_VIDEO_STEM_RE = re.compile(r"^patient_(\d+)_([LR]\d+)(?:_(\d+))?$", re.IGNORECASE)


def _parse_video_stem(stem: str) -> tuple[str, str, str | None]:
    m = _VIDEO_STEM_RE.match(stem.strip())
    if not m:
        raise ValueError(f"Unrecognized COVID-BLUES video stem: {stem!r}")
    patient_id, blue_point, duplicate = m.group(1), m.group(2).upper(), m.group(3)
    return patient_id, blue_point, duplicate


def _yes_no(val: str) -> bool | None:
    if val is None:
        return None
    v = str(val).strip().lower()
    if v in ("yes", "y", "1", "true"):
        return True
    if v in ("no", "n", "0", "false"):
        return False
    return None


def _load_clinical(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = str(row.get("patient_id", "")).strip()
            if pid:
                out[pid] = row
    return out


def _load_severity(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = str(row.get("video_file", "")).strip()
            if stem:
                rows.append(row)
    return rows


def _assign_patient_splits(patient_ids: list[str]) -> dict[str, str]:
    """Deterministic 80/10/10 split by sorted patient id."""
    patients = sorted(set(patient_ids), key=lambda x: int(x))
    n = len(patients)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    splits: dict[str, str] = {}
    for i, pid in enumerate(patients):
        if i < n_train:
            splits[pid] = "train"
        elif i < n_train + n_val:
            splits[pid] = "val"
        else:
            splits[pid] = "test"
    return splits


def format_labels(root: Path) -> dict[str, int]:
    root = root.resolve()
    severity_path = root / "severity.csv"
    clinical_path = root / "clinical_variables.csv"
    video_dir = root / "lus_videos"
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    if not severity_path.exists():
        raise FileNotFoundError(f"Missing {severity_path}")
    if not clinical_path.exists():
        raise FileNotFoundError(f"Missing {clinical_path}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Missing video directory {video_dir}")

    clinical = _load_clinical(clinical_path)
    severity_rows = _load_severity(severity_path)

    patient_ids = [pid for pid in clinical]
    patient_splits = _assign_patient_splits(patient_ids)

    written = 0
    skipped = 0
    out_jsonl = meta_dir / "video_labels.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for row in severity_rows:
            stem = Path(str(row["video_file"]).strip()).stem
            try:
                patient_id, blue_point, duplicate_idx = _parse_video_stem(stem)
            except ValueError:
                skipped += 1
                continue

            video_name = f"{stem}.mp4"
            video_path = video_dir / video_name
            if not video_path.is_file():
                skipped += 1
                continue

            clin = clinical.get(patient_id, {})
            cov_raw = clin.get("cov_test", "")
            try:
                cov_test = int(float(cov_raw)) if str(cov_raw).strip() != "" else None
            except (TypeError, ValueError):
                cov_test = None

            sev_raw = row.get("Severity Score", row.get("severity_score", ""))
            try:
                severity_score = float(sev_raw) if str(sev_raw).strip() != "" else None
            except (TypeError, ValueError):
                severity_score = None

            record = {
                "video_file": video_name,
                "video_path": str(video_path),
                "patient_id": patient_id,
                "blue_point": blue_point,
                "lung_side": "left" if blue_point.startswith("L") else "right",
                "duplicate_idx": int(duplicate_idx) if duplicate_idx else None,
                "severity_score": severity_score,
                "a_lines": _yes_no(row.get("A-lines", row.get("A_lines"))),
                "b_lines": _yes_no(row.get("B-lines", row.get("B_lines"))),
                "comments": (row.get("comments") or "").strip(),
                "cov_test": cov_test,
                "covid_positive": cov_test == 1 if cov_test is not None else None,
                "split": patient_splits.get(patient_id, "train"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    splits_path = meta_dir / "patient_splits.json"
    splits_path.write_text(json.dumps(patient_splits, indent=2), encoding="utf-8")

    return {
        "videos_written": written,
        "videos_skipped": skipped,
        "patients": len(patient_splits),
        "train": sum(1 for s in patient_splits.values() if s == "train"),
        "val": sum(1 for s in patient_splits.values() if s == "val"),
        "test": sum(1 for s in patient_splits.values() if s == "test"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Format COVID-BLUES labels for Ultatron")
    parser.add_argument(
        "--root",
        default="/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/COVID-BLUES",
        help="COVID-BLUES dataset root (contains lus_videos/, severity.csv, ...)",
    )
    args = parser.parse_args()
    stats = format_labels(Path(args.root))
    print("COVID-BLUES label formatting complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
