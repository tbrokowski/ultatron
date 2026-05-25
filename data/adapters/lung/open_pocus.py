"""
data/adapters/lung/open_pocus.py  ·  OpenPOCUS adapter
=======================================================

OpenPOCUS — "Creation of an Open-Access Lung POCUS Image Database for
Deep Learning and Neural Network Applications"
Kumar, Nandakishore, Gordon et al., Stanford, 2025.

  1,874 lung POCUS video clips from 226 patients.
  Derived from a multi-centre prospective cohort (NCT04384055).
  Acquired 2020–2022, emergency departments, various POCUS devices.
  Standardised 12-zone (or modified 8-zone) scanning protocol.
  Frames extracted and resized to 128×128 px, stored as MP4.

  Findings per clip (multi-label classification):
    - normal (no abnormality)
    - b_lines         (discrete or confluent)
    - consolidation
    - b_lines_and_consolidation
    - indeterminate

GitHub : https://github.com/kumarandre/OpenPOCUS
DOI    : https://doi.org/10.1101/2025.05.09.25327337
SonoDQS: gold (multi-centre, 3 blinded expert raters, consensus)
Probe  : phased_array / curvilinear (mixed, various POCUS devices)

Dataset layout
--------------
  {root}/
    metadata.csv             ← per-clip labels + patient info
      expected columns:
        clip_id / filename, patient_id, lung_zone,
        finding / label (normal | b_lines | consolidation |
                         b_lines_and_consolidation | indeterminate),
        covid_positive, device, fps, num_frames
    clips/                   ← MP4 video clips
      {clip_id}.mp4

  Fallback: if no metadata.csv, all MP4s are yielded as ssl_only.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mpeg"}
_IMG_EXTS   = {".png", ".jpg", ".jpeg"}

# Raw finding → (label_raw, label_ontology)
_FINDING_MAP: dict[str, tuple[str, str]] = {
    "normal":                      ("normal",                     "lung_normal"),
    "no abnormality":              ("normal",                     "lung_normal"),
    "no_abnormality":              ("normal",                     "lung_normal"),
    "b_lines":                     ("b_lines",                    "lung_b_lines"),
    "b-lines":                     ("b_lines",                    "lung_b_lines"),
    "discrete b-lines":            ("b_lines_discrete",           "lung_b_lines"),
    "confluent b-lines":           ("b_lines_confluent",          "lung_b_lines"),
    "consolidation":               ("consolidation",              "lung_consolidation"),
    "b_lines_and_consolidation":   ("b_lines_and_consolidation",  "lung_b_lines"),
    "b-lines and consolidation":   ("b_lines_and_consolidation",  "lung_b_lines"),
    "indeterminate":               ("indeterminate",              "lung_other"),
}

_CSV_NAMES = (
    "metadata.csv", "labels.csv", "annotations.csv",
    "clip_metadata.csv", "data.csv",
)
_CLIP_DIR_NAMES = ("clips", "videos", "mp4", "data", ".")


def _find_csv(root: Path) -> Path | None:
    for name in _CSV_NAMES:
        p = root / name
        if p.exists():
            return p
    for p in root.glob("*.csv"):
        return p
    return None


def _find_clip_dir(root: Path) -> Path:
    for name in _CLIP_DIR_NAMES:
        d = root / name
        if d.is_dir() and any(
            f.suffix.lower() in _VIDEO_EXTS | _IMG_EXTS for f in d.iterdir()
        ):
            return d
    return root


def _is_media(p: Path) -> bool:
    return p.suffix.lower() in _VIDEO_EXTS | _IMG_EXTS


def _parse_finding(raw: str) -> tuple[str, str]:
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return _FINDING_MAP.get(
        key,
        _FINDING_MAP.get(raw.strip().lower(), ("unknown", "lung_other"))
    )


def _load_metadata(csv_path: Path) -> dict[str, dict]:
    """Load metadata CSV → {clip_id_stem: row_dict}."""
    meta: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Detect filename / clip_id column
            id_col = next(
                (k for k in row if k.lower().replace(" ", "_") in
                 ("clip_id", "filename", "file_name", "id", "video_id", "name")),
                None,
            )
            if id_col is None:
                continue
            stem = Path(row[id_col].strip()).stem
            meta[stem] = row
    return meta


class OpenPOCUSAdapter(BaseAdapter):
    """
    Adapter for the OpenPOCUS Stanford lung POCUS dataset (Kumar 2025).

    Yields one USManifestEntry per clip / frame image. When metadata.csv is
    present, lung findings and patient info are joined. Without CSV: ssl_only.

    Parameters
    ----------
    root : str | Path
        Root directory containing clips/ and metadata.csv.
    split_override : str, optional
        Force all entries to a single split.
    """

    DATASET_ID     = "OpenPOCUS"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.1101/2025.05.09.25327337"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        csv_path = _find_csv(self.root)
        clip_dir = _find_clip_dir(self.root)
        meta     = _load_metadata(csv_path) if csv_path else {}

        clips = sorted(f for f in clip_dir.iterdir() if _is_media(f))
        n     = len(clips)

        for i, clip_path in enumerate(clips):
            split = self._infer_split(clip_path.stem, i, n)

            row        = meta.get(clip_path.stem, {})
            has_meta   = bool(row)
            modality   = "video" if clip_path.suffix.lower() in _VIDEO_EXTS else "image"

            # Finding / label
            finding_col = next(
                (k for k in row if k.lower().replace(" ", "_") in
                 ("finding", "label", "class", "diagnosis", "pathology")),
                None,
            )
            raw_finding = row.get(finding_col, "").strip() if finding_col else ""
            label_raw, label_onto = _parse_finding(raw_finding) if raw_finding \
                else ("unknown", "lung_other")

            # Metadata fields
            patient_id  = row.get("patient_id", row.get("Patient_ID", ""))
            lung_zone   = row.get("lung_zone",  row.get("zone", ""))
            covid_pos   = row.get("covid_positive", row.get("COVID", ""))
            device      = row.get("device", row.get("Device", ""))
            num_frames  = int(row["num_frames"]) if "num_frames" in row and \
                          str(row["num_frames"]).isdigit() else 0

            instances = []
            if raw_finding and label_raw != "unknown":
                instances.append(self._make_instance(
                    instance_id    = clip_path.stem,
                    label_raw      = label_raw,
                    label_ontology = label_onto,
                    mask_path      = None,
                    is_promptable  = False,
                ))

            entry = self._make_entry(
                str(clip_path),
                split,
                modality      = modality,
                instances     = instances,
                has_mask      = False,
                task_type     = "classification" if raw_finding else "ssl_only",
                ssl_stream    = modality,
                is_promptable = False,
                probe_type    = "phased_array",
                source_meta   = {
                    "patient_id":   patient_id,
                    "lung_zone":    lung_zone,
                    "covid_pos":    covid_pos,
                    "device":       device,
                    "raw_finding":  raw_finding,
                    "doi":          self.DOI,
                },
            )
            if modality == "video" and num_frames > 0:
                entry.num_frames         = num_frames
                entry.has_temporal_order = True
            yield entry
