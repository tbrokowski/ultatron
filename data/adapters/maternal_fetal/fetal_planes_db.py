"""
data/adapters/maternal_fetal/fetal_planes_db.py  ·  FETAL_PLANES_DB adapter
============================================================================

FETAL_PLANES_DB: 12,400 2-D maternal-fetal ultrasound frames from 1,792
patients, labelled by standard fetal plane (6 classes).

Layout on disk:
  {root}/
  ├── Images/
  │   └── Patient<id5>_Plane<N>_<k>_of_<total>.png   (RGBA, variable size)
  ├── FETAL_PLANES_DB_data.csv
  └── FETAL_PLANES_DB_data.xlsx

CSV (semicolon-delimited):
  Image_name   stem only (append .png for path)
  Patient_num  integer patient ID (1–1792)
  Plane        6-class label (see PLANE_LABELS)
  Brain_plane  fine-grained brain subtype (only meaningful for Fetal brain)
  Operator     Op. 1 / Op. 2 / Op. 3 / Other
  US_Machine   Aloka / Voluson E6 / Voluson S10 / Other
  Train        1 = train (7129 images), 0 = test (5271)

CSV parsing note:
  The delimiter is ';' and column headers have trailing spaces (e.g. 'Train ').
  All header names are stripped on load.

Split strategy:
  The Train column provides a pre-defined split — do not re-split randomly.
  split="train" if Train==1, split="test" if Train==0.  No val set in this
  dataset; carve one from train by Patient_num if cross-validation is needed.

Image format: RGBA (4 channels) — drop the alpha channel downstream.

Reference: https://zenodo.org/record/3904280
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.labels.label_interface import ANATOMY_LABEL_SPACES
from data.schema.manifest import USManifestEntry

# Canonical 6-class label space — order must match ANATOMY_LABEL_SPACES["fetal_planes"].
PLANE_LABELS: List[str] = ANATOMY_LABEL_SPACES["fetal_planes"]


class FetalPlanesDBAdapter(BaseAdapter):
    DATASET_ID     = "FETAL_PLANES_DB"
    ANATOMY_FAMILY = "fetal_planes"
    SONODQS        = "gold"
    DOI            = "https://zenodo.org/record/3904280"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )
        self._metadata_rows = self._load_csv()

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "Images").is_dir():
            return root
        for name in ("FETAL-PLANES-DB", "FETAL_PLANES_DB"):
            candidate = root / name
            if (candidate / "Images").is_dir():
                return candidate
        if root.is_dir():
            for candidate in sorted(root.iterdir()):
                if candidate.is_dir() and (candidate / "Images").is_dir():
                    return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'Images/' under {root}"
        )

    def _load_csv(self) -> list[Dict[str, str]]:
        """Load semicolon-delimited metadata with stripped headers and values."""
        csv_path = self.root / "FETAL_PLANES_DB_data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"FETAL_PLANES_DB: metadata CSV not found at {csv_path}"
            )

        import csv
        rows: list[Dict[str, str]] = []
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            try:
                headers = [h.strip() for h in next(reader)]
            except StopIteration:
                return rows
            for raw_row in reader:
                if not raw_row or not any(cell.strip() for cell in raw_row):
                    continue
                padded = raw_row + [""] * (len(headers) - len(raw_row))
                rows.append({
                    headers[i]: padded[i].strip()
                    for i in range(len(headers))
                })
        return rows

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self.root / "Images"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"FETAL_PLANES_DB: image directory not found at {img_dir}"
            )

        for row in self._metadata_rows:
            stem        = (row.get("Image_name") or "").strip()
            plane       = (row.get("Plane")       or "").strip()
            patient_num = (row.get("Patient_num") or "").strip()
            brain_plane = (row.get("Brain_plane") or "").strip()
            operator    = (row.get("Operator")    or "").strip()
            us_machine  = (row.get("US_Machine")  or "").strip()
            train_flag  = (row.get("Train")        or "").strip()

            if not stem:
                continue

            img_path = img_dir / f"{stem}.png"
            if not img_path.exists():
                continue

            try:
                cls_idx = PLANE_LABELS.index(plane)
            except ValueError:
                continue  # unknown plane — skip

            split = self.split_override or self._split_from_train_flag(train_flag)

            instance = self._make_instance(
                instance_id          = stem,
                label_raw            = plane,
                label_ontology       = "fetal_planes",
                is_promptable        = False,
                classification_label = cls_idx,
            )

            yield self._make_entry(
                str(img_path),
                split         = split,
                modality      = "image",
                instances     = [instance],
                study_id      = patient_num,
                series_id     = stem,
                view_type     = self._view_type(plane),
                has_mask      = False,
                task_type     = "classification",
                ssl_stream    = "image",
                is_promptable = False,
                source_meta   = {
                    "image_name":  stem,
                    "patient_num": patient_num,
                    "plane":       plane,
                    "brain_plane": brain_plane or None,
                    "operator":    operator or None,
                    "us_machine":  us_machine or None,
                    "train_flag":  train_flag or None,
                    "image_format": "rgba_png",
                },
            )

    @staticmethod
    def _split_from_train_flag(train_flag: str) -> str:
        if train_flag == "1":
            return "train"
        if train_flag == "0":
            return "test"
        return "unlabeled"

    @staticmethod
    def _view_type(plane: str) -> str:
        return plane.lower().replace(" ", "_") if plane else "fetal_plane_unknown"
