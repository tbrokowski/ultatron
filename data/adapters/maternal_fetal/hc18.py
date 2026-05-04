"""
data/adapters/maternal_fetal/hc18.py  ·  HC18 fetal-head adapter
=================================================================

HC18: Head Circumference 18-Week Dataset (Grand Challenge).
  999 training + 335 test 2D ultrasound images of the fetal head.
  Training images are paired with expert ellipse-contour masks and a CSV
  providing ground-truth head circumference (mm) + pixel size (mm/px).

Layout on disk:
  {root}/
  ├── training_set/
  │   ├── <id>_HC.png              ← ultrasound image
  │   ├── <id>_2HC.png             ← second sweep of same subject
  │   ├── <id>_3HC.png             ← third sweep (rare)
  │   └── <id>_HC_Annotation.png  ← ellipse contour mask (nonzero = foreground)
  ├── test_set/
  │   └── <id>_HC.png              ← ultrasound image, no annotations
  ├── training_set_pixel_size_and_HC.csv
  └── test_set_pixel_size.csv

Annotation note:
  Masks contain values {0, 1, 255} — an ellipse outline, NOT a filled region.
  1 and 255 mark inner/outer contour edges (~1120 nonzero pixels per image).
  Any nonzero pixel is foreground.  source_meta["annotation_type"] = "contour"
  flags this for downstream loaders.  Image sizes are mostly 800×540 but ~24
  annotations vary slightly — dimensions must not be hardcoded.

Split strategy:
  Patient ID = numeric prefix from stem (1_HC → "1", 1_2HC → "1").
  Training entries are split into train/val by patient; test entries are "test".
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_PATIENT_RE = re.compile(r"^(\d+)_")


class HC18Adapter(BaseAdapter):
    DATASET_ID     = "HC18"
    ANATOMY_FAMILY = "fetal_head"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.1371/journal.pone.0200412"

    _TRAIN_DIRS = ("training_set", "training")
    _TEST_DIRS  = ("test_set",     "test")

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )
        self._train_csv = self._load_train_csv()
        self._test_csv  = self._load_test_csv()

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "training_set").is_dir() or (root / "training").is_dir():
            return root
        candidate = root / "HC18"
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'training_set/' under {root}"
        )

    def _first_existing(self, names: Tuple[str, ...]) -> Path:
        for n in names:
            d = self.root / n
            if d.exists():
                return d
        return self.root / names[0]

    def _load_train_csv(self) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Return {filename: (pixel_size_mm, hc_mm)}."""
        csv_path = self.root / "training_set_pixel_size_and_HC.csv"
        out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        if not csv_path.exists():
            return out
        with csv_path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fname = (row.get("filename") or "").strip()
                px_s  = self._first_value(row, "pixel size(mm)", "pixel size (mm)", "pixel size")
                hc_s  = self._first_value(row, "head circumference (mm)", "head circumference(mm)")
                if not fname:
                    continue
                out[fname] = (self._to_float(px_s), self._to_float(hc_s))
        return out

    def _load_test_csv(self) -> Dict[str, Optional[float]]:
        """Return {filename: pixel_size_mm}."""
        csv_path = self.root / "test_set_pixel_size.csv"
        out: Dict[str, Optional[float]] = {}
        if not csv_path.exists():
            return out
        with csv_path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                fname = (row.get("filename") or "").strip()
                px_s  = self._first_value(row, "pixel size(mm)", "pixel size (mm)", "pixel size")
                if not fname:
                    continue
                out[fname] = self._to_float(px_s)
        return out

    def iter_entries(self) -> Iterator[USManifestEntry]:
        train_dir = self._first_existing(self._TRAIN_DIRS)
        test_dir  = self._first_existing(self._TEST_DIRS)

        if not train_dir.exists() and not test_dir.exists():
            raise FileNotFoundError(
                f"HC18: no data directory found under {self.root}"
            )

        # ── Training set ──────────────────────────────────────────────────────
        if train_dir.exists():
            images = sorted(
                p for p in train_dir.glob("*.png")
                if "_Annotation" not in p.stem
            )
            # Patient-level split to prevent leakage across repeated measurements.
            patient_ids = [self._patient_id(p.stem) for p in images]
            patients    = sorted(set(patient_ids))
            patient_split: Dict[str, str] = {
                pid: self._infer_split(pid, i, len(patients))
                for i, pid in enumerate(patients)
            }

            for img_path in images:
                stem       = img_path.stem
                patient_id = self._patient_id(stem)
                mask_path  = train_dir / f"{stem}_Annotation.png"
                has_mask   = mask_path.exists()

                px_mm, hc_mm = self._train_csv.get(img_path.name, (None, None))
                split = self.split_override or patient_split.get(patient_id, "train")

                if has_mask:
                    instance = self._make_instance(
                        instance_id    = stem,
                        label_raw      = "fetal_head_circumference",
                        label_ontology = "head_circumference",
                        mask_path      = str(mask_path),
                        is_promptable  = True,
                        measurement_mm = hc_mm,
                    )
                    task_type = "regression"
                else:
                    instance = self._make_instance(
                        instance_id    = stem,
                        label_raw      = "fetal_head_circumference",
                        label_ontology = "head_circumference",
                        is_promptable  = False,
                        measurement_mm = hc_mm,
                    )
                    task_type = "ssl_only"

                yield self._make_entry(
                    str(img_path),
                    split         = split,
                    modality      = "image",
                    instances     = [instance],
                    study_id      = patient_id,
                    series_id     = stem,
                    view_type     = "fetal_head_standard_plane",
                    has_mask      = has_mask,
                    task_type     = task_type,
                    ssl_stream    = "image",
                    is_promptable = has_mask,
                    source_meta   = {
                        "pixel_size_mm":   px_mm,
                        "hc_mm":           hc_mm,
                        "patient_id":      patient_id,
                        "annotation_type": "contour" if has_mask else None,
                    },
                )

        # ── Test set (no labels) ──────────────────────────────────────────────
        if test_dir.exists():
            for img_path in sorted(test_dir.glob("*.png")):
                stem       = img_path.stem
                patient_id = self._patient_id(stem)
                px_mm      = self._test_csv.get(img_path.name)
                split      = self.split_override or "test"

                yield self._make_entry(
                    str(img_path),
                    split         = split,
                    modality      = "image",
                    instances     = [],
                    study_id      = patient_id,
                    series_id     = stem,
                    view_type     = "fetal_head_standard_plane",
                    has_mask      = False,
                    task_type     = "ssl_only",
                    ssl_stream    = "image",
                    is_promptable = False,
                    source_meta   = {
                        "pixel_size_mm": px_mm,
                        "patient_id":    patient_id,
                    },
                )

    @staticmethod
    def _patient_id(stem: str) -> str:
        """Extract numeric patient prefix: '001_2HC' → '001'."""
        m = _PATIENT_RE.match(stem)
        return m.group(1) if m else stem

    @staticmethod
    def _first_value(row: dict, *keys: str) -> str:
        for k in keys:
            v = row.get(k, "")
            if v:
                return v.strip()
        return ""

    @staticmethod
    def _to_float(s: str) -> Optional[float]:
        try:
            return float(s) if s else None
        except ValueError:
            return None
