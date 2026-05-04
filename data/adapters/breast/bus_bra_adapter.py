"""
data/adapters/breast/bus_bra_adapter.py  ·  BUS-BRA breast ultrasound adapter
==============================================================================
BUS-BRA (Gomez-Flores 2024): 1,875 images + 1,875 masks.
  Classes: benign, malignant
  Labels:  binary tumour segmentation masks
  Source:  https://zenodo.org/record/8231412
  SonoDQS: gold  (public, well-curated, expert labels, pre-defined CV splits)

Dataset layout on disk
-----------------------
BUS-BRA/
├── annotations.csv       ← metadata: image_filename, pathology, birads
├── 5-fold-cv.csv         ← pre-defined 5-fold cross-validation splits (optional)
├── 10-fold-cv.csv        ← pre-defined 10-fold cross-validation splits (optional)
├── images/
│   ├── img_0000.png
│   └── ...
└── masks/
    ├── img_0000.png      ← same filename as corresponding image
    └── ...
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# ── Label ontology mapping ─────────────────────────────────────────────────
# raw CSV "pathology" value → (label_raw, label_ontology)
CLASSES = {
    "benign":    ("benign_lesion",    "breast_lesion_benign"),
    "malignant": ("malignant_lesion", "breast_lesion_malignant"),
}


class BUSBRAAdapter(BaseAdapter):
    """
    Adapter for the BUS-BRA breast ultrasound dataset (Gomez-Flores 2024).

    Parameters
    ----------
    root : str | Path
        Root directory containing images/, masks/, and annotations.csv.
    split_override : str, optional
        If set, all entries get this split label ("train"/"val"/"test").
    fold_csv : str | Path, optional
        Path to a fold CSV for pre-defined splits.
    fold_index : int, optional
        Which fold column to use as val (0-based). Only used with fold_csv.
    """

    DATASET_ID     = "BUS-BRA"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "gold"
    DOI            = "https://zenodo.org/record/8231412"

    def __init__(
        self,
        root: str | Path,
        split_override: Optional[str] = None,
        fold_csv: Optional[str | Path] = None,
        fold_index: Optional[int] = None,
    ):
        super().__init__(root=root, split_override=split_override)

        self.images_dir      = self.root / "images"
        self.masks_dir       = self.root / "masks"
        self.annotations_csv = self.root / "annotations.csv"
        self.fold_csv        = Path(fold_csv) if fold_csv else None
        self.fold_index      = fold_index

        self._meta        = self._load_metadata()
        self._fold_splits = self._load_fold_splits() if (
            self.fold_csv and self.fold_csv.exists() and fold_index is not None
        ) else {}

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_metadata(self) -> dict[str, dict]:
        """
        Read annotations.csv → dict keyed by image_filename.
        Expected columns: image_filename, pathology, birads
        """
        meta: dict[str, dict] = {}

        if not self.annotations_csv.exists():
            return meta

        with open(self.annotations_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("image_filename", "").strip()
                if not fname:
                    continue
                meta[fname] = {
                    "pathology": row.get("pathology", "").strip(),
                    "birads":    row.get("birads", "").strip(),
                }

        return meta

    def _load_fold_splits(self) -> dict[str, str]:
        """Read fold CSV → mapping filename → "train" | "val"."""
        splits: dict[str, str] = {}

        with open(self.fold_csv, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            fold_col = next(
                (h for h in headers if str(self.fold_index) in h.lower()), None
            )
            if fold_col is None:
                return splits

            img_col = next(
                (c for c in headers if c.lower() in {"image_filename", "filename", "image"}),
                headers[0] if headers else None,
            )

            for row in reader:
                if not img_col:
                    continue
                fname = row[img_col].strip()
                splits[fname] = "val" if row[fold_col].strip().lower() == "test" else "train"

        return splits

    def _resolve_split(self, fname: str, idx: int, total: int) -> str:
        if self.split_override:
            return self.split_override
        if self._fold_splits:
            return self._fold_splits.get(fname, "train")
        return self._infer_split(fname, idx, total)

    # ── Public interface ───────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[USManifestEntry]:
        """Yield one USManifestEntry per image in images/."""

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"BUS-BRA: images/ directory not found under {self.root}"
            )

        all_images = sorted(self.images_dir.glob("*.png"))
        n = len(all_images)

        for i, img_path in enumerate(all_images):
            fname = img_path.name

            # ── Metadata ──────────────────────────────────────────────────
            row_meta  = self._meta.get(fname, {})
            pathology = row_meta.get("pathology", "")
            birads    = row_meta.get("birads") or None

            # ── Split ─────────────────────────────────────────────────────
            split = self._resolve_split(fname, i, n)

            # ── Mask ──────────────────────────────────────────────────────
            mask_path = self.masks_dir / fname
            has_mask  = mask_path.exists()

            # ── Ontology + task ───────────────────────────────────────────
            if pathology in CLASSES:
                label_raw, label_ontology = CLASSES[pathology]
                task_type = "segmentation" if has_mask else "classification"
            else:
                label_raw, label_ontology = "unknown", "unknown"
                task_type = "ssl_only"

            # ── Instance ──────────────────────────────────────────────────
            instance = self._make_instance(
                instance_id    = img_path.stem,
                label_raw      = label_raw,
                label_ontology = label_ontology,
                mask_path      = str(mask_path) if has_mask else None,
                is_promptable  = has_mask,
            )

            # ── Entry ─────────────────────────────────────────────────────
            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = [instance],
                has_mask      = has_mask,
                task_type     = task_type,
                ssl_stream    = "image",
                is_promptable = has_mask,
                probe_type    = "linear",
                source_meta   = {
                    "birads":        birads,
                    "original_name": fname,
                    "doi":           self.DOI,
                },
            )
