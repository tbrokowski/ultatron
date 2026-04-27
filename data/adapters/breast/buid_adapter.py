"""
data/adapters/breast/buid_adapter.py  ·  BUID breast ultrasound adapter
========================================================================
BUID (Breast Ultrasound Images Dataset): 232 images / 236 masks.
  Classes: benign, malignant
  Labels:  binary tumour segmentation masks (.tif)
  Source:  US-43d
  SonoDQS: silver

Dataset layout on disk
-----------------------
BUID/
├── Benign/
│   ├── 1 Benign Image.bmp     ← ultrasound image
│   ├── 1 Benign Lesion.bmp    ← visual annotation (ignored)
│   ├── 1 Benign Mask.tif      ← binary segmentation mask
│   └── ...
└── Malignant/
    ├── 1 Malignant Image.bmp
    ├── 1 Malignant Lesion.bmp
    ├── 1 Malignant Mask.tif
    └── ...

Key observations
----------------
- Label comes from the parent folder name (Benign / Malignant).
- 3 files per case: Image.bmp (used), Lesion.bmp (ignored), Mask.tif (used).
- Mask files are TIFF format.
- Filenames contain spaces: "1 Benign Image.bmp"
- Case number is the leading integer in the filename.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# Matches "1 Benign Image.bmp", "10 Malignant Image.bmp", etc.
_IMAGE_RE = re.compile(r"^(\d+)\s+\w+\s+Image\.bmp$", re.IGNORECASE)
_MASK_RE  = re.compile(r"^(\d+)\s+\w+\s+Mask\.tif$",  re.IGNORECASE)

# ── Label ontology mapping ─────────────────────────────────────────────────
CLASSES = {
    "benign":    ("benign_lesion",    "breast_lesion_benign"),
    "malignant": ("malignant_lesion", "breast_lesion_malignant"),
}


class BUIDAdapter(BaseAdapter):
    """
    Adapter for the BUID breast ultrasound dataset.

    Parameters
    ----------
    root : str | Path
        Root directory containing Benign/ and Malignant/ subdirectories.
    split_override : str, optional
        If set, all entries get this split label.
    """

    DATASET_ID     = "BUID"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1016/j.dib.2022.108437"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        """Yield one USManifestEntry per image across Benign/ and Malignant/."""

        all_samples = []

        for cls_name, (label_raw, label_ontology) in CLASSES.items():
            cls_dir = self._find_class_dir(cls_name)
            if cls_dir is None:
                continue

            # Build mask index: case_number → mask path
            mask_index: dict[str, Path] = {}
            for f in cls_dir.glob("*.tif"):
                m = _MASK_RE.match(f.name)
                if m:
                    mask_index[m.group(1)] = f

            # Collect images
            for f in sorted(cls_dir.glob("*.bmp")):
                m = _IMAGE_RE.match(f.name)
                if m:
                    case_num = m.group(1)
                    all_samples.append((f, mask_index.get(case_num), cls_name, label_raw, label_ontology))

        n = len(all_samples)

        for i, (img_path, mask_path, cls_name, label_raw, label_ontology) in enumerate(all_samples):
            has_mask = mask_path is not None
            split    = self._infer_split(img_path.stem, i, n)

            instance = self._make_instance(
                instance_id    = img_path.stem.replace(" ", "_"),
                label_raw      = label_raw,
                label_ontology = label_ontology,
                mask_path      = str(mask_path) if has_mask else None,
                is_promptable  = has_mask,
            )

            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = [instance],
                has_mask      = has_mask,
                task_type     = "segmentation" if has_mask else "classification",
                ssl_stream    = "image",
                is_promptable = has_mask,
                probe_type    = "linear",
                source_meta   = {
                    "cls_name": cls_name,
                    "doi":      self.DOI,
                },
            )

    def _find_class_dir(self, cls_name: str) -> Optional[Path]:
        """Find class directory supporting Benign/ or benign/ capitalisation."""
        for candidate in [
            self.root / cls_name.capitalize(),
            self.root / cls_name.lower(),
            self.root / cls_name.upper(),
        ]:
            if candidate.exists():
                return candidate
        return None
