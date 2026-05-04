"""
data/adapters/breast/bus_uc_adapter.py  ·  BUS-UC breast ultrasound adapter
============================================================================
BUS-UC (Breast Ultrasound UC dataset): 810 images / 791 masks.
  Classes: benign, malignant
  Labels:  binary tumour segmentation masks
  SonoDQS: silver

Dataset layout on disk
-----------------------
BUS-UC/
├── Benign/
│   ├── images/   ←  01.png, 02.png, 010.png, ...
│   └── masks/    ←  same filename as image
└── Malignant/
    ├── images/
    └── masks/

Key observations
----------------
- Label comes from the parent folder name (Benign / Malignant).
- Image/mask pairing is by identical filename.
- No CSV metadata file.
- All/ folder (union of Benign + Malignant) is ignored to avoid duplicates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# ── Label ontology mapping ─────────────────────────────────────────────────
# folder name (lowercased) → (label_raw, label_ontology)
CLASSES = {
    "benign":    ("benign_lesion",    "breast_lesion_benign"),
    "malignant": ("malignant_lesion", "breast_lesion_malignant"),
}


class BUSUCAdapter(BaseAdapter):
    """
    Adapter for the BUS-UC breast ultrasound dataset.

    Parameters
    ----------
    root : str | Path
        Root directory containing Benign/ and Malignant/ subdirectories.
    split_override : str, optional
        If set, all entries get this split label ("train"/"val"/"test").
    """

    DATASET_ID     = "BUS-UC"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1016/j.dib.2023.109473"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        """Yield one USManifestEntry per image across Benign/ and Malignant/."""

        all_samples = []

        for cls_name, (label_raw, label_ontology) in CLASSES.items():
            # Support both capitalised (Benign/) and lowercase (benign/) folders
            cls_dir = self._find_class_dir(cls_name)
            if cls_dir is None:
                continue

            images_dir = cls_dir / "images"
            masks_dir  = cls_dir / "masks"

            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.glob("*.png")):
                all_samples.append((img_path, masks_dir, cls_name, label_raw, label_ontology))

        n = len(all_samples)

        for i, (img_path, masks_dir, cls_name, label_raw, label_ontology) in enumerate(all_samples):
            mask_path = masks_dir / img_path.name
            has_mask  = mask_path.exists()
            split     = self._infer_split(img_path.stem, i, n)

            instance = self._make_instance(
                instance_id    = f"{cls_name}_{img_path.stem}",
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
