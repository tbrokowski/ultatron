"""
data/adapters/breast/bus_b_adapter.py  ·  Breast US B Dataset adapter
======================================================================

Breast US B Dataset — Al-Dhabyani et al. (2020), Data in Brief.
  310 B-mode breast ultrasound images + binary segmentation masks.
  Labels: benign | malignant | normal (stored in DatasetB.xlsx).
  Probe : linear.

Dataset layout
--------------
  {root}/
    DatasetB.xlsx          ← per-image metadata (image_name, label, ...)
    original/
      000001.png           ← US images (zero-padded 6-digit index)
      000002.png
      ...
    GT/
      000001.png           ← binary masks (same filename as image)
      000002.png
      ...

DOI     : https://doi.org/10.1016/j.dib.2019.104863
SonoDQS : silver (single-centre, single rater, 310 images)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}

# Label mapping: raw xlsx value → (label_raw, label_ontology)
_LABEL_MAP: dict[str, tuple[str, str]] = {
    "benign":    ("benign_lesion",    "breast_lesion_benign"),
    "malignant": ("malignant_lesion", "breast_lesion_malignant"),
    "normal":    ("normal",           "breast_normal"),
    "b":         ("benign_lesion",    "breast_lesion_benign"),
    "m":         ("malignant_lesion", "breast_lesion_malignant"),
    "n":         ("normal",           "breast_normal"),
    "0":         ("benign_lesion",    "breast_lesion_benign"),
    "1":         ("malignant_lesion", "breast_lesion_malignant"),
}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMG_EXTS


def _load_xlsx_labels(root: Path) -> dict[str, str]:
    """
    Load DatasetB.xlsx → {image_stem: label_raw}.
    Returns empty dict if openpyxl not available.
    """
    labels: dict[str, str] = {}
    xlsx = root / "DatasetB.xlsx"
    if not xlsx.exists():
        return labels
    try:
        import openpyxl
        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return labels
        headers = [str(h).strip().lower() if h is not None else f"col_{i}"
                   for i, h in enumerate(rows[0])]

        # Find name and label columns
        name_col = next(
            (i for i, h in enumerate(headers)
             if h in ("image_name", "filename", "name", "image", "id")), 0
        )
        label_col = next(
            (i for i, h in enumerate(headers)
             if h in ("label", "class", "category", "type", "diagnosis")), 1
        )

        for row in rows[1:]:
            if row[name_col] is None:
                continue
            stem  = Path(str(row[name_col]).strip()).stem
            label = str(row[label_col]).strip().lower() if row[label_col] else ""
            if label:
                labels[stem] = label
        wb.close()
    except Exception:
        pass
    return labels


class BUSBAdapter(BaseAdapter):
    """
    Adapter for the Breast US B Dataset (Al-Dhabyani 2020).

    Yields one USManifestEntry per image:
    - task_type = "segmentation" when GT mask exists (benign / malignant)
    - task_type = "classification" when no mask (normal)
    - Label from DatasetB.xlsx when available; falls back to "unknown".

    Parameters
    ----------
    root : str | Path
        Root directory containing DatasetB.xlsx, original/, and GT/.
    split_override : str, optional
        Force all entries to a single split.
    """

    DATASET_ID     = "BUS-B"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1016/j.dib.2019.104863"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir  = self.root / "original"
        mask_dir = self.root / "GT"

        if not img_dir.is_dir():
            img_dir = self.root   # fallback

        # Load xlsx metadata
        xlsx_labels = _load_xlsx_labels(self.root)

        # Build mask index: stem → path
        mask_index: dict[str, Path] = {}
        if mask_dir.is_dir():
            for f in mask_dir.iterdir():
                if _is_image(f):
                    mask_index[f.stem] = f

        imgs = sorted(f for f in img_dir.iterdir() if _is_image(f))
        n    = len(imgs)

        for i, img_path in enumerate(imgs):
            split     = self._infer_split(img_path.stem, i, n)
            mask_path = mask_index.get(img_path.stem)
            has_mask  = mask_path is not None

            # Resolve label
            raw_label = xlsx_labels.get(img_path.stem, "").lower()
            if not raw_label:
                # Infer from mask presence as fallback
                raw_label = "benign" if has_mask else "normal"
            label_raw, label_onto = _LABEL_MAP.get(
                raw_label, ("unknown", "breast_lesion")
            )

            # Normal images typically have no mask
            is_normal = label_raw == "normal"
            has_mask  = has_mask and not is_normal

            instance = self._make_instance(
                instance_id    = img_path.stem,
                label_raw      = label_raw,
                label_ontology = label_onto,
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
                    "doi":       self.DOI,
                    "label_raw": label_raw,
                },
            )
