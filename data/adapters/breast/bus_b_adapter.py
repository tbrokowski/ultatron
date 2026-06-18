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
    DatasetB.xlsx          ← per-image metadata (Image, Type, Diagnosis, ...)
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

import logging
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterator, Sequence

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}

_NAME_COLS = ("image_name", "filename", "name", "image", "id")
# Prefer coarse lesion type (Benign/Malignant) over fine-grained diagnosis.
_LABEL_COLS = ("label", "class", "category", "type", "diagnosis")

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


def _parse_label_rows(rows: Sequence[Sequence]) -> dict[str, str]:
    """Parse spreadsheet rows into {image_stem: coarse_label}."""
    labels: dict[str, str] = {}
    if not rows:
        return labels

    headers = [
        str(h).strip().lower() if h is not None else f"col_{i}"
        for i, h in enumerate(rows[0])
    ]
    name_col = next(
        (i for i, h in enumerate(headers) if h in _NAME_COLS), 0
    )
    label_col = next(
        (i for i, h in enumerate(headers) if h in _LABEL_COLS), 1
    )

    for row in rows[1:]:
        if name_col >= len(row) or row[name_col] is None:
            continue
        stem = Path(str(row[name_col]).strip()).stem
        raw = row[label_col] if label_col < len(row) else None
        label = str(raw).strip().lower() if raw is not None else ""
        if label:
            labels[stem] = label
    return labels


def _load_xlsx_stdlib(xlsx: Path) -> list[tuple]:
    """
    Read the first worksheet from an .xlsx file using only the stdlib.

    Handles simple single-sheet workbooks such as DatasetB.xlsx.
    """
    with zipfile.ZipFile(xlsx) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("m:si", ns):
                text = si.find("m:t", ns)
                if text is not None and text.text is not None:
                    shared.append(text.text)
                else:
                    parts = [node.text or "" for node in si.findall(".//m:t", ns)]
                    shared.append("".join(parts))

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in zf.namelist():
            sheet_candidates = sorted(
                n for n in zf.namelist() if n.startswith("xl/worksheets/sheet")
            )
            if not sheet_candidates:
                return []
            sheet_name = sheet_candidates[0]

        sheet = ET.fromstring(zf.read(sheet_name))
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: list[tuple] = []
        for row_el in sheet.findall(".//m:row", ns):
            vals: list = []
            for cell in row_el.findall("m:c", ns):
                value_el = cell.find("m:v", ns)
                if value_el is None or value_el.text is None:
                    vals.append(None)
                elif cell.get("t") == "s":
                    vals.append(shared[int(value_el.text)])
                else:
                    vals.append(value_el.text)
            rows.append(tuple(vals))
        return rows


def _load_xlsx_labels(root: Path) -> dict[str, str]:
    """
    Load DatasetB.xlsx → {image_stem: coarse_label}.

    Tries openpyxl first, then a stdlib .xlsx reader. Logs and returns an
    empty dict only when the file is missing or unreadable.
    """
    xlsx = root / "DatasetB.xlsx"
    if not xlsx.exists():
        return {}

    rows: list[tuple] | None = None

    try:
        import openpyxl

        wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
    except ImportError:
        log.warning(
            "BUS-B: openpyxl not installed — reading %s via stdlib fallback",
            xlsx,
        )
    except Exception as exc:
        log.warning("BUS-B: openpyxl failed for %s: %s", xlsx, exc)

    if rows is None:
        try:
            rows = _load_xlsx_stdlib(xlsx)
        except Exception as exc:
            log.warning("BUS-B: failed to read %s: %s", xlsx, exc)
            return {}

    labels = _parse_label_rows(rows)
    if not labels:
        log.warning("BUS-B: no labels parsed from %s", xlsx)
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

        xlsx_labels = _load_xlsx_labels(self.root)

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

            raw_label = xlsx_labels.get(img_path.stem, "").lower()
            label_raw, label_onto = _LABEL_MAP.get(
                raw_label, ("unknown", "breast_lesion")
            )

            # Normal images typically have no mask.
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
                    "type_raw":  raw_label or None,
                },
            )
