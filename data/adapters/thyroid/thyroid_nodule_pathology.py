"""
data/adapters/thyroid/thyroid_nodule_pathology.py  ·  Thyroid Nodule Pathology adapter
========================================================================================

Hou et al. 2024 — Figshare thyroid nodule pathology dataset (extracted).

Layout on Store:

    {root}/26067475/
      dataset/                         ← batch1 images (flat, Chinese filenames)
        殳永明_005.Jpg
        ...
      batch1_image.csv                 ← patient_name, path (filename only)
      batch1_image_label.csv           ← patient_index, patient_name, histo_label
      batch2_image/
        thyroid_3_10_month/
          612/                         ← patient_index subfolders
            ??????_004_174047.Jpg
            ...
        batch2_image.csv               ← path (relative from batch2_image/), patient_name
        batch2_image_label.csv         ← patient_index, patient_name, histo_label

histo_label: 0 = benign, 1 = malignant.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_LABEL_MAP = {0: "benign", 1: "malignant"}


def _find_data_root(root: Path) -> Path:
    """Return the 26067475/ directory regardless of where root points."""
    candidate = root / "26067475"
    if candidate.is_dir():
        return candidate
    if (root / "dataset").is_dir() or (root / "batch1_image.csv").exists():
        return root
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "batch1_image.csv").exists():
            return child
    return root


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _load_label_csv(path: Path, key: str) -> Dict[str, int]:
    """Return {key_value: histo_label} from a label CSV."""
    result: Dict[str, int] = {}
    if not path.exists():
        log.warning("ThyroidNodulePathology: label CSV not found at %s", path)
        return result
    for row in _read_csv(path):
        k = str(row.get(key, "")).strip()
        raw = str(row.get("histo_label", "")).strip()
        if k and raw.lstrip("-").isdigit():
            result[k] = int(raw)
    return result


class ThyroidNodulePathologyAdapter(BaseAdapter):
    DATASET_ID     = "thyroid-nodule-pathology"
    ANATOMY_FAMILY = "thyroid"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.6084/m9.figshare.26067475"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_root = _find_data_root(self.root)

    def iter_entries(self) -> Iterator[USManifestEntry]:
        yield from self._iter_batch1()
        yield from self._iter_batch2()

    # ── Batch 1 ───────────────────────────────────────────────────────────────

    def _iter_batch1(self) -> Iterator[USManifestEntry]:
        img_csv   = self._data_root / "batch1_image.csv"
        label_csv = self._data_root / "batch1_image_label.csv"
        img_dir   = self._data_root / "dataset"

        if not img_csv.exists():
            log.warning("ThyroidNodulePathology: batch1_image.csv not found under %s", self._data_root)
            return

        labels_by_name = _load_label_csv(label_csv, "patient_name")
        split = self.split_override or "train"

        for row in _read_csv(img_csv):
            patient_name = str(row.get("patient_name", "")).strip()
            filename     = str(row.get("path", "")).strip()
            if not filename:
                continue

            img_path = img_dir / filename
            if not img_path.exists():
                log.warning("ThyroidNodulePathology: batch1 image not found: %s", img_path)
                continue

            histo = labels_by_name.get(patient_name, -1)
            instances, task_type = self._make_cls_instance(
                instance_id=img_path.stem,
                histo=histo,
            )

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                study_id=img_path.stem,
                label_raw=[_LABEL_MAP[histo]] if histo in _LABEL_MAP else None,
                has_mask=False,
                has_box=False,
                has_temporal_order=False,
                num_frames=1,
                task_type=task_type,
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "batch":        "batch1",
                    "patient_name": patient_name,
                    "histo_label":  histo,
                },
            )

    # ── Batch 2 ───────────────────────────────────────────────────────────────

    def _iter_batch2(self) -> Iterator[USManifestEntry]:
        batch2_dir = self._data_root / "batch2_image"
        img_csv    = batch2_dir / "batch2_image.csv"
        label_csv  = batch2_dir / "batch2_image_label.csv"

        if not img_csv.exists():
            log.warning("ThyroidNodulePathology: batch2_image.csv not found under %s", batch2_dir)
            return

        labels_by_index = _load_label_csv(label_csv, "patient_index")
        split = self.split_override or "train"

        for row in _read_csv(img_csv):
            rel_path = str(row.get("path", "")).strip()
            if not rel_path:
                continue

            # patient_index is the first numeric folder component in the path
            # e.g. thyroid_3_10_month/612/filename.jpg → "612"
            patient_index = self._extract_patient_index(rel_path)

            img_path = batch2_dir / rel_path
            if not img_path.exists():
                log.warning("ThyroidNodulePathology: batch2 image not found: %s", img_path)
                continue

            histo = labels_by_index.get(patient_index, -1)
            instances, task_type = self._make_cls_instance(
                instance_id=f"{patient_index}_{img_path.stem}",
                histo=histo,
            )

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                study_id=patient_index or img_path.stem,
                label_raw=[_LABEL_MAP[histo]] if histo in _LABEL_MAP else None,
                has_mask=False,
                has_box=False,
                has_temporal_order=False,
                num_frames=1,
                task_type=task_type,
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "batch":          "batch2",
                    "patient_index":  patient_index,
                    "patient_name":   str(row.get("patient_name", "")).strip(),
                    "histo_label":    histo,
                    "rel_path":       rel_path,
                },
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_patient_index(rel_path: str) -> str:
        """Extract the first numeric path component (e.g. '612' from 'dir/612/file.jpg')."""
        for part in Path(rel_path).parts:
            if part.isdigit():
                return part
        return ""

    def _make_cls_instance(
        self,
        instance_id: str,
        histo: int,
    ) -> Tuple[list, str]:
        if histo not in _LABEL_MAP:
            return [], "ssl_only"
        label = _LABEL_MAP[histo]
        inst = self._make_instance(
            instance_id=instance_id,
            label_raw=label,
            label_ontology="thyroid_nodule_class",
            is_promptable=False,
            classification_label=histo,
        )
        return [inst], "classification"
