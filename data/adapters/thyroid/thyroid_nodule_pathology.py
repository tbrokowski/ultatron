"""
data/adapters/thyroid/thyroid_nodule_pathology.py  ·  Thyroid Nodule Pathology adapter
======================================================================================

Figshare thyroid nodule pathology dataset stored as zip archives:

    {root}/26067475/
        batch1_image.zip              unlabelled thyroid US images
        batch2_image.zip              images + batch2_image_label.csv (histo_label)

Images remain inside zip archives; manifest paths use zip::member notation unless
extracted to a sibling directory.
"""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _zip_image_path(zip_path: Path, member: str, extract_root: Optional[Path]) -> str:
    if extract_root is not None:
        rel = Path(*Path(member).parts[1:]) if len(Path(member).parts) > 1 else Path(member).name
        candidate = extract_root / rel
        if candidate.exists():
            return str(candidate)
        candidate = extract_root / Path(member).name
        if candidate.exists():
            return str(candidate)
    return f"{zip_path}::{member}"


def _is_image_member(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(_IMAGE_EXTS) and not lower.endswith("/")


class ThyroidNodulePathologyAdapter(BaseAdapter):
    DATASET_ID = "Thyroid-Nodule-Pathology"
    ANATOMY_FAMILY = "thyroid"
    SONODQS = "silver"
    DOI = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._archive_root = self._find_archive_root(self.root)

    @staticmethod
    def _find_archive_root(root: Path) -> Path:
        if (root / "26067475").is_dir():
            return root / "26067475"
        if any(root.glob("batch*_image.zip")):
            return root
        for child in sorted(root.iterdir()):
            if child.is_dir() and any(child.glob("batch*_image.zip")):
                return child
        return root

    def _load_batch2_labels(self, zip_path: Path) -> Dict[str, dict]:
        labels: Dict[str, dict] = {}
        if not zip_path.exists():
            return labels
        with zipfile.ZipFile(zip_path) as zf:
            csv_members = [
                n for n in zf.namelist()
                if n.lower().endswith("_label.csv") or n.lower().endswith("label.csv")
            ]
            if not csv_members:
                return labels
            text = zf.read(csv_members[0]).decode("utf-8", errors="replace")
            for row in csv.DictReader(io.StringIO(text)):
                pid = str(row.get("patient_index", "")).strip()
                if not pid:
                    continue
                histo = str(row.get("histo_label", "")).strip()
                labels[pid] = {
                    "patient_name": row.get("patient_name", ""),
                    "histo_label": int(histo) if histo.isdigit() else -1,
                }
        return labels

    def _yield_zip_images(
        self,
        zip_path: Path,
        labels: Dict[str, dict],
        subset: str,
    ) -> Iterator[USManifestEntry]:
        extract_root = zip_path.parent / zip_path.stem
        patient_ids = sorted(labels.keys()) if labels else []
        patient_splits = {
            pid: self._infer_split(pid, idx, len(patient_ids))
            for idx, pid in enumerate(patient_ids)
        }

        with zipfile.ZipFile(zip_path) as zf:
            members = sorted(n for n in zf.namelist() if _is_image_member(n))
            for member in members:
                parts = Path(member).parts
                patient_id = None
                for part in parts:
                    if part.isdigit():
                        patient_id = part
                        break

                if labels and patient_id in labels:
                    label_info = labels[patient_id]
                    split = patient_splits.get(patient_id, "train")
                    histo = label_info["histo_label"]
                    instances = [
                        self._make_instance(
                            instance_id=patient_id,
                            label_raw="malignant" if histo == 1 else "benign",
                            label_ontology="thyroid_nodule_class",
                            is_promptable=False,
                        )
                    ]
                    task_type = "classification"
                else:
                    split = self.split_override or "train"
                    instances = []
                    task_type = "ssl_only"

                img_path = _zip_image_path(zip_path, member, extract_root)
                yield self._make_entry(
                    img_path,
                    split=split,
                    modality="image",
                    instances=instances,
                    task_type=task_type,
                    ssl_stream="image",
                    is_promptable=bool(instances),
                    study_id=patient_id or Path(member).stem,
                    source_meta={
                        "subset": subset,
                        "zip_member": member,
                        "patient_index": patient_id,
                    },
                )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        batch1 = self._archive_root / "batch1_image.zip"
        batch2 = self._archive_root / "batch2_image.zip"

        if batch1.exists():
            yield from self._yield_zip_images(batch1, labels={}, subset="batch1")
        else:
            log.warning("Thyroid-Nodule-Pathology: batch1_image.zip not found under %s", self._archive_root)

        if batch2.exists():
            labels = self._load_batch2_labels(batch2)
            yield from self._yield_zip_images(batch2, labels=labels, subset="batch2")
        else:
            log.warning("Thyroid-Nodule-Pathology: batch2_image.zip not found under %s", self._archive_root)
