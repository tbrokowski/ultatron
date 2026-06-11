"""
data/adapters/liver/behsof.py  - BEHSOF liver steatosis adapter

Capstor layout:
  {version}/image_Data.zip   BEH#####/Image*.jpg
  {version}/BEHSOF_images.csv  Patient ID, Steatosis stage, Fibroscan F
"""
from __future__ import annotations

import csv
import logging
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)


def _norm_patient_id(pid: str) -> str:
    return pid.replace("\\Data\\", "").replace("/Data/", "").strip()


def _zip_image_path(zip_path: Path, member: str, extract_root: Path) -> str:
    rel = Path(member)
    candidate = extract_root / rel
    if candidate.exists():
        return str(candidate)
    return f"{zip_path}::{member}"


class BEHSOFAdapter(BaseAdapter):
    DATASET_ID     = "BEHSOF"
    ANATOMY_FAMILY = "liver"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_root = self._find_data_root(self.root)
        self._labels = self._load_labels()
        self._patient_splits = self._build_patient_splits()

    @staticmethod
    def _find_data_root(root: Path) -> Path:
        if (root / "BEHSOF_images.csv").exists():
            return root
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / "BEHSOF_images.csv").exists():
                return child
        return root

    def _load_labels(self) -> Dict[str, dict]:
        csv_path = self._data_root / "BEHSOF_images.csv"
        labels: Dict[str, dict] = {}
        if not csv_path.exists():
            log.warning("BEHSOF: BEHSOF_images.csv not found at %s", csv_path)
            return labels

        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                pid = _norm_patient_id(row.get("Patient ID", ""))
                if not pid:
                    continue
                labels[pid] = {
                    "steatosis_stage": int(row.get("Steatosis stage", -1)),
                    "fibroscan_f": int(row.get("Fibroscan F", -1)),
                }
        return labels

    def _build_patient_splits(self) -> Dict[str, str]:
        patients = sorted(self._labels.keys())
        n = len(patients)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        splits: Dict[str, str] = {}
        for i, pid in enumerate(patients):
            if self.split_override:
                splits[pid] = self.split_override
            elif i < n_train:
                splits[pid] = "train"
            elif i < n_train + n_val:
                splits[pid] = "val"
            else:
                splits[pid] = "test"
        return splits

    def _iter_image_refs(self) -> Iterator[Tuple[str, str]]:
        extract_root = self._data_root / "image_Data"
        zip_path = self._data_root / "image_Data.zip"
        resolved: Dict[str, str] = {}

        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                for member in sorted(zf.namelist()):
                    if not member.lower().endswith(".jpg"):
                        continue
                    patient_id = member.split("/")[0]
                    resolved[member] = _zip_image_path(zip_path, member, extract_root)

        if extract_root.is_dir():
            for img_path in sorted(extract_root.rglob("*.jpg")):
                patient_id = img_path.parent.name
                rel = f"{patient_id}/{img_path.name}"
                resolved[rel] = str(img_path)

        for member, img_path in sorted(resolved.items()):
            patient_id = member.split("/")[0]
            yield patient_id, img_path

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._labels:
            return

        seen: set[str] = set()
        for patient_id, img_path in self._iter_image_refs():
            if img_path in seen:
                continue
            seen.add(img_path)

            meta = self._labels.get(patient_id)
            if meta is None:
                continue

            stage = meta["steatosis_stage"]
            split = self._patient_splits.get(patient_id, "train")
            stem = Path(img_path.split("::")[-1]).stem
            instance_id = f"{patient_id}_{stem}"

            instances: List[Instance] = [
                self._make_instance(
                    instance_id=instance_id,
                    label_raw=f"steatosis_stage_{stage}",
                    label_ontology="liver_steatosis",
                    is_promptable=False,
                )
            ]

            yield self._make_entry(
                img_path,
                split=split,
                modality="image",
                instances=instances,
                study_id=patient_id,
                task_type="multiclass_cls",
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "patient_id": patient_id,
                    "steatosis_stage": stage,
                    "fibroscan_f": meta["fibroscan_f"],
                },
            )
