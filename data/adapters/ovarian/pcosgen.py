"""
data/adapters/ovarian/pcosgen.py  - PCOSGen ovarian ultrasound adapter

Capstor layout:
  updated test dataset/class label.csv + images/image#####.jpg  (test, extracted)
  10430727/PCOSGen-train.zip  (train images + class_label.xlsx, not extracted)
"""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _zip_image_path(zip_path: Path, member: str, extract_root: Optional[Path]) -> str:
    if extract_root is not None:
        candidate = extract_root / Path(member).name
        if len(Path(member).parts) > 1:
            candidate = extract_root / Path(*Path(member).parts[1:])
        if candidate.exists():
            return str(candidate)
    return f"{zip_path}::{member}"


def _load_xlsx_rows_from_bytes(data: bytes) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        ws = wb.active
        raw = list(ws.iter_rows(values_only=True))
        if not raw:
            return rows
        for row in raw[1:]:
            if row and len(row) >= 2 and row[0]:
                rows.append((str(row[0]).strip(), str(row[1]).strip()))
        return rows
    except ImportError:
        pass

    with zipfile.ZipFile(io.BytesIO(data)) as xz:
        sst = ET.fromstring(xz.read("xl/sharedStrings.xml"))
        strings: List[str] = []
        for si in sst.findall("m:si", _XLSX_NS):
            t = si.find(".//m:t", _XLSX_NS)
            strings.append(t.text if t is not None else "")

        sheet = ET.fromstring(xz.read("xl/worksheets/sheet1.xml"))
        for i, row in enumerate(sheet.findall(".//m:row", _XLSX_NS)):
            if i == 0:
                continue
            vals: List[str] = []
            for cell in row.findall("m:c", _XLSX_NS):
                v = cell.find("m:v", _XLSX_NS)
                if v is None:
                    vals.append("")
                elif cell.get("t") == "s":
                    vals.append(strings[int(v.text)])
                else:
                    vals.append(v.text or "")
            if len(vals) >= 2 and vals[0]:
                rows.append((vals[0].strip(), vals[1].strip()))
    return rows


class PCOSGenAdapter(BaseAdapter):
    DATASET_ID     = "PCOSGen"
    ANATOMY_FAMILY = "ovarian"
    SONODQS        = "bronze"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._test_root = self.root / "updated test dataset"
        self._train_zip = self.root / "10430727" / "PCOSGen-train.zip"

    @staticmethod
    def _parse_test_labels(row: dict) -> Tuple[int, int]:
        cols = list(row.keys())
        abn_raw = row[cols[1]].strip().lower()
        pcos_raw = row[cols[2]].strip().lower()
        abnormal = 1 if "abnormal" in abn_raw else 0
        pcos_visible = 1 if pcos_raw == "visible" else 0
        return abnormal, pcos_visible

    @staticmethod
    def _parse_train_healthy(value: str) -> int:
        return 1 if str(value).strip() in ("1", "1.0", "healthy", "Healthy") else 0

    def _yield_test_entries(self) -> Iterator[USManifestEntry]:
        csv_path = self._test_root / "class label.csv"
        img_dir = self._test_root / "images"
        if not csv_path.exists() or not img_dir.is_dir():
            return

        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                fname = row.get("imagePath", "").strip()
                if not fname:
                    continue
                img_path = img_dir / fname
                if not img_path.exists():
                    continue

                abnormal, pcos_visible = self._parse_test_labels(row)
                split = self.split_override or "test"
                instances: List[Instance] = [
                    self._make_instance(
                        instance_id=f"{img_path.stem}_abnormal",
                        label_raw="abnormal" if abnormal else "normal",
                        label_ontology="ovarian_appearance",
                        is_promptable=False,
                    ),
                    self._make_instance(
                        instance_id=f"{img_path.stem}_pcos",
                        label_raw="pcos_visible" if pcos_visible else "pcos_not_visible",
                        label_ontology="pcos_visibility",
                        is_promptable=False,
                    ),
                ]

                yield self._make_entry(
                    str(img_path),
                    split=split,
                    modality="image",
                    instances=instances,
                    task_type="multilabel_cls",
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={
                        "abnormal": abnormal,
                        "pcos_visible": pcos_visible,
                        "split_source": "updated_test_dataset",
                    },
                )

    def _yield_train_entries(self) -> Iterator[USManifestEntry]:
        if not self._train_zip.exists():
            log.warning("PCOSGen: train zip not found at %s", self._train_zip)
            return

        with zipfile.ZipFile(self._train_zip) as zf:
            xlsx_members = [n for n in zf.namelist() if n.endswith(".xlsx")]
            if not xlsx_members:
                log.warning("PCOSGen: no xlsx labels in %s", self._train_zip)
                return
            label_rows = _load_xlsx_rows_from_bytes(zf.read(xlsx_members[0]))
            members = {Path(n).name: n for n in zf.namelist() if n.endswith(".jpg")}

        extract_root = self.root / "10430727" / "PCOSGen-train"
        split = self.split_override or "train"

        for fname, healthy_raw in label_rows:
            member = members.get(fname)
            if not member:
                continue
            healthy = self._parse_train_healthy(healthy_raw)
            abnormal = 0 if healthy else 1
            img_path = _zip_image_path(self._train_zip, member, extract_root)

            instances: List[Instance] = [
                self._make_instance(
                    instance_id=f"{Path(fname).stem}_healthy",
                    label_raw="healthy" if healthy else "not_healthy",
                    label_ontology="ovarian_health",
                    is_promptable=False,
                ),
            ]

            yield self._make_entry(
                img_path,
                split=split,
                modality="image",
                instances=instances,
                task_type="binary_cls",
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "healthy": healthy,
                    "abnormal": abnormal,
                    "split_source": "PCOSGen-train.zip",
                    "zip_member": member,
                },
            )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        yield from self._yield_test_entries()
        yield from self._yield_train_entries()
