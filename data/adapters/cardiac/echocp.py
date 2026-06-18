"""
data/adapters/cardiac/echocp.py  ·  EchoCP adapter
=====================================================

EchoCP: 30-patient contrast transthoracic echocardiography dataset for
  Patent Foramen Ovale (PFO) diagnosis.
  Views:  Apical 4-chamber; two NIfTI volumes per patient (rest + Valsalva)
  Labels: PFO severity level (0 = no PFO, ≥1 = PFO present) from
          echoCP_diagnosis_label.xlsx
  Format: .nii.gz volumes + segmentation labels + echoCP_diagnosis_label.xlsx

Actual layout after extraction:
  {root}/
    echoCP_diagnosis_label.xlsx     # Idx, Action, PFO level
    EchoCP_dataset/
      001_r_image.nii.gz            # patient 001, rest
      001_r_label.nii.gz
      001_v_image.nii.gz            # patient 001, valsalva
      001_v_label.nii.gz
      002_r_image.nii.gz
      ...

Filename pattern: {idx:03d}_{action}_image.nii.gz
  action: "r" = rest, "v" = valsalva

Reference:
  Wei et al., "EchoCP: An Echocardiography Dataset for PFO Diagnosis",
  MICCAI 2021. https://doi.org/10.1007/978-3-030-87237-3_12
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import Instance, USManifestEntry

log = logging.getLogger(__name__)

_ACTION_MAP = {"r": "rest", "v": "valsalva"}


def _normalize_action(raw: object) -> Optional[str]:
    """Map xlsx/filename action codes to canonical rest | valsalva."""
    action = str(raw).strip().lower() if raw is not None else ""
    if action in ("r", "rest"):
        return "rest"
    if action in ("v", "valsalva"):
        return "valsalva"
    return action or None


def _parse_pfo_level(raw: object) -> int:
    """Parse PFO level; 'o' in the spreadsheet means no PFO (level 0)."""
    if raw is None:
        return -1
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip().lower()
    if text in ("o", "no", "none", ""):
        return 0
    try:
        return int(float(text))
    except ValueError:
        return -1


def _load_labels(root: Path) -> Dict[Tuple[str, str], int]:
    """
    Returns dict of (patient_idx_zfill3, action_full) -> pfo_level.
    e.g. ("001", "rest") -> 0

    Accepts action codes from the xlsx as either short (r/v) or full
    (rest/valsalva) so they align with volume filenames {idx}_{r|v}_image.nii.gz.
    """
    xlsx_path = root / "echoCP_diagnosis_label.xlsx"
    if not xlsx_path.exists():
        return {}

    try:
        import openpyxl
        wb   = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        ws   = wb.active
        rows = list(ws.iter_rows(values_only=True))
        wb.close()
    except ImportError:
        log.warning(
            "EchoCP: openpyxl not installed — cannot read %s; "
            "entries will be ssl_only",
            xlsx_path,
        )
        return {}
    except Exception as exc:
        log.warning("EchoCP: failed to read %s: %s", xlsx_path, exc)
        return {}

    labels: Dict[Tuple[str, str], int] = {}
    for row in rows[1:]:   # skip header
        if row[0] is None:
            continue
        idx         = str(int(row[0])).zfill(3)
        action_full = _normalize_action(row[1])
        if action_full is None:
            continue
        level = _parse_pfo_level(row[2])
        labels[(idx, action_full)] = level
    return labels


class EchoCPAdapter(BaseAdapter):
    """
    EchoCP adapter.  Yields one volume entry per *_image.nii.gz file.

    PFO level 0 → no PFO (label_raw = "no_pfo", classification_label = 0)
    PFO level ≥1 → PFO present (label_raw = "pfo", classification_label = 1)
    """

    DATASET_ID     = "EchoCP"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1007/978-3-030-87237-3_12"

    def _data_dir(self) -> Path:
        inner = self.root / "EchoCP_dataset"
        return inner if inner.exists() else self.root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        data_dir = self._data_dir()
        if not data_dir.exists():
            raise FileNotFoundError(
                f"EchoCP: dataset directory not found at {data_dir}.\n"
                "Extract the multi-part zip first:\n"
                f"  cd {self.root}\n"
                "  cp EchoCP_dataset.change2zip EchoCP_dataset.zip\n"
                "  zip -s 0 EchoCP_dataset.zip --out EchoCP_combined.zip\n"
                "  unzip EchoCP_combined.zip"
            )

        labels = _load_labels(self.root)

        image_files = sorted(data_dir.glob("*_image.nii.gz"))
        n           = len(image_files)

        for i, img_path in enumerate(image_files):
            split = self._infer_split(img_path.stem, i, n)
            if self.split_override:
                split = self.split_override

            # Parse filename: "001_r_image.nii.gz" → idx="001", action_code="r"
            parts = img_path.name.split("_")   # ["001", "r", "image.nii.gz"]
            pat_idx     = parts[0].zfill(3) if len(parts) >= 1 else "000"
            action_code = parts[1].lower()     if len(parts) >= 2 else "r"
            action_full = _ACTION_MAP.get(action_code, action_code)

            # Paired segmentation label
            mask_path: Optional[str] = None
            label_file = img_path.parent / img_path.name.replace("_image.nii.gz", "_label.nii.gz")
            if label_file.exists():
                mask_path = str(label_file)

            pfo_level = labels.get((pat_idx, action_full), -1)
            has_label = pfo_level >= 0

            instances: list = []
            if has_label:
                label_raw      = "no_pfo" if pfo_level == 0 else "pfo"
                label_ontology = "pfo_absent" if pfo_level == 0 else "pfo_present"
                cls_label      = 0 if pfo_level == 0 else 1
                inst_kwargs: dict = dict(
                    instance_id          = img_path.stem,
                    label_raw            = label_raw,
                    label_ontology       = label_ontology,
                    anatomy_family       = "cardiac",
                    classification_label = cls_label,
                    is_promptable        = mask_path is not None,
                )
                if mask_path:
                    inst_kwargs["mask_path"] = mask_path
                instances.append(Instance(**inst_kwargs))

            yield self._make_entry(
                str(img_path), split,
                modality           = "volume",
                instances          = instances,
                study_id           = pat_idx,
                view_type          = "A4C",
                is_cine            = True,
                has_temporal_order = True,
                task_type          = "classification" if has_label else "ssl_only",
                ssl_stream         = "video",
                is_promptable      = mask_path is not None,
                has_mask           = mask_path is not None,
                source_meta        = {
                    "root":       str(self.root),
                    "doi":        self.DOI,
                    "patient_id": pat_idx,
                    "action":     action_full,
                    "pfo_level":  pfo_level,
                },
            )
