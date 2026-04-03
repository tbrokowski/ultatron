"""
data/adapters/cardiac/mimic_lvvol_a4c.py  ·  MIMIC-IV-Echo-LVVol-A4C adapter
===============================================================================

MIMIC-IV-Echo-LVVol-A4C: 1,007 A4C echocardiography DICOM clips with
  expert LV volume measurements from Beth Israel Deaconess Medical Center.
  Labels:  LVEDV_A4C, LVESV_A4C, LVEF_A4C (biplane and A4C methods)
  Format:  DICOM (.dcm) files + FileList.csv
  Access:  PhysioNet credentialled access

  DICOM files are named by study_id (e.g. 90004661.dcm).
  The FileList.csv links study_id to LV volume measurements.

Directory layout:
  {root}/physionet.org/files/mimic-iv-echo-ext-lvvol-a4c/1.0.0/
      FileList.csv
      dicom/{study_id}.dcm
      masks/      (per-frame LV masks — may be incomplete in v1.0.0)
      npz/        (pre-computed feature arrays — may be incomplete in v1.0.0)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# Canonical sub-path inside the physionet download mirror
_PHYSIONET_SUBPATH = Path(
    "physionet.org/files/mimic-iv-echo-ext-lvvol-a4c/1.0.0"
)


class MIMICLVVolA4CAdapter(BaseAdapter):
    """
    MIMIC-IV-Echo-LVVol-A4C adapter.  Yields one video entry per DICOM file.

    Directory layout (physionet wget mirror):
        {root}/physionet.org/files/mimic-iv-echo-ext-lvvol-a4c/1.0.0/
            FileList.csv
            dicom/{study_id}.dcm
    """

    DATASET_ID     = "MIMIC-IV-Echo-LVVol-A4C"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.13026/4frm-we74"

    @property
    def _data_root(self) -> Path:
        return self.root / _PHYSIONET_SUBPATH

    def iter_entries(self) -> Iterator[USManifestEntry]:
        data_root     = self._data_root
        filelist_path = data_root / "FileList.csv"
        dicom_dir     = data_root / "dicom"

        if not filelist_path.exists():
            raise FileNotFoundError(
                f"MIMIC-IV-Echo-LVVol-A4C: FileList.csv not found at {data_root}.\n"
                "Run the download script: sbatch scripts/download_mimic_lvvol_a4c.sh"
            )

        with open(filelist_path) as f:
            rows = list(csv.DictReader(f))

        n = len(rows)
        for i, row in enumerate(rows):
            study_id  = row["study_id"]
            dcm_path  = dicom_dir / f"{study_id}.dcm"
            if not dcm_path.exists():
                continue

            split = self._infer_split(study_id, i, n)
            if self.split_override:
                split = self.split_override

            def _float(key: str) -> float:
                val = row.get(key, "")
                try:
                    return float(val) if val not in ("", "NA") else float("nan")
                except ValueError:
                    return float("nan")

            lvedv_a4c = _float("LVEDV_A4C")
            lvesv_a4c = _float("LVESV_A4C")
            lvef_a4c  = _float("LVEF_A4C")
            lvedv_bp  = _float("LVEDV_BP")
            lvesv_bp  = _float("LVESV_BP")
            lvef_bp   = _float("LVEF_BP")

            fps = float(row.get("frame_rate", 30.0) or 30.0)

            yield self._make_entry(
                str(dcm_path), split,
                modality           = "video",
                instances          = [],
                study_id           = study_id,
                view_type          = "A4C",
                is_cine            = True,
                has_temporal_order = True,
                fps                = fps,
                task_type          = "regression",
                ssl_stream         = "video",
                is_promptable      = False,
                has_mask           = False,
                source_meta        = {
                    "root":        str(self.root),
                    "doi":         self.DOI,
                    "patient_id":  row.get("patient_id", ""),
                    "study_id":    study_id,
                    "lvedv_a4c":   lvedv_a4c,
                    "lvesv_a4c":   lvesv_a4c,
                    "lvef_a4c":    lvef_a4c,
                    "lvedv_bp":    lvedv_bp,
                    "lvesv_bp":    lvesv_bp,
                    "lvef_bp":     lvef_bp,
                    "manufacturer": row.get("manufacturer", ""),
                    "n_frames":    row.get("number_of_frames", ""),
                },
            )
