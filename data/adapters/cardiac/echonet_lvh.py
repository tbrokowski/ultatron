"""
data/adapters/cardiac/echonet_lvh.py  ·  EchoNet-LVH adapter
==============================================================

EchoNet-LVH: ~12,000 parasternal long-axis (PLAX) echocardiography videos
  from Stanford Medicine (2008-2020).
  Labels:  IVS, LVID, LVPW wall-thickness measurements at ED and ES
  Format:  .avi videos across Batch1–Batch4 + MeasurementsList.csv
  Split:   provided in MeasurementsList.csv ("split" column, lowercase)

Actual layout after extraction of EchoNet-LVH.zip:
  {root}/
    Batch1/*.avi
    Batch2/*.avi
    Batch3/*.avi
    Batch4/*.avi
    MeasurementsList.csv   columns: HashedFileName, Calc, CalcValue, Frame,
                                    X1, X2, Y1, Y2, Frames, FPS, Width,
                                    Height, split
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_BATCH_DIRS = ("Batch1", "Batch2", "Batch3", "Batch4")

# Numeric measurements stored in MeasurementsList.csv
_CALC_FIELDS = ("IVSd", "IVSs", "LVIDd", "LVIDs", "LVPWd", "LVPWs")


def _build_video_index(root: Path) -> Dict[str, Path]:
    """Map HashedFileName stem → absolute .avi path across all Batch dirs."""
    index: Dict[str, Path] = {}
    for batch in _BATCH_DIRS:
        bdir = root / batch
        if not bdir.exists():
            continue
        for p in bdir.glob("*.avi"):
            index[p.stem.upper()] = p
    return index


class EchoNetLVHAdapter(BaseAdapter):
    """
    EchoNet-LVH adapter.  Yields one video entry per .avi clip.

    The MeasurementsList.csv may contain multiple rows per video (one per
    measurement type).  All measurements for a video are collapsed into
    source_meta as averaged CalcValues keyed by Calc name (lowercase).
    """

    DATASET_ID     = "EchoNet-LVH"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1038/s41746-022-00698-5"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        mfile = self.root / "MeasurementsList.csv"
        if not mfile.exists():
            raise FileNotFoundError(
                f"EchoNet-LVH: MeasurementsList.csv not found at {self.root}.\n"
                "The dataset zip has not been extracted yet.  Run:\n"
                f"  cd {self.root}\n"
                "  unzip EchoNet-LVH.zip"
            )

        video_index = _build_video_index(self.root)

        # Group rows by HashedFileName; first row per file also carries split
        rows_by_file: Dict[str, List[dict]] = defaultdict(list)
        split_by_file: Dict[str, str] = {}
        fps_by_file:   Dict[str, float] = {}

        with open(mfile) as f:
            for row in csv.DictReader(f):
                key = row.get("HashedFileName", "").strip().upper()
                if not key:
                    continue
                rows_by_file[key].append(row)
                if key not in split_by_file:
                    split_by_file[key] = row.get("split", "train").lower()
                    try:
                        fps_by_file[key] = float(row.get("FPS", 25.0))
                    except (ValueError, TypeError):
                        fps_by_file[key] = 25.0

        for key, rows in rows_by_file.items():
            vpath = video_index.get(key)
            if vpath is None:
                continue

            split = self.split_override or split_by_file.get(key, "train")

            # Aggregate measurements: average CalcValue per Calc name
            calc_vals: Dict[str, List[float]] = defaultdict(list)
            for r in rows:
                calc  = r.get("Calc", "").strip()
                val_s = r.get("CalcValue", "")
                if calc and val_s:
                    try:
                        calc_vals[calc.lower()].append(float(val_s))
                    except ValueError:
                        pass

            source_meta: dict = {
                "root": str(self.root),
                "doi":  self.DOI,
            }
            for calc, vals in calc_vals.items():
                source_meta[calc] = sum(vals) / len(vals)

            has_measurements = bool(calc_vals)
            instances: list = []
            if has_measurements:
                instances.append(Instance(
                    instance_id    = key,
                    label_raw      = "LV_wall_thickness",
                    label_ontology = "lv_wall_measurement",
                    anatomy_family = "cardiac",
                    is_promptable  = False,
                ))

            yield self._make_entry(
                str(vpath), split,
                modality           = "video",
                instances          = instances,
                study_id           = key,
                view_type          = "PLAX",
                is_cine            = True,
                has_temporal_order = True,
                fps                = fps_by_file.get(key, 25.0),
                task_type          = "regression" if has_measurements else "ssl_only",
                ssl_stream         = "both",
                is_promptable      = False,
                has_mask           = False,
                source_meta        = source_meta,
            )
