"""
data/adapters/echonet.py  ·  EchoNet-Dynamic adapter
==========================================================

EchoNet-Dynamic: 10,030 labelled echocardiogram videos
  Labels: LV volume tracings, ejection fraction (EF%)
  Format: .avi videos + FileList.csv + VolumeTracings.csv
  Split:  provided in FileList.csv (TRAIN / VAL / TEST)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class EchoNetDynamicAdapter(BaseAdapter):
    """
    EchoNet-Dynamic adapter.

    Directory layout:
        {root}/Videos/*.avi
        {root}/FileList.csv       (FileName, Split, EF, ESV, EDV, FrameHeight, FrameWidth, FPS)
        {root}/VolumeTracings.csv (FileName, X1, Y1, X2, Y2, Frame)
    """

    DATASET_ID     = "EchoNet-Dynamic"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1016/j.cell.2021.03.056"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        filelist_path = self.root / "FileList.csv"
        tracings_path = self.root / "VolumeTracings.csv"

        # Load volume tracings (LV contour keyframes)
        tracings: Dict[str, List[dict]] = {}
        if tracings_path.exists():
            with open(tracings_path) as f:
                for row in csv.DictReader(f):
                    tracings.setdefault(row["FileName"], []).append(row)

        with open(filelist_path) as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            fname = row["FileName"]
            if not fname.endswith(".avi"):
                fname += ".avi"
            vpath = self.root / "Videos" / fname
            if not vpath.exists():
                continue

            split = (self.split_override or row.get("Split", "TRAIN")).upper()
            split = {"TRAIN": "train", "VAL": "val", "TEST": "test"}.get(split, "train")

            ef   = float(row.get("EF", 0.0))
            fps  = float(row.get("FPS", 25.0))
            esv  = float(row.get("ESV", 0.0))
            edv  = float(row.get("EDV", 0.0))

            # Instance for LV contour tracing (if available)
            instances = []
            if fname in tracings:
                instances.append(Instance(
                    instance_id    = fname,
                    label_raw      = "LV_contour",
                    label_ontology = "lv_segmentation",
                    anatomy_family = "cardiac",
                    is_promptable  = True,
                ))

            yield self._make_entry(
                str(vpath), split,
                modality           = "video",
                instances          = instances,
                study_id           = fname.replace(".avi", ""),
                view_type          = "A4C",
                is_cine            = True,
                has_temporal_order = True,
                fps                = fps,
                task_type          = "regression",
                # Include EchoNet videos in BOTH streams so Phase 3 can sample
                # frames (image stream) paired with clips (video stream).
                ssl_stream         = "both",
                is_promptable      = bool(instances),
                has_mask           = bool(instances),
                source_meta        = {
                    "root": str(self.root),
                    "doi":  self.DOI,
                    "ef":   ef,
                    "esv":  esv,
                    "edv":  edv,
                },
            )
