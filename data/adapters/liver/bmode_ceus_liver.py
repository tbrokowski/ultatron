"""
data/adapters/liver/bmode_ceus_liver.py  - B-mode CEUS liver DICOM adapter
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class BModeCEUSLiverAdapter(BaseAdapter):
    DATASET_ID     = "B-mode-CEUS-liver"
    ANATOMY_FAMILY = "liver"
    SONODQS        = "silver"
    DOI            = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        series_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and not p.name.endswith(".zip")
        )
        n = len(series_dirs)
        for i, series_dir in enumerate(series_dirs):
            frames = sorted(series_dir.glob("*.dcm"))
            if not frames:
                continue

            split = self._infer_split(series_dir.name, i, n)
            yield self._make_entry(
                [str(f) for f in frames],
                split=split,
                modality="video",
                is_cine=True,
                has_temporal_order=True,
                task_type="ssl_only",
                ssl_stream="video",
                is_promptable=False,
                source_meta={"series_uid": series_dir.name, "n_frames": len(frames)},
            )
