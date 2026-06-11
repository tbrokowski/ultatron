"""
data/adapters/lung/benin_videos.py  - BeninVideos SSL adapter

Capstor layout:
  lung/BeninVideos/{PatientID}_{Site}_{Depth}_{Count}.mp4  (flat, no labels)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class BeninVideosAdapter(BaseAdapter):
    DATASET_ID     = "BeninVideos"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._patient_splits = self._build_patient_splits()

    @staticmethod
    def _parse_filename(stem: str) -> Optional[Tuple[str, str, str, str]]:
        parts = stem.rsplit("_", 3)
        if len(parts) != 4:
            return None
        patient_id, site, depth, count = parts
        if not depth.isdigit() or not count.isdigit():
            return None
        return patient_id, site, depth, count

    def _build_patient_splits(self) -> Dict[str, str]:
        patients = sorted({
            parsed[0]
            for vpath in self.root.glob("*.mp4")
            if (parsed := self._parse_filename(vpath.stem)) is not None
        })
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

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for vpath in sorted(self.root.glob("*.mp4")):
            parsed = self._parse_filename(vpath.stem)
            if parsed is None:
                continue
            patient_id, site, depth, count = parsed
            split = self._patient_splits.get(patient_id, "train")

            yield self._make_entry(
                str(vpath),
                split=split,
                modality="video",
                study_id=patient_id,
                view_type=site,
                is_cine=True,
                has_temporal_order=True,
                task_type="ssl_only",
                ssl_stream="both",
                is_promptable=False,
                source_meta={
                    "patient_id": patient_id,
                    "site": site,
                    "depth": depth,
                    "count": count,
                },
            )
