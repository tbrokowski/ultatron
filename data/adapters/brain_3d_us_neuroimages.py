"""
data/adapters/brain_3d_us_neuroimages.py  ·  3D brain ultrasound volumes
============================================================================

3D-US-Neuroimages-Dataset contains standalone intraoperative brain ultrasound
volumes stored as NRRD files. The dataset is unlabeled and used as SSL-only
brain data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class ThreeDUSNeuroimagesAdapter(BaseAdapter):
    DATASET_ID = "3D-US-Neuroimages-Dataset"
    ANATOMY_FAMILY = "brain"
    SONODQS = "silver"
    DOI = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        volumes = sorted(self.root.glob("*.nrrd"))
        split_map = self._group_split_map(self._study_id(p.stem) for p in volumes)

        for vol_path in volumes:
            study_id = self._study_id(vol_path.stem)
            yield self._make_entry(
                str(vol_path),
                split=split_map.get(study_id, "train"),
                modality="volume",
                study_id=study_id,
                series_id=vol_path.stem,
                is_3d=True,
                view_type="intraoperative_3d",
                task_type="ssl_only",
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "study_id": study_id,
                    "file_name": vol_path.name,
                },
            )

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    @staticmethod
    def _study_id(stem: str) -> str:
        return stem.split("_", 1)[0]
