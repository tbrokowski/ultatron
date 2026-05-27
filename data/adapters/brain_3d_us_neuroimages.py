"""
data/adapters/brain_3d_us_neuroimages.py  ·  3D brain ultrasound volumes
============================================================================

3D-US-Neuroimages-Dataset contains standalone intraoperative brain ultrasound
volumes stored as NRRD files. The dataset is unlabeled and used as SSL-only
brain data.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)


class ThreeDUSNeuroimagesAdapter(BaseAdapter):
    DATASET_ID = "3D-US-Neuroimages-Dataset"
    ANATOMY_FAMILY = "brain"
    SONODQS = "silver"
    DOI = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        volumes = sorted(self.root.glob("*.nrrd"))
        if volumes:
            log.warning(
                "3D-US-Neuroimages-Dataset: skipped %d NRRD (.nrrd) files because "
                "the training loader does not support NRRD yet.",
                len(volumes),
            )
        return iter(())

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    @staticmethod
    def _study_id(stem: str) -> str:
        return stem.split("_", 1)[0]
