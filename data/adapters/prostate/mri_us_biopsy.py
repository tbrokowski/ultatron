"""
data/adapters/prostate/mri_us_biopsy.py  ·  Prostate MRI-US Biopsy adapter
============================================================================

TCIA / Kaggle prostate MRI-US biopsy collection. Expect transrectal ultrasound
DICOM series under the dataset root (MRI series pruned before staging).

Yields one entry per DICOM file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)


class ProstateMRIUSBiopsyAdapter(BaseAdapter):
    DATASET_ID = "Prostate-MRI-US-Biopsy"
    ANATOMY_FAMILY = "prostate"
    SONODQS = "silver"
    DOI = "https://www.kaggle.com/datasets/dsptlp/prostate-mri-us-biopsy"

    @staticmethod
    def _study_id(dcm_path: Path, root: Path) -> str:
        rel = dcm_path.relative_to(root)
        return rel.parts[0] if rel.parts else dcm_path.parent.name

    def iter_entries(self) -> Iterator[USManifestEntry]:
        dicoms = sorted(self.root.rglob("*.dcm"))
        if not dicoms:
            log.warning(
                "%s: no DICOM files found under %s",
                self.DATASET_ID,
                self.root,
            )
            return

        study_ids = sorted({self._study_id(p, self.root) for p in dicoms})
        study_splits = {
            sid: self._infer_split(sid, idx, len(study_ids))
            for idx, sid in enumerate(study_ids)
        }

        for dcm_path in dicoms:
            study_id = self._study_id(dcm_path, self.root)
            rel = dcm_path.relative_to(self.root)
            split = self.split_override or study_splits.get(study_id, "train")
            yield self._make_entry(
                str(dcm_path),
                split=split,
                modality="image",
                study_id=study_id,
                series_id=rel.parts[-2] if len(rel.parts) >= 2 else dcm_path.stem,
                task_type="ssl_only",
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "relative_path": str(rel),
                },
            )
