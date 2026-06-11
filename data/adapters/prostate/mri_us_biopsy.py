"""
data/adapters/prostate/mri_us_biopsy.py  ·  Prostate MRI-US Biopsy adapter
============================================================================

Kaggle dsptlp/prostate-mri-us-biopsy dataset. After download and MRI pruning,
only transrectal ultrasound DICOM series remain under the dataset root.

Yields one entry per DICOM file once the download completes.
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
    DOI = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        dicoms = sorted(self.root.rglob("*.dcm"))
        if not dicoms:
            log.warning(
                "%s: no DICOM files found under %s — run download_prostate_mri_us_biopsy.sh",
                self.DATASET_ID,
                self.root,
            )
            return

        study_ids = sorted({p.parent.name for p in dicoms})
        study_splits = {
            sid: self._infer_split(sid, idx, len(study_ids))
            for idx, sid in enumerate(study_ids)
        }

        for dcm_path in dicoms:
            study_id = dcm_path.parent.name
            split = self.split_override or study_splits.get(study_id, "train")
            yield self._make_entry(
                str(dcm_path),
                split=split,
                modality="image",
                study_id=study_id,
                series_id=dcm_path.stem,
                task_type="ssl_only",
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "relative_path": str(dcm_path.relative_to(self.root)),
                },
            )
