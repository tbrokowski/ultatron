"""
data/adapters/remind_brain_ius.py  ·  ReMIND raw intraoperative brain US
============================================================================

REMIND-Brain-iUS stores mixed MRI, US, and SEG DICOM series per case. For the
foundation manifest we keep only the ultrasound DICOM objects and ignore the
MRI/SEG modalities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class REMINDBrainIUSAdapter(BaseAdapter):
    DATASET_ID = "REMIND-Brain-iUS"
    ANATOMY_FAMILY = "brain"
    SONODQS = "gold"
    DOI = "https://doi.org/10.7937/3RAG-D070"

    def __init__(self, root, split_override=None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "remind").exists():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "remind").exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected 'remind/' under {root}")

    def iter_entries(self) -> Iterator[USManifestEntry]:
        case_dirs = sorted(p for p in (self.root / "remind").iterdir() if p.is_dir())
        split_map = self._group_split_map(case_dir.name for case_dir in case_dirs)

        for case_dir in case_dirs:
            case_id = case_dir.name
            split = split_map.get(case_id, "train")
            study_dirs = sorted(p for p in case_dir.iterdir() if p.is_dir())
            for study_dir in study_dirs:
                us_series_dirs = sorted(
                    p for p in study_dir.iterdir()
                    if p.is_dir() and p.name.startswith("US_")
                )
                for series_dir in us_series_dirs:
                    dcm_files = sorted(series_dir.glob("*.dcm"))
                    for dcm_path in dcm_files:
                        yield self._make_entry(
                            str(dcm_path),
                            split=split,
                            modality="volume",
                            study_id=case_id,
                            series_id=series_dir.name,
                            is_3d=True,
                            view_type="intraoperative_us_dicom",
                            task_type="ssl_only",
                            ssl_stream="image",
                            is_promptable=False,
                            source_meta={
                                "case_id": case_id,
                                "study_uid": study_dir.name,
                                "series_uid": series_dir.name,
                            },
                        )

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }
