"""
data/adapters/resect.py  ·  RESECT intraoperative brain ultrasound adapter
============================================================================

RESECT provides pre-, during-, and post-resection 3D ultrasound NIfTI volumes
for each case plus MRI and landmark files. The adapter emits only the US
volumes for SSL pretraining.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class RESECTAdapter(BaseAdapter):
    DATASET_ID = "RESECT"
    ANATOMY_FAMILY = "brain"
    SONODQS = "gold"
    DOI = ""

    def __init__(self, root, split_override=None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "NIFTI").exists():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "NIFTI").exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected 'NIFTI/' under {root}")

    def iter_entries(self) -> Iterator[USManifestEntry]:
        case_dirs = sorted(
            p for p in (self.root / "NIFTI").iterdir()
            if p.is_dir() and p.name.startswith("Case")
        )
        split_map = self._group_split_map(case_dir.name for case_dir in case_dirs)

        for case_dir in case_dirs:
            case_id = case_dir.name
            split = split_map.get(case_id, "train")
            for vol_path in sorted((case_dir / "US").glob("*.nii.gz")):
                stem = vol_path.name.replace(".nii.gz", "")
                stage = stem.split("-US-", 1)[-1]
                yield self._make_entry(
                    str(vol_path),
                    split=split,
                    modality="volume",
                    study_id=case_id,
                    series_id=stem,
                    is_3d=True,
                    view_type=f"resection_{stage}",
                    task_type="ssl_only",
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={
                        "case_id": case_id,
                        "resection_stage": stage,
                    },
                )

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }
