"""
data/adapters/remind2reg.py  ·  ReMIND2Reg post-resection brain US adapter
=============================================================================

ReMIND2Reg packages paired post-resection iUS (_0000) and pre-operative MRI
(_0001 ceT1, _0002 T2) volumes for a registration challenge.  There are no
segmentation labels in this release.  The adapter emits only the iUS channel
for SSL pretraining and records paired MRI paths in ``source_meta`` when
present on disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.adapters.brain._volume_utils import nifti_depth
from data.schema.manifest import USManifestEntry


class ReMIND2RegAdapter(BaseAdapter):
    DATASET_ID = "ReMIND2Reg"
    ANATOMY_FAMILY = "brain"
    SONODQS = "gold"
    DOI = "https://doi.org/10.1101/2023.09.14.23295596"

    _MRI_CHANNELS = {
        "0001": "cet1_preoperative",
        "0002": "t2_preoperative",
    }

    def __init__(self, root, split_override=None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "imagesTr").exists():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "imagesTr").exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected 'imagesTr/' under {root}")

    def __iter_us_paths(self) -> List[Path]:
        dataset_json = self.root / "ReMIND2Reg_dataset.json"
        if dataset_json.exists():
            payload = json.loads(dataset_json.read_text())
            out = []
            for item in payload.get("training", []):
                rel = item.get("image", "")
                if not rel.endswith("_0000.nii.gz"):
                    continue
                path = (self.root / rel).resolve()
                if path.exists():
                    out.append(path)
            if out:
                return sorted(set(out))
        return sorted((self.root / "imagesTr").glob("*_0000.nii.gz"))

    def iter_entries(self) -> Iterator[USManifestEntry]:
        us_paths = self.__iter_us_paths()
        split_map = self._group_split_map(self._case_id(p) for p in us_paths)

        for us_path in us_paths:
            case_id = self._case_id(us_path)
            stem = us_path.name.replace(".nii.gz", "")
            num_frames = nifti_depth(us_path)
            paired_mri = self._paired_mri_paths(case_id)
            yield self._make_entry(
                str(us_path),
                split=split_map.get(case_id, "train"),
                modality="volume",
                study_id=case_id,
                series_id=stem,
                is_3d=True,
                is_cine=True,
                has_temporal_order=True,
                num_frames=num_frames,
                view_type="post_resection_ius",
                task_type="ssl_only",
                ssl_stream="both",
                is_promptable=False,
                source_meta={
                    "case_id": case_id,
                    "channel": "us_post_resection",
                    "num_z_slices": num_frames,
                    **paired_mri,
                },
            )

    def _paired_mri_paths(self, case_id: str) -> Dict[str, Optional[str]]:
        out: Dict[str, Optional[str]] = {}
        for suffix, label in self._MRI_CHANNELS.items():
            path = self.root / "imagesTr" / f"{case_id}_{suffix}.nii.gz"
            out[f"mri_{label}_path"] = str(path) if path.exists() else None
        return out

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    @staticmethod
    def _case_id(path: Path) -> str:
        stem = path.name.replace(".nii.gz", "")
        return stem.rsplit("_", 1)[0]
