"""
data/adapters/bite.py  ·  BITE intraoperative brain ultrasound adapter
========================================================================

BITE contains tracked 2D intraoperative ultrasound slices plus reconstructed
3D ultrasound volumes grouped by fold and subject. MRI and landmark files are
ignored here; we expose only the ultrasound content for pretraining.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class BITEAdapter(BaseAdapter):
    DATASET_ID = "BITE"
    ANATOMY_FAMILY = "brain"
    SONODQS = "silver"
    DOI = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        studies = self._study_dirs()
        split_map = self._group_split_map(study_id for study_id, _ in studies)

        for study_id, study_dir in studies:
            group_name, subject_id = study_id.split(":", 1)

            us3d_path = study_dir / "3D" / "US3DT.mnc"
            if us3d_path.exists():
                yield self._make_entry(
                    str(us3d_path),
                    split=split_map.get(study_id, "train"),
                    modality="volume",
                    study_id=study_id,
                    series_id=f"{study_id}:3d",
                    is_3d=True,
                    view_type="reconstructed_3d",
                    task_type="ssl_only",
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={
                        "group": group_name,
                        "subject_id": subject_id,
                        "sample_kind": "3d_volume",
                    },
                )

            two_d_dir = study_dir / "2D"
            for img_path in sorted(two_d_dir.glob("*.mnc")):
                yield self._make_entry(
                    str(img_path),
                    split=split_map.get(study_id, "train"),
                    modality="image",
                    study_id=study_id,
                    series_id=img_path.stem,
                    view_type="tracked_bmode_2d",
                    task_type="ssl_only",
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={
                        "group": group_name,
                        "subject_id": subject_id,
                        "sample_kind": "2d_slice",
                    },
                )

    def _study_dirs(self) -> List[Tuple[str, Path]]:
        out: List[Tuple[str, Path]] = []
        for group_dir in sorted(p for p in self.root.glob("group*") if p.is_dir()):
            group_name = group_dir.name
            for subject_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
                study_id = f"{group_name}:{subject_dir.name}"
                out.append((study_id, subject_dir))
        return out

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }
