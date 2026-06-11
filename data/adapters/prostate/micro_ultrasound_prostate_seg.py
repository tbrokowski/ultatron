"""
data/adapters/prostate/micro_ultrasound_prostate_seg.py  ·  Micro-US prostate seg adapter
==========================================================================================

Micro-Ultrasound Prostate Segmentation challenge dataset.

Layout:

    {root}/Micro_Ultrasound_Prostate_Segmentation_Dataset/
        train|test/
            micro_ultrasound_scans/     microUS_{split}_{id}.nii.gz
            expert_annotations/         expert_annotation_{split}_{id}.nii.gz
            non_expert_annotations/     (train only)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_ANNOT_RE = re.compile(
    r"^microUS_(train|test)_(\d+)\.nii\.gz$", re.IGNORECASE
)


class MicroUltrasoundProstateSegAdapter(BaseAdapter):
    DATASET_ID = "Micro-Ultrasound-Prostate-Segmentation"
    ANATOMY_FAMILY = "prostate"
    SONODQS = "gold"
    DOI = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_root = self._resolve_data_root(self.root)

    @classmethod
    def _resolve_data_root(cls, root: Path) -> Path:
        nested = root / "Micro_Ultrasound_Prostate_Segmentation_Dataset"
        if nested.is_dir():
            return nested
        return root

    @staticmethod
    def _expert_mask_path(scan_path: Path, annot_dir: Path) -> Optional[Path]:
        m = _ANNOT_RE.match(scan_path.name)
        if not m:
            return None
        split_tag, idx = m.group(1), m.group(2)
        candidate = annot_dir / f"expert_annotation_{split_tag}_{idx}.nii.gz"
        return candidate if candidate.exists() else None

    def iter_entries(self) -> Iterator[USManifestEntry]:
        split_dirs = sorted(
            p for p in self._data_root.iterdir()
            if p.is_dir() and p.name in ("train", "val", "test")
        )
        if not split_dirs:
            log.warning(
                "%s: no train/val/test splits under %s",
                self.DATASET_ID,
                self._data_root,
            )
            return

        for split_dir in split_dirs:
            split_name = split_dir.name
            scan_dir = split_dir / "micro_ultrasound_scans"
            expert_dir = split_dir / "expert_annotations"
            if not scan_dir.is_dir():
                continue

            for scan_path in sorted(scan_dir.glob("*.nii.gz")):
                split = self.split_override or split_name
                mask_path = (
                    self._expert_mask_path(scan_path, expert_dir)
                    if expert_dir.is_dir()
                    else None
                )
                instances: List[Instance] = []
                has_mask = mask_path is not None
                if has_mask:
                    instances.append(
                        self._make_instance(
                            instance_id=scan_path.stem,
                            label_raw="prostate",
                            label_ontology="prostate_gland",
                            mask_path=str(mask_path),
                            is_promptable=True,
                        )
                    )

                yield self._make_entry(
                    str(scan_path),
                    split=split,
                    modality="volume",
                    instances=instances,
                    study_id=scan_path.stem,
                    series_id=scan_path.stem,
                    is_3d=True,
                    has_mask=has_mask,
                    task_type="segmentation" if has_mask else "ssl_only",
                    ssl_stream="image",
                    is_promptable=has_mask,
                    source_meta={"split_dir": split_name},
                )
