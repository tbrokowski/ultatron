"""
data/adapters/thyroid/segthy.py  - Segthy thyroid NIfTI adapter

Segthy: 3D tracked thyroid ultrasound volumes with manual whole-gland
segmentation from 28 healthy volunteers (PLOS ONE 2022).

Layout
------
  {root}/ground_truth_data/US/*_US.nii
  {root}/ground_truth_data/US_thyroid_label/*.nii

Each 3D volume is emitted as one manifest entry per axial slice that contains
thyroid tissue.  Entries reference the source NIfTI paths and set
``source_meta["frame_idx"]`` so the dataloader selects the correct slice.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.pipeline.dataset import _read_nifti_array
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_VOLUME_RE = re.compile(
    r"^(?P<volunteer>\d{3})_(?P<physician>P\d)_(?P<scan>\d)_(?P<lobe>left|right)"
)


class SegthyAdapter(BaseAdapter):
    DATASET_ID     = "Segthy-Dataset"
    ANATOMY_FAMILY = "thyroid"
    SONODQS        = "gold"
    DOI            = "https://www.cs.cit.tum.de/camp/publications/segthy-dataset/"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._gt_us  = self.root / "ground_truth_data" / "US"
        self._gt_lbl = self.root / "ground_truth_data" / "US_thyroid_label"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._gt_us.is_dir():
            log.warning("SegthyAdapter: missing %s", self._gt_us)
            return

        volumes = sorted(self._gt_us.glob("*.nii"))
        if not volumes:
            log.warning("SegthyAdapter: no NIfTI volumes under %s", self._gt_us)
            return

        split_map = self._volunteer_split_map(volumes)

        for vol_path in volumes:
            meta     = self._parse_volume_name(vol_path.stem)
            lbl_path = self._resolve_label_path(vol_path)
            if lbl_path is None:
                log.warning("SegthyAdapter: no label for %s — skipping", vol_path.name)
                continue

            try:
                lbl = _read_nifti_array(str(lbl_path))
            except Exception as exc:
                log.warning("SegthyAdapter: failed to read %s — %s", lbl_path, exc)
                continue

            split = split_map.get(meta["volunteer_id"], "train")

            for z in range(lbl.shape[0]):
                if not (lbl[z] > 0).any():
                    continue

                sid = f"{vol_path.stem}_z{z:03d}"
                instances = [
                    self._make_instance(
                        instance_id=sid,
                        label_raw="thyroid",
                        label_ontology="whole_thyroid",
                        mask_path=str(lbl_path),
                        is_promptable=True,
                    )
                ]

                yield self._make_entry(
                    str(vol_path),
                    split=split,
                    modality="image",
                    instances=instances,
                    study_id=meta["volunteer_id"],
                    series_id=vol_path.stem,
                    view_type=f"thyroid_{meta['lobe']}",
                    has_mask=True,
                    task_type="segmentation",
                    ssl_stream="image",
                    is_promptable=True,
                    source_meta={
                        "volume": vol_path.name,
                        "label_volume": lbl_path.name,
                        "slice_idx": z,
                        "frame_idx": z,
                        **meta,
                    },
                )

    def _resolve_label_path(self, vol_path: Path) -> Optional[Path]:
        candidates = [
            self._gt_lbl / vol_path.name.replace("_US.nii", ".nii"),
            self._gt_lbl / vol_path.name,
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _parse_volume_name(self, stem: str) -> Dict[str, str]:
        clean = stem.replace("_US", "")
        match = _VOLUME_RE.match(clean)
        if match:
            return {
                "volunteer_id": match.group("volunteer"),
                "physician":    match.group("physician"),
                "scan_num":     match.group("scan"),
                "lobe":         match.group("lobe"),
            }
        return {
            "volunteer_id": clean.split("_", 1)[0],
            "physician":    "",
            "scan_num":     "",
            "lobe":         "",
        }

    def _volunteer_split_map(self, volumes: List[Path]) -> Dict[str, str]:
        volunteers = sorted({
            self._parse_volume_name(v.stem)["volunteer_id"] for v in volumes
        })
        return {
            vid: self._infer_split(vid, idx, len(volunteers))
            for idx, vid in enumerate(volunteers)
        }
