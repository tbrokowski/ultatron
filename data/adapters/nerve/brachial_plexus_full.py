"""
data/adapters/nerve/brachial_plexus_full.py  ·  Brachial Plexus Full adapter
==============================================================================

Multi-machine brachial plexus ultrasound video dataset with per-frame
segmentation masks and bounding-box annotations.

Layout on Store:

    {root}/data/
      Sonosite/
        videos/           s005.mp4  s071.mp4  ...
        ac_masks/         s005/s005_000.jpg  s005_001.jpg  ...
        bb_annotations/   s005.txt  ...
        subjects_data.csv ← vid_name, nerve, frame_cnt, needle, gac_iter
      Butterfly/  (same structure)
      eSaote/     (same structure)

bb_annotations format — one JSON line per frame:
  {"0": {"tracker": "human", "bounding_boxes": "[[x, y, w, h], ...]"}}

One entry per video. image_paths is empty; the video path goes in
source_meta["video_path"]. Frame-level mask paths are stored in
source_meta["mask_paths"].
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_MACHINES = ("Sonosite", "Butterfly", "eSaote")


def _find_data_dir(root: Path) -> Path:
    candidate = root / "data"
    return candidate if candidate.is_dir() else root


def _load_subjects_csv(csv_path: Path) -> Dict[str, dict]:
    meta: Dict[str, dict] = {}
    if not csv_path.exists():
        return meta
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            vid = row.get("vid_name", "").strip()
            if vid:
                meta[vid] = row
    return meta


class BrachialPlexusFullAdapter(BaseAdapter):
    DATASET_ID     = "brachial-plexus-full"
    ANATOMY_FAMILY = "nerve"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_dir = _find_data_dir(self.root)

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for machine in _MACHINES:
            yield from self._iter_machine(machine)

    def _iter_machine(self, machine: str) -> Iterator[USManifestEntry]:
        machine_dir  = self._data_dir / machine
        vid_dir      = machine_dir / "videos"
        mask_root    = machine_dir / "ac_masks"
        bb_root      = machine_dir / "bb_annotations"
        subjects_csv = machine_dir / "subjects_data.csv"

        if not vid_dir.is_dir():
            return

        meta = _load_subjects_csv(subjects_csv)
        split = self.split_override or "train"

        for vpath in sorted(vid_dir.glob("*.mp4")):
            vid_id   = vpath.stem
            row      = meta.get(vid_id, meta.get(vpath.name, {}))

            # Frame count from CSV; fall back to counting mask files
            try:
                frame_cnt = int(row.get("frame_cnt", 0) or 0)
            except (ValueError, TypeError):
                frame_cnt = 0

            mask_dir   = mask_root / vid_id
            mask_files = (
                sorted(mask_dir.glob("*.jpg"), key=lambda p: p.stem)
                if mask_dir.is_dir() else []
            )
            if frame_cnt == 0:
                frame_cnt = len(mask_files)

            bb_path: Optional[Path] = bb_root / f"{vid_id}.txt"
            if not bb_path.exists():
                bb_path = None

            instances: List[Instance] = [
                self._make_instance(
                    instance_id=vid_id,
                    label_raw="brachial_plexus",
                    label_ontology="brachial_plexus",
                    is_promptable=False,
                )
            ]

            yield self._make_entry(
                [],                          # image_paths empty — video in source_meta
                split=split,
                modality="video",
                instances=instances,
                series_id=str(vpath),        # used for sample_id generation
                study_id=vid_id,
                label_raw=["brachial_plexus"],
                has_mask=True,
                has_box=True,
                has_temporal_order=True,
                is_cine=True,
                num_frames=frame_cnt,
                frame_indices=list(range(frame_cnt)) if frame_cnt > 0 else None,
                task_type="segmentation",
                ssl_stream="video",
                is_promptable=False,
                source_meta={
                    "video_path":        str(vpath),
                    "mask_paths":        [str(p) for p in mask_files],
                    "bb_annotations_path": str(bb_path) if bb_path else None,
                    "machine":           machine,
                    "nerve":             row.get("nerve", ""),
                    "needle":            row.get("needle", ""),
                },
            )
