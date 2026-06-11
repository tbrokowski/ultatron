"""
data/adapters/nerve/us_guided_anesthesia.py  - US-guided anesthesia adapter

Layout:
  <root>/Sonosite/videos/*.mp4  + ac_masks/{subject}/*.jpg
  <root>/eSaote/videos/*.mp4    + ac_masks/{subject}/*.jpg
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class USGuidedAnesthesiaAdapter(BaseAdapter):
    DATASET_ID     = "us-guided-anesthesia"
    ANATOMY_FAMILY = "nerve"
    SONODQS        = "silver"
    DOI            = ""

    _SCANNERS = ("Sonosite", "eSaote")

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._meta: Dict[str, dict] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        for scanner in self._SCANNERS:
            csv_path = self.root / scanner / "subjects_data.csv"
            if not csv_path.exists():
                continue
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    vid = row.get("vid_name", "").strip()
                    if vid:
                        self._meta[vid] = row

    def iter_entries(self) -> Iterator[USManifestEntry]:
        all_videos: List[tuple[Path, str, str]] = []
        for scanner in self._SCANNERS:
            vid_dir = self.root / scanner / "videos"
            if not vid_dir.is_dir():
                continue
            for vpath in sorted(vid_dir.glob("*.mp4")):
                all_videos.append((vpath, scanner, vpath.stem))

        n = len(all_videos)
        for i, (vpath, scanner, vid_id) in enumerate(all_videos):
            mask_dir = self.root / scanner / "ac_masks" / vid_id
            has_masks = mask_dir.is_dir() and any(mask_dir.glob("*.jpg"))
            split = self._infer_split(vid_id, i, n)
            meta = self._meta.get(vpath.name, self._meta.get(vid_id, {}))

            instances: List[Instance] = []
            if has_masks:
                instances.append(
                    self._make_instance(
                        instance_id=vid_id,
                        label_raw=meta.get("nerve", "nerve"),
                        label_ontology="peripheral_nerve",
                        is_promptable=False,
                    )
                )

            yield self._make_entry(
                str(vpath),
                split=split,
                modality="video",
                instances=instances,
                is_cine=True,
                has_temporal_order=True,
                task_type="segmentation" if has_masks else "ssl_only",
                ssl_stream="both",
                is_promptable=False,
                source_meta={
                    "scanner": scanner,
                    "vid_name": vpath.name,
                    "mask_dir": str(mask_dir) if has_masks else None,
                    **{k: v for k, v in meta.items() if v},
                },
            )
