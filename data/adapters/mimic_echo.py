"""
data/adapters/mimic_echo.py  ·  MIMIC-IV-ECHO adapter
===========================================================

MIMIC-IV-ECHO: ~350,000 echocardiogram videos from Beth Israel Deaconess.
  Labels: None (pure SSL stream — no segmentation or regression targets).
  Format: .avi / .mp4 videos, nested directory structure.
  Access: PhysioNet credentialled access required.

  {root}/files/{subject_id}/{study_id}/{series}.avi

Gold quality: large scale, real clinical data, diverse pathology.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class MIMICEchoAdapter(BaseAdapter):
    """
    MIMIC-IV-ECHO adapter.  Pure SSL — no labels.

    The adapter recursively finds all .avi/.mp4 files under {root}/files/
    and emits one video entry per file.  Subject and study IDs are
    inferred from the directory hierarchy.
    """

    DATASET_ID     = "MIMIC-IV-ECHO"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.13026/7rbq-q661"

    VIDEO_EXTS = {".avi", ".mp4"}

    def iter_entries(self) -> Iterator[USManifestEntry]:
        # Collect all videos deterministically
        video_files = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.VIDEO_EXTS
        )
        n = len(video_files)

        for i, vpath in enumerate(video_files):
            # Directory structure: .../files/{subject_id}/{study_id}/{file}
            parts    = vpath.parts
            study_id = vpath.parent.name
            split    = self._infer_split(vpath.stem, i, n)

            yield self._make_entry(
                str(vpath), split,
                modality           = "video",
                study_id           = study_id,
                is_cine            = True,
                has_temporal_order = True,
                fps                = 30.0,
                task_type          = "ssl_only",
                ssl_stream         = "video",
                is_promptable      = False,
            )
