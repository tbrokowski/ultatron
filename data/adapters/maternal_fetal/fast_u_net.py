"""
data/adapters/maternal_fetal/fast_u_net.py  ·  Fast-U-Net stub adapter
======================================================================

Placeholder for the Fast-U-Net fetal ultrasound dataset.
Yields entries once images or annotations are present under the dataset root.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".dcm", ".nii", ".nii.gz", ".mp4", ".avi"}


class FastUNetAdapter(BaseAdapter):
    DATASET_ID = "Fast-U-Net"
    ANATOMY_FAMILY = "fetal"
    SONODQS = "unrated"
    DOI = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        media = [
            p for p in self.root.rglob("*")
            if p.is_file() and (
                p.suffix.lower() in _MEDIA_EXTS
                or p.name.endswith(".nii.gz")
            )
        ]
        if not media:
            log.warning(
                "%s: no media files found under %s — download not complete",
                self.DATASET_ID,
                self.root,
            )
            return

        media = sorted(media)
        for idx, path in enumerate(media):
            split = self._infer_split(path.stem, idx, len(media))
            yield self._make_entry(
                str(path),
                split=split,
                modality="image",
                task_type="ssl_only",
                ssl_stream="image",
                is_promptable=False,
                source_meta={"relative_path": str(path.relative_to(self.root))},
            )
