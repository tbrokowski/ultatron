"""
data/adapters/breast/midi_b.py  - Midi-B DICOM stub (zips not extracted)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)


class MidiBAdapter(BaseAdapter):
    DATASET_ID     = "midi-b"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "unrated"
    DOI            = ""

    def __init__(self, root, split_override=None):
        self.root = Path(root)
        self.split_override = split_override

    def iter_entries(self) -> Iterator[USManifestEntry]:
        dicoms = self.root / "dicoms"
        if not dicoms.is_dir() or not any(dicoms.glob("*.zip")):
            log.warning("MidiBAdapter: DICOM zips not found at %s", dicoms)
            return
        log.warning("MidiBAdapter: DICOM zips present but not extracted — yielding no entries")
        return
        yield
