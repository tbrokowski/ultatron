"""data/adapters/lung/ultrasound_lus.py  - ULTRASOUND-LUS off-store stub"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator
from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry
log = logging.getLogger(__name__)

class ULTRASOUNDLUSAdapter(BaseAdapter):
    DATASET_ID = "ULTRASOUND-LUS"
    ANATOMY_FAMILY = "lung"
    SONODQS = "unrated"
    DOI = ""

    def __init__(self, root, split_override=None):
        self.root = Path(root)
        self.split_override = split_override

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            log.warning("ULTRASOUND-LUS: root missing at %s", self.root)
            return
        log.warning("ULTRASOUND-LUS: not yet on store — yielding no entries")
        return
        yield
