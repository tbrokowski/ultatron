"""data/adapters/gallbladder/gbcu.py  - GBCU off-store stub"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator
from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry
log = logging.getLogger(__name__)

class GBCUAdapter(BaseAdapter):
    DATASET_ID = "GBCU"
    ANATOMY_FAMILY = "gallbladder"
    SONODQS = "unrated"
    DOI = ""

    def __init__(self, root, split_override=None):
        self.root = Path(root)
        self.split_override = split_override

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            log.warning("GBCU: root missing at %s", self.root)
            return
        log.warning("GBCU: not yet on store — yielding no entries")
        return
        yield
