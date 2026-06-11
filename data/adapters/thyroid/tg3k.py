"""data/adapters/thyroid/tg3k.py  - TG3K off-store stub"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator
from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry
log = logging.getLogger(__name__)

class TG3KAdapter(BaseAdapter):
    DATASET_ID = "TG3K"
    ANATOMY_FAMILY = "thyroid"
    SONODQS = "unrated"
    DOI = ""

    def __init__(self, root, split_override=None):
        self.root = Path(root)
        self.split_override = split_override

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            log.warning("TG3K: root missing at %s", self.root)
            return
        log.warning("TG3K: not yet on store — yielding no entries")
        return
        yield
