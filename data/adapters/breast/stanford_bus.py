"""data/adapters/breast/stanford_bus.py  - STAnford-BUS off-store stub"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator
from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry
log = logging.getLogger(__name__)

class STAnfordBUSAdapter(BaseAdapter):
    DATASET_ID = "STAnford-BUS"
    ANATOMY_FAMILY = "breast"
    SONODQS = "unrated"
    DOI = ""

    def __init__(self, root, split_override=None):
        self.root = Path(root)
        self.split_override = split_override

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            log.warning("STAnford-BUS: root missing at %s", self.root)
            return
        log.warning("STAnford-BUS: not yet on store — yielding no entries")
        return
        yield
