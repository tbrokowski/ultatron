"""data/adapters/maternal_fetal/oc4us.py  - OC4US off-store stub"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterator
from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry
log = logging.getLogger(__name__)

class OC4USAdapter(BaseAdapter):
    DATASET_ID = "OC4US"
    ANATOMY_FAMILY = "fetal"
    SONODQS = "unrated"
    DOI = ""

    def __init__(self, root, split_override=None):
        self.root = Path(root)
        self.split_override = split_override

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            log.warning("OC4US: root missing at %s", self.root)
            return
        log.warning("OC4US: not yet on store — yielding no entries")
        return
        yield
