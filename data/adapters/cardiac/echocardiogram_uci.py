"""
data/adapters/cardiac/echocardiogram_uci.py  - Echocardiogram-UCI tabular stub
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)


class EchocardiogramUCIAdapter(BaseAdapter):
    DATASET_ID     = "Echocardiogram-UCI"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "unrated"
    DOI            = ""

    def __init__(self, root, split_override=None):
        self.root = Path(root)
        self.split_override = split_override

    def iter_entries(self) -> Iterator[USManifestEntry]:
        csv_path = self.root / "echocardiogram.csv"
        if not csv_path.exists():
            log.warning("Echocardiogram-UCI: echocardiogram.csv not found at %s", csv_path)
            return
        log.warning(
            "Echocardiogram-UCI: tabular-only manifest at %s — no imaging files on store",
            csv_path,
        )
        return
        yield
