"""
data/adapters/lung/lus_multicenter.py  - LUS Multicenter 2025 adapter

Dataset:  LUS-multicenter-2025 (internal)
Task:     Multi-label lung pathology classification
Layout:   TBD — yields no entries when root is absent or not yet populated.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)


class LUSMulticenterAdapter(BaseAdapter):
    """
    Placeholder adapter for the LUS-multicenter-2025 dataset.

    Until the dataset is staged and a concrete layout is defined, this adapter
    yields no entries, allowing the registry to import without crashing.
    """

    DATASET_ID     = "LUS-multicenter-2025"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "unrated"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        if not self.root.exists():
            log.warning(
                "LUSMulticenterAdapter: root not found at %s — will yield no entries.",
                self.root,
            )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            return
        log.warning(
            "LUSMulticenterAdapter: dataset layout not yet implemented — "
            "yielding no entries from %s",
            self.root,
        )
        return
        yield  # make this a generator
