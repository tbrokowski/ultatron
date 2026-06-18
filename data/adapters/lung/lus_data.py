"""
data/adapters/lung/lus_data.py  ·  Deprecated — merged into COVIDx-US

Processed frames/videos/masks live under ``COVIDx-US/data/`` and are emitted by
:class:`~data.adapters.lung.covidx_us.COVIDxUSAdapter` only.  Building both
``COVIDx-US`` and ``LUS-data`` in the same manifest would duplicate entries.
"""
from __future__ import annotations

import logging
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)


class LUSDataAdapter(BaseAdapter):
    DATASET_ID = "LUS-data"
    ANATOMY_FAMILY = "lung"
    SONODQS = "silver"
    DOI = "https://github.com/nrc-cnrc/COVID-US"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        log.warning(
            "LUS-data is merged into COVIDx-US; skipping LUS-data adapter "
            "(use dataset_id=COVIDx-US with root=%s parent)", self.root,
        )
        return
        yield  # pragma: no cover
