"""
data/adapters/cardiac/mimic_echoqa.py  - MIMIC-EchoQA VQA manifest stub
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_JSON_CANDIDATES = (
    "physionet.org/files/mimic-iv-ext-echoqa/1.0.0/MIMICEchoQA/MIMICEchoQA.json",
    "files/mimic-iv-ext-echoqa/1.0.0/MIMICEchoQA/MIMICEchoQA.json",
)


class MIMICEchoQAAdapter(BaseAdapter):
    DATASET_ID     = "MIMIC-EchoQA"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "unrated"
    DOI            = ""

    def __init__(self, root, split_override=None):
        self.root = Path(root)
        self.split_override = split_override

    def iter_entries(self) -> Iterator[USManifestEntry]:
        json_path = next((self.root / rel for rel in _JSON_CANDIDATES if (self.root / rel).exists()), None)
        if json_path is None:
            log.warning("MIMIC-EchoQA: MIMICEchoQA.json not found under %s", self.root)
            return
        log.warning(
            "MIMIC-EchoQA: VQA manifest present at %s but referenced MP4s are not on store",
            json_path,
        )
        return
        yield
