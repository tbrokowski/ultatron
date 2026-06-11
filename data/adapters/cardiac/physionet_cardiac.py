"""
data/adapters/cardiac/physionet_cardiac.py  ·  PhysioNet-cardiac adapter
=========================================================================

Standalone PhysioNet wget mirror under cardiac/physionet.org/.
Uses the same CSV-driven MIMIC-IV-ECHO layout as MIMIC-IV-ECHO but keeps a
distinct dataset_id for partial or alternate download roots.
"""
from __future__ import annotations

from pathlib import Path

from .mimic_echo import MIMICEchoAdapter


class PhysioNetCardiacAdapter(MIMICEchoAdapter):
    DATASET_ID = "PhysioNet-cardiac"

    def _base_dir(self) -> Path:
        """Root already points at the physionet.org mirror directory."""
        direct = self.root / "files" / "mimic-iv-echo" / "1.0"
        if direct.exists():
            return direct
        if (self.root / "echo-record-list.csv").exists():
            return self.root
        return super()._base_dir()
