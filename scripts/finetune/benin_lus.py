#!/usr/bin/env python3
"""Alias for scripts/finetune/lus.py (Benin-LUS patient-level TB MIL)."""
from __future__ import annotations

import sys
from pathlib import Path

# scripts/finetune.py (file) shadows scripts/finetune/ (directory) as a package,
# so add this directory to sys.path and import lus directly.
_FINETUNE_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_FINETUNE_SCRIPTS.parent.parent))
sys.path.insert(0, str(_FINETUNE_SCRIPTS))

from lus import main

if __name__ == "__main__":
    main()
