#!/usr/bin/env python3
"""Alias for scripts/finetune/lus.py (Benin-LUS + RSA-LUS patient TB MIL)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.finetune.lus import main

if __name__ == "__main__":
    main()
