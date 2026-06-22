#!/usr/bin/env python3
"""
Benin-LUS patient-level TB classification (MIL) — all backbones, one GPU.

Also runnable as: python scripts/finetune/benin_lus.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/finetune.py (file) shadows scripts/finetune/ (directory) as a package,
# so add this directory to sys.path and import common directly.
_FINETUNE_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_FINETUNE_SCRIPTS.parent.parent))
sys.path.insert(0, str(_FINETUNE_SCRIPTS))

from common import add_common_args, launch_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finetune comparison: LUS patient TB (Benin-LUS only, gated MIL)",
    )
    add_common_args(parser)
    args = parser.parse_args()
    launch_experiment("lus", args)


if __name__ == "__main__":
    main()
