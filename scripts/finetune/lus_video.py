#!/usr/bin/env python3
"""LUS clip-level 7-finding multilabel (Benin + RSA) — all backbones, one GPU."""
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
    parser = argparse.ArgumentParser(description="Finetune comparison: LUS video")
    add_common_args(parser)
    args = parser.parse_args()
    launch_experiment("lus_video", args)


if __name__ == "__main__":
    main()
