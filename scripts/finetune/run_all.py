#!/usr/bin/env python3
"""
Run the full finetune comparison sweep (all experiments × all backbones).

Replaces scripts/submit_finetune.sh for comparison mode.

Usage:
    python scripts/finetune/run_all.py
    python scripts/finetune/run_all.py --local
    python scripts/finetune/run_all.py --backbones biomedclip usfm echocare
    python scripts/finetune/run_all.py --gpus 4 --after-job 2527484
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/finetune.py (file) shadows scripts/finetune/ (directory) so
# `from scripts.finetune.common import ...` fails.  Add this directory
# to sys.path and import common directly instead.
_FINETUNE_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _FINETUNE_SCRIPTS.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_FINETUNE_SCRIPTS))

from common import add_common_args, launch_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ultatron finetune comparison — all experiments in parallel",
    )
    add_common_args(parser)
    args = parser.parse_args()
    launch_all(args)


if __name__ == "__main__":
    main()
