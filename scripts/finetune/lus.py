#!/usr/bin/env python3
"""
Benin-LUS + RSA-LUS patient-level TB classification (MIL) — all backbones, one GPU.

Also runnable as: python scripts/finetune/benin_lus.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.finetune.common import add_common_args, launch_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finetune comparison: LUS patient TB (Benin + RSA)",
    )
    add_common_args(parser)
    args = parser.parse_args()
    launch_experiment("lus", args)


if __name__ == "__main__":
    main()
