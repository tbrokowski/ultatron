#!/usr/bin/env python3
"""LUS clip-level 7-finding multilabel (Benin + RSA) — all backbones, one GPU."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.finetune.common import add_common_args, launch_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune comparison: LUS video")
    add_common_args(parser)
    args = parser.parse_args()
    launch_experiment("lus_video", args)


if __name__ == "__main__":
    main()
