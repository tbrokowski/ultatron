#!/usr/bin/env python3
"""TN3K thyroid nodule segmentation — all backbones, one GPU."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.finetune.common import add_common_args, launch_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune comparison: TN3K")
    add_common_args(parser)
    args = parser.parse_args()
    launch_experiment("tn3k", args)


if __name__ == "__main__":
    main()
