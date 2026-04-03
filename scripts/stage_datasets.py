#!/usr/bin/env python3
"""
Stage ultrasound datasets from Capstor Store → Scratch (rsync via StorageConfig).

Requires Python >= 3.10 (same as the rest of Ultatron). The cluster default
`python3` on bare nodes is often too old; use the training stack, e.g.:

  module load cray-python/3.11.7   # example — check `module avail python`
  python3 scripts/stage_datasets.py --dry-run

Or run inside the same container / environment you use for training.
"""
import argparse
import logging
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    print(
        "ERROR: This script needs Python 3.10 or newer (project requirement).\n"
        f"  Current interpreter: {sys.executable}\n"
        f"  Version: {sys.version}\n"
        "  On CSCS: `module avail python` / `module load cray-python/...` "
        "or run inside your training container.",
        file=sys.stderr,
    )
    sys.exit(1)

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.infra.storage import StorageConfig


def main() -> int:
    p = argparse.ArgumentParser(description="Stage Store → Scratch via rsync.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rsync commands without running them.",
    )
    p.add_argument(
        "--anatomy",
        type=str,
        default=None,
        metavar="FAMILY",
        help="Only stage datasets under this anatomy family (e.g. cardiac).",
    )
    p.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="ID",
        help="Stage a single dataset id (repeatable). Default: all known datasets.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    cfg = StorageConfig()
    if cfg.scratch_root is None:
        print(
            "ERROR: scratch not configured. Set CSCS_USER or US_SCRATCH_ROOT.",
            file=sys.stderr,
        )
        return 1

    print(f"store    : {cfg.store_root}")
    print(f"scratch  : {cfg.scratch_root}")
    print()

    if args.datasets:
        results = {
            did: cfg.stage_dataset(did, dry_run=args.dry_run)
            for did in args.datasets
        }
    else:
        results = cfg.stage_all(anatomy_family=args.anatomy, dry_run=args.dry_run)

    ok = all(results.values())
    for did, success in sorted(results.items()):
        print(f"{'OK' if success else 'FAIL':4}  {did}")

    if ok:
        print()
        print(cfg.status_report())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
