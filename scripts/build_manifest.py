"""
build_manifest.py  ·  CLI to build the master manifest from all datasets
=========================================================================

Usage:
    python build_manifest.py \
        --config ../configs/data_config.yaml \
        --out /data/manifests/us_foundation_train.jsonl \
        --split train

This script:
  1. Reads dataset root paths from config
  2. Calls each registered adapter
  3. Writes all entries to a single JSONL manifest
  4. Prints statistics

Can be run incrementally: pass --datasets CAMUS EchoNet-Dynamic
to build a manifest for a subset only.
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.schema.manifest import ManifestWriter, manifest_stats, load_manifest
from data.adapters import build_manifest_for_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_manifest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None,
                        help="Force all entries to this split (train/val/test)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of dataset IDs to process")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dataset_roots = cfg.get("datasets", {})

    ds_ids = args.datasets or list(dataset_roots.keys())
    log.info(f"Building manifest for {len(ds_ids)} datasets → {args.out}")

    total = 0
    with ManifestWriter(Path(args.out)) as writer:
        for ds_id in ds_ids:
            root = dataset_roots.get(ds_id)
            if not root:
                log.warning(f"No root configured for {ds_id}. Skipping.")
                continue
            if not Path(root).exists():
                log.warning(f"Root not found for {ds_id}: {root}. Skipping.")
                continue
            n = build_manifest_for_dataset(ds_id, Path(root), writer,
                                           split_override=args.split)
            total += n

    log.info(f"Total entries written: {total}")

    # Print stats
    entries = load_manifest(Path(args.out))
    stats = manifest_stats(entries)
    log.info("Manifest statistics:")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
