"""
build_manifest.py  ·  CLI to build the master manifest from all datasets
=========================================================================

Usage:
    python scripts/build_manifest.py \
        --config configs/run1/data_run1.yaml \
        --out dataset_exploration_outputs/run1/run1_train.jsonl

    # Subset only:
    python scripts/build_manifest.py \
        --config configs/run1/data_run1.yaml \
        --out run1_train.jsonl \
        --datasets CAMUS EchoNet-Dynamic

    # Filter an existing manifest to remove entries with missing files:
    python scripts/build_manifest.py \
        --filter-missing dataset_exploration_outputs/run1/run1_train.jsonl \
        --out dataset_exploration_outputs/run1/run1_train_filtered.jsonl

This script:
  1. Reads dataset root paths from config
  2. Calls each registered adapter
  3. Writes all entries to a single JSONL manifest
  4. Prints per-dataset counts and final statistics

A dataset is skipped (with a warning) if:
  - its root path is missing from config
  - its root directory does not exist on disk
  - its adapter raises any exception (e.g. zip not yet extracted)
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


def _filter_missing(src: Path, dst: Path) -> None:
    """Copy src manifest to dst, dropping entries where any image_path is missing."""
    entries = load_manifest(src)
    log.info(f"Loaded {len(entries):,} entries from {src}")
    kept, dropped = 0, 0
    with ManifestWriter(dst) as writer:
        for e in entries:
            paths = e.image_paths or []
            missing = [p for p in paths if not Path(p).exists()]
            if missing:
                dropped += 1
            else:
                writer.write(e)
                kept += 1
    log.info(f"Kept {kept:,} entries, dropped {dropped:,} with missing files → {dst}")
    stats = manifest_stats(load_manifest(dst))
    log.info("Filtered manifest statistics:")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", default=None,
                        help="Force all entries to this split (train/val/test)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of dataset IDs to process")
    parser.add_argument("--filter-missing", default=None, metavar="MANIFEST",
                        help="Filter an existing manifest, dropping entries with missing files")
    args = parser.parse_args()

    if args.filter_missing:
        _filter_missing(Path(args.filter_missing), Path(args.out))
        return

    if not args.config:
        parser.error("--config is required unless --filter-missing is used")

    cfg = yaml.safe_load(open(args.config))  # type: ignore[arg-type]
    dataset_roots = cfg.get("datasets", {})

    ds_ids = args.datasets or list(dataset_roots.keys())
    log.info(f"Building manifest for {len(ds_ids)} datasets → {args.out}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    total   = 0
    counts  = {}
    skipped = []

    with ManifestWriter(Path(args.out)) as writer:
        for ds_id in ds_ids:
            root = dataset_roots.get(ds_id)
            if not root:
                log.warning(f"[{ds_id}] No root configured — skipping.")
                skipped.append((ds_id, "no root in config"))
                continue
            if not Path(root).exists():
                log.warning(f"[{ds_id}] Root not found: {root} — skipping.")
                skipped.append((ds_id, f"root path missing"))
                continue
            try:
                n = build_manifest_for_dataset(ds_id, Path(root), writer,
                                               split_override=args.split)
                log.info(f"[{ds_id}] {n:>7,} entries written.")
                counts[ds_id] = n
                total += n
            except FileNotFoundError as exc:
                msg = str(exc).split("\n")[0]
                log.warning(f"[{ds_id}] Skipped — not ready: {msg}")
                skipped.append((ds_id, msg))
            except Exception as exc:
                log.warning(f"[{ds_id}] Skipped — error: {exc}")
                skipped.append((ds_id, str(exc)))

    log.info(f"Total entries written: {total:,}")

    if skipped:
        log.warning(f"{len(skipped)} dataset(s) skipped:")
        for ds_id, reason in skipped:
            log.warning(f"  {ds_id}: {reason}")

    # Print stats
    entries = load_manifest(Path(args.out))
    stats = manifest_stats(entries)
    log.info("Manifest statistics:")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
