#!/usr/bin/env python3
"""
Prepare BUS-BRA split manifest for OpenUS downstream replication.

OpenUS (arXiv:2511.11510, Sec. 7.2): 1200 train / 299 val / 376 test at fold 0.
Protocol: test = 5-fold-cv kFold==1; val = first 299 of remaining pool (sorted ID).

Usage:
    python scripts/prepare_busbra_openus_split.py \\
        --data-root /capstor/.../BUSBRA \\
        --output-dir dataset_exploration_outputs/busbra_openus
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_OPENUS_VAL_COUNT = 299
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def _openus_fold0_splits(csv_path: Path) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    test_ids = sorted(r["ID"] for r in rows if int(r["kFold"]) == 1)
    pool     = sorted(r["ID"] for r in rows if int(r["kFold"]) != 1)
    val_ids  = set(pool[:_OPENUS_VAL_COUNT])
    splits: dict[str, str] = {sid: "test" for sid in test_ids}
    for sid in pool:
        splits[sid] = "val" if sid in val_ids else "train"
    return splits


def _mask_path(masks_dir: Path, img_path: Path) -> Path | None:
    stem = img_path.stem
    for candidate in (
        masks_dir / img_path.name,
        masks_dir / f"{stem.replace('bus_', 'mask_')}{img_path.suffix}",
        masks_dir / f"mask_{stem.removeprefix('bus_')}{img_path.suffix}",
    ):
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BUS-BRA OpenUS split manifest")
    parser.add_argument(
        "--data-root",
        default="/capstor/store/cscs/swissai/a127/ultrasound/raw/breast/BUSBRA",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_exploration_outputs/busbra_openus",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    manifest_path = out_dir / "split_manifest.json"

    if manifest_path.exists() and not args.force:
        print(f"[prepare_busbra_openus] Reusing {manifest_path}")
        return

    csv_path = root / "5-fold-cv.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"5-fold-cv.csv not found under {root}")

    id_splits = _openus_fold0_splits(csv_path)
    images_dir = root / "Images" if (root / "Images").is_dir() else root / "images"
    masks_dir  = root / "Masks" if (root / "Masks").is_dir() else root / "masks"

    samples = []
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
            continue
        mask_path = _mask_path(masks_dir, img_path)
        if mask_path is None:
            continue
        sid = img_path.stem
        samples.append({
            "sample_id": sid,
            "split":     id_splits[sid],
            "img_path":  str(img_path),
            "lbl_path":  str(mask_path),
        })

    counts = {k: sum(1 for s in samples if s["split"] == k) for k in ("train", "val", "test")}
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "openus_fold0",
        "paper": "arXiv:2511.11510",
        "fold": 0,
        "counts": counts,
        "samples": samples,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    print(f"[prepare_busbra_openus] Wrote {manifest_path}")
    print(f"[prepare_busbra_openus] Counts: {counts}")


if __name__ == "__main__":
    main()
