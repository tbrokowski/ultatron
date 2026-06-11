#!/usr/bin/env python3
"""
Extract password-protected BEHSOF ``image_Data.zip`` archives.

The Mendeley BEHSOF release ships an encrypted zip. Set the password via
``BEHSOF_ZIP_PASSWORD`` (or ``--password``) before running:

    export BEHSOF_ZIP_PASSWORD='...'
    python scripts/extract_behsof.py --root /capstor/scratch/cscs/$USER/ultrasound/raw/liver/BEHSOF
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _extract_one(zip_path: Path, password: str, force: bool) -> bool:
    out_dir = zip_path.parent / zip_path.stem
    if out_dir.is_dir() and any(out_dir.rglob("*.jpg")) and not force:
        print(f"skip  {zip_path}  ({out_dir} already populated)")
        return True

    pwd = password.encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir, pwd=pwd)
    except RuntimeError as exc:
        print(f"FAIL  {zip_path}: {exc}", file=sys.stderr)
        return False

    n_jpg = sum(1 for _ in out_dir.rglob("*.jpg"))
    print(f"ok    {zip_path}  ->  {out_dir}  ({n_jpg:,} jpg)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract BEHSOF image_Data.zip archives")
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="BEHSOF dataset root (searched recursively for image_Data.zip)",
    )
    ap.add_argument(
        "--password",
        default=os.environ.get("BEHSOF_ZIP_PASSWORD", ""),
        help="Zip password (default: BEHSOF_ZIP_PASSWORD env var)",
    )
    ap.add_argument("--force", action="store_true", help="Re-extract even if image_Data/ exists")
    args = ap.parse_args()

    if not args.password:
        print(
            "Set BEHSOF_ZIP_PASSWORD or pass --password "
            "(password is provided with the Mendeley download).",
            file=sys.stderr,
        )
        return 1

    zips = sorted(args.root.rglob("image_Data.zip"))
    if not zips:
        print(f"No image_Data.zip found under {args.root}", file=sys.stderr)
        return 1

    ok = all(_extract_one(zp, args.password, args.force) for zp in zips)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
