#!/usr/bin/env python3
"""Extract TCIA/NBIA per-series DICOM zips under a dataset's dicoms/ folder.

Each <series_uid>.zip is unpacked to dicoms/<series_uid>/*.dcm (LICENSE skipped).
Already-extracted series are skipped unless --force is passed.
"""
import argparse
import zipfile
from pathlib import Path


def _series_done(out_dir):
    return out_dir.is_dir() and any(out_dir.glob("*.dcm"))


def unpack_dicoms(dicoms_dir, force=False):
    zips = sorted(dicoms_dir.glob("*.zip"))
    extracted = skipped = failed = 0

    for zip_path in zips:
        out_dir = dicoms_dir / zip_path.stem
        if not force and _series_done(out_dir):
            skipped += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    name = Path(member.filename).name
                    if not name.lower().endswith(".dcm"):
                        continue
                    dst = out_dir / name
                    if dst.exists() and not force:
                        continue
                    with zf.open(member) as src, dst.open("wb") as out:
                        out.write(src.read())
            extracted += 1
            print(f"  extracted {zip_path.name} -> {out_dir.name}/")
        except zipfile.BadZipFile as exc:
            failed += 1
            print(f"  FAILED {zip_path.name}: {exc}")

    return extracted, skipped, failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dicoms_dir", type=Path, help="Path to dataset dicoms/ directory")
    parser.add_argument("--force", action="store_true", help="Re-extract even if .dcm files exist")
    args = parser.parse_args()

    dicoms_dir = args.dicoms_dir.resolve()
    if not dicoms_dir.is_dir():
        raise SystemExit(f"Not a directory: {dicoms_dir}")

    print(f"Unpacking DICOM zips in {dicoms_dir}")
    extracted, skipped, failed = unpack_dicoms(dicoms_dir, force=args.force)
    print(f"Done: {extracted} extracted, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
