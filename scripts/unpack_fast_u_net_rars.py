#!/usr/bin/env python3
"""Extract Fast-U-Net AC/HC .rar archives into resized_data/image and mask/.

The upstream repo ships train/val split files that reference
``resized_data/image/*.png`` and ``resized_data/mask/*.png``, but the actual
pixels live in per-plane .rar archives (AC_image1.rar, HC_mask.rar, …).
Until those archives are unpacked, the Fast-U-Net adapter only indexes the
small bundled test_data/ set (typically 3 images).

Requires the ``unrar`` command (RARLab) or Python ``rarfile`` with unrar on PATH.
Already-populated resized_data/ directories are skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_IMAGE_RAR_GLOBS = ("*_image*.rar", "*_image.rar")
_MASK_RAR_SUFFIX = "_mask.rar"


def _find_unrar() -> str | None:
    for name in ("unrar", "unar", "bsdtar"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _extract_member(unrar: str, rar_path: Path, member: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    # unrar x -o+ archive member dest_dir/
    subprocess.check_call(
        [unrar, "x", "-o+", "-inul", str(rar_path), member, str(dst.parent) + "/"],
        stdout=subprocess.DEVNULL,
    )
    extracted = dst.parent / Path(member).name
    if extracted != dst and extracted.exists():
        extracted.rename(dst)


def _flatten_rar(unrar: str, rar_path: Path, out_dir: Path) -> int:
    """Extract all image/mask members from one archive, flattening subfolders."""
    listing = subprocess.check_output(
        [unrar, "lb", str(rar_path)],
        text=True,
    )
    written = 0
    for member in listing.splitlines():
        member = member.strip()
        if not member or member.endswith("/"):
            continue
        name = Path(member).name
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        dst = out_dir / name
        if dst.exists():
            continue
        _extract_member(unrar, rar_path, member, dst)
        written += 1
    return written


def unpack_plane(plane_dir: Path, *, force: bool = False) -> tuple[int, int]:
    image_out = plane_dir / "resized_data" / "image"
    mask_out = plane_dir / "resized_data" / "mask"
    if not force and image_out.is_dir() and any(image_out.glob("*.png")):
        return 0, 0

    unrar = _find_unrar()
    if unrar is None:
        raise SystemExit(
            "unrar not found on PATH. Install RARLab unrar, then re-run.\n"
            "  https://www.rarlab.com/rar_add.htm"
        )

    image_rars = sorted(
        rar for pattern in _IMAGE_RAR_GLOBS for rar in plane_dir.glob(pattern)
    )
    mask_rars = sorted(plane_dir.glob(f"*{_MASK_RAR_SUFFIX}"))

    images = masks = 0
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for rar_path in image_rars:
        images += _flatten_rar(unrar, rar_path, image_out)
    for rar_path in mask_rars:
        masks += _flatten_rar(unrar, rar_path, mask_out)

    return images, masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Fast-U-Net-main root (contains Dataset/AC and Dataset/HC)",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract even if resized_data/ exists")
    args = parser.parse_args()

    root = args.root.resolve()
    dataset_dir = root / "Dataset"
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset/ not found under {root}")

    total_images = total_masks = 0
    for plane in ("AC", "HC"):
        plane_dir = dataset_dir / plane
        if not plane_dir.is_dir():
            print(f"  skip {plane}: directory missing")
            continue
        print(f"Unpacking {plane_dir} …")
        images, masks = unpack_plane(plane_dir, force=args.force)
        print(f"  {plane}: {images} images, {masks} masks extracted")
        total_images += images
        total_masks += masks

    print(f"Done: {total_images} images, {total_masks} masks")


if __name__ == "__main__":
    main()
