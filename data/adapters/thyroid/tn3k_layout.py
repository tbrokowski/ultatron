"""
TN3K on-disk layout helpers.

Official release layout (capstor / HuggingFace):
    {root}/trainval-image/*.jpg
    {root}/trainval-mask/*.jpg
    {root}/test-image/*.jpg
    {root}/test-mask/*.jpg
    {root}/tn3k-trainval-fold{N}.json   (optional train/val split within trainval)

Legacy test layout (image/ + label/ with .png masks) is still supported.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
_BUNDLED_FOLD0 = Path(__file__).with_name("tn3k_trainval_fold0.json")


def detect_layout(root: Path) -> str:
    """Return ``official`` or ``legacy`` depending on which folders exist."""
    if (root / "trainval-image").is_dir() or (root / "test-image").is_dir():
        return "official"
    if (root / "image").is_dir():
        return "legacy"
    raise FileNotFoundError(
        f"TN3K: expected trainval-image/ or image/ under {root}"
    )


def _stem_index(stem: str) -> int:
    try:
        return int(stem)
    except ValueError:
        return -1


def _sorted_images(img_dir: Path) -> list[Path]:
    if not img_dir.is_dir():
        return []
    files = [
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    ]
    return sorted(files, key=lambda p: (_stem_index(p.stem), p.name))


def _resolve_mask(mask_dir: Path, img_path: Path) -> Optional[Path]:
    for ext in _IMG_EXTS:
        candidate = mask_dir / f"{img_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_trainval_fold(root: Path, fold: int = 0) -> dict[int, str]:
    """
    Map trainval image index → ``train`` | ``val``.

    Prefers ``tn3k-trainval-fold{N}.json`` at the dataset root, then
    ``tn3k-trainval.json``, then the bundled fold0 file shipped with the adapter.
    """
    candidates = [
        root / f"tn3k-trainval-fold{fold}.json",
        root / "tn3k-trainval.json",
    ]
    if fold == 0:
        candidates.append(_BUNDLED_FOLD0)

    for path in candidates:
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        mapping: dict[int, str] = {}
        for split_name in ("train", "val"):
            for idx in data.get(split_name, []):
                mapping[int(idx)] = split_name
        log.debug("TN3K: loaded train/val fold from %s (%d entries)", path, len(mapping))
        return mapping

    return {}


def _infer_trainval_split(idx: int, total: int) -> str:
    """80/20 train/val fallback when no fold JSON is available."""
    frac = idx / max(total - 1, 1)
    return "val" if frac >= 0.80 else "train"


def iter_tn3k_pairs(
    root: str | Path,
    split: Optional[str] = None,
    fold: int = 0,
    split_override: Optional[str] = None,
) -> Iterator[tuple[Path, Path, str, str]]:
    """
    Yield ``(image_path, mask_path, sample_id, split)`` for TN3K samples.

    Parameters
    ----------
    split
        If set, only yield samples belonging to this split
        (``train``, ``val``, or ``test``).
    fold
        Which official trainval fold JSON to use (default 0).
    split_override
        Force every yielded sample to this split label.
    """
    root = Path(root)
    layout = detect_layout(root)

    if layout == "legacy":
        yield from _iter_legacy_pairs(root, split, split_override)
        return

    trainval_fold = _load_trainval_fold(root, fold)
    subsets: list[tuple[str, Path, Path]] = [
        ("trainval", root / "trainval-image", root / "trainval-mask"),
        ("test",     root / "test-image",     root / "test-mask"),
    ]

    for subset_name, img_dir, mask_dir in subsets:
        imgs = _sorted_images(img_dir)
        n    = len(imgs)
        for i, img_path in enumerate(imgs):
            mask_path = _resolve_mask(mask_dir, img_path)
            if mask_path is None:
                continue

            if subset_name == "test":
                sample_split = "test"
            else:
                idx = _stem_index(img_path.stem)
                if trainval_fold:
                    sample_split = trainval_fold.get(idx, "train")
                else:
                    sample_split = _infer_trainval_split(i, n)

            if split_override:
                sample_split = split_override

            if split is not None and sample_split != split:
                continue

            yield img_path, mask_path, img_path.stem, sample_split


def _iter_legacy_pairs(
    root: Path,
    split: Optional[str],
    split_override: Optional[str],
) -> Iterator[tuple[Path, Path, str, str]]:
    img_dir = root / "image"
    lbl_dir = root / "label"
    imgs    = _sorted_images(img_dir)
    n       = len(imgs)
    n_test  = max(1, int(n * 0.15))
    n_val   = max(1, int(n * 0.15))

    for i, img_path in enumerate(imgs):
        mask_path = _resolve_mask(lbl_dir, img_path)
        if mask_path is None:
            continue

        if split_override:
            sample_split = split_override
        elif i >= n - n_test:
            sample_split = "test"
        elif i >= n - n_test - n_val:
            sample_split = "val"
        else:
            sample_split = "train"

        if split is not None and sample_split != split:
            continue

        yield img_path, mask_path, img_path.stem, sample_split


def list_tn3k_samples(
    root: str | Path,
    split: str,
    fold: int = 0,
) -> list[dict[str, str]]:
    """Return sample dicts with ``img``/``img_path``, ``lbl``/``lbl_path``, ``sample_id``."""
    samples = []
    for img_path, mask_path, sample_id, _ in iter_tn3k_pairs(root, split=split, fold=fold):
        samples.append({
            "img":       str(img_path),
            "lbl":       str(mask_path),
            "img_path":  str(img_path),
            "lbl_path":  str(mask_path),
            "sample_id": sample_id,
        })
    return samples
