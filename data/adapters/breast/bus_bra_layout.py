"""
BUS-BRA on-disk layout helpers (OpenUS downstream protocol).

OpenUS (Zheng et al., arXiv:2511.11510) evaluates breast lesion segmentation on
BUS-BRA with a fixed split of 1200 train / 299 val / 376 test images at 224×224.

Supported layouts
-----------------
1. ``organized_data/`` (OpenUS release): ``fold_{N}/{train,validation,test}/images|masks``
2. Official Zenodo release: ``Images/`` + ``Masks/`` + ``5-fold-cv.csv``
   — fold 0 protocol: test = kFold==1, val = first 299 of remaining pool (sorted by ID)
3. ``split_manifest.json``: explicit sample_id → split mapping (preferred when present)
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
_OPENUS_VAL_COUNT = 299


def _find_images_dir(root: Path) -> Path:
    for name in ("Images", "images"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"BUS-BRA: Images/ not found under {root}")


def _find_masks_dir(root: Path) -> Path:
    for name in ("Masks", "masks"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"BUS-BRA: Masks/ not found under {root}")


def _mask_path(masks_dir: Path, img_path: Path) -> Optional[Path]:
    """Resolve mask for bus_XXXX.png or img_XXXX.png naming."""
    stem = img_path.stem
    candidates = [
        masks_dir / img_path.name,
        masks_dir / f"{stem.replace('bus_', 'mask_')}{img_path.suffix}",
        masks_dir / f"mask_{stem.removeprefix('bus_')}{img_path.suffix}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def detect_layout(root: Path) -> str:
    if (root / "organized_data").is_dir():
        return "organized"
    if _find_images_dir(root).exists():
        return "official"
    raise FileNotFoundError(f"BUS-BRA: no recognized layout under {root}")


def default_manifest_path() -> Path:
    return Path("dataset_exploration_outputs/busbra_openus/split_manifest.json")


def load_split_manifest(path: str | Path) -> dict[str, str]:
    with Path(path).open() as f:
        data = json.load(f)
    if "samples" in data:
        return {s["sample_id"]: s["split"] for s in data["samples"]}
    return {k: v for k, v in data.items() if k in {"train", "val", "test"}}


def _openus_fold0_splits(csv_path: Path) -> dict[str, str]:
    """
    OpenUS fold-0 split from 5-fold-cv.csv:
      test = kFold == 1 (376)
      val  = first 299 IDs from train pool (kFold != 1), sorted
      train = remaining 1200 from train pool
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    test_ids = sorted(r["ID"] for r in rows if int(r["kFold"]) == 1)
    pool     = sorted(r["ID"] for r in rows if int(r["kFold"]) != 1)
    val_ids  = set(pool[:_OPENUS_VAL_COUNT])
    splits: dict[str, str] = {sid: "test" for sid in test_ids}
    for sid in pool:
        splits[sid] = "val" if sid in val_ids else "train"
    return splits


def _organized_samples(
    root: Path,
    fold: int,
    split: Optional[str],
) -> Iterator[tuple[Path, Path, str, str]]:
    organized = root / "organized_data" / f"fold_{fold}"
    split_map = {
        "train": "train",
        "val":   "validation",
        "test":  "test",
    }
    for our_split, folder in split_map.items():
        if split is not None and split != our_split:
            continue
        img_dir  = organized / folder / "images"
        mask_dir = organized / folder / "masks"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
                continue
            mask_path = _mask_path(mask_dir, img_path)
            if mask_path is None:
                continue
            yield img_path, mask_path, img_path.stem, our_split


def iter_bus_bra_pairs(
    root: str | Path,
    split: Optional[str] = None,
    fold: int = 0,
    split_manifest: Optional[str | Path] = None,
) -> Iterator[tuple[Path, Path, str, str]]:
    """
    Yield ``(image_path, mask_path, sample_id, split)`` for BUS-BRA samples.
    """
    root = Path(root)
    layout = detect_layout(root)

    if layout == "organized":
        yield from _organized_samples(root, fold, split)
        return

    images_dir = _find_images_dir(root)
    masks_dir  = _find_masks_dir(root)

    manifest_splits: dict[str, str] = {}
    manifest_path = Path(split_manifest) if split_manifest else default_manifest_path()
    if manifest_path.exists():
        with manifest_path.open() as f:
            raw = json.load(f)
        if "samples" in raw:
            manifest_splits = {s["sample_id"]: s["split"] for s in raw["samples"]}
        log.debug("BUS-BRA: loaded split manifest from %s (%d entries)",
                  manifest_path, len(manifest_splits))

    if not manifest_splits:
        csv_path = root / "5-fold-cv.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"BUS-BRA: need split_manifest.json or 5-fold-cv.csv under {root}"
            )
        id_splits = _openus_fold0_splits(csv_path)
    else:
        id_splits = manifest_splits

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMG_EXTS:
            continue
        mask_path = _mask_path(masks_dir, img_path)
        if mask_path is None:
            continue

        sample_id = img_path.stem
        lookup_id = sample_id if sample_id in id_splits else img_path.name
        sample_split = id_splits.get(lookup_id, id_splits.get(sample_id, "train"))

        if split is not None and sample_split != split:
            continue

        yield img_path, mask_path, sample_id, sample_split


def list_bus_bra_samples(
    root: str | Path,
    split: str,
    fold: int = 0,
    split_manifest: Optional[str | Path] = None,
) -> list[dict[str, str]]:
    """Return sample dicts with img/lbl paths and sample_id."""
    samples = []
    for img_path, mask_path, sample_id, _ in iter_bus_bra_pairs(
        root, split=split, fold=fold, split_manifest=split_manifest,
    ):
        samples.append({
            "img":       str(img_path),
            "lbl":       str(mask_path),
            "img_path":  str(img_path),
            "lbl_path":  str(mask_path),
            "sample_id": sample_id,
        })
    return samples
