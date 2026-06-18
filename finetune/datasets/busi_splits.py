"""
finetune/datasets/busi_splits.py  ·  BUSI split helpers (USFM manifest + fallback)
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

log = logging.getLogger(__name__)

CLASS_NAMES = ("benign", "malignant", "normal")
SPLIT_SEED = 42


def load_split_manifest(manifest_path: str | Path) -> dict | None:
    path = Path(manifest_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read BUSI split manifest %s: %s", path, exc)
        return None


def sample_ids_for_split(manifest: dict, split: str) -> set[str] | None:
    """Return sample_id set for train/val/test from USFM split_manifest.json."""
    splits = manifest.get("splits", {})
    entry = splits.get(split) or splits.get(
        {"train": "train", "val": "val", "test": "test"}.get(split, split)
    )
    if not entry:
        return None
    ids = entry.get("sample_ids")
    if ids:
        return set(ids)
    return None


def default_repo_manifest_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "dataset_exploration_outputs"
        / "busi_usfm"
        / "split_manifest.json"
    )


def collect_busi_samples(root: Path) -> list[dict]:
    samples = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = root / cls_name
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.glob("*.png")):
            if "_mask" in img_path.name:
                continue
            mask_path = img_path.with_name(img_path.name.replace(".png", "_mask.png"))
            samples.append({
                "img": str(img_path),
                "mask": str(mask_path) if mask_path.exists() else None,
                "cls_label": cls_idx,
                "cls_name": cls_name,
                "sample_id": f"{cls_name}_{img_path.stem}",
            })
    return samples


def split_busi_samples(
    samples: list[dict],
    split: str,
    *,
    test_frac: float = 0.20,
    val_frac: float = 0.20,
    seed: int = SPLIT_SEED,
    manifest: dict | None = None,
    tumor_only_train: bool = False,
) -> list[dict]:
    """
    Assign samples to a split.

    Priority:
      1. USFM ``split_manifest.json`` sample_ids (70/15/15 per class)
      2. Per-class seeded holdout (60/20/20 default)
    """
    if manifest is not None:
        allowed = sample_ids_for_split(manifest, split)
        if allowed is not None:
            out = [s for s in samples if s["sample_id"] in allowed]
            if tumor_only_train and split == "train":
                out = [s for s in out if s["cls_name"] != "normal"]
            log.info(
                "BUSI %s: %d samples (USFM manifest, tumor_only_train=%s)",
                split, len(out), tumor_only_train and split == "train",
            )
            return out

    by_class: dict[str, list[dict]] = {c: [] for c in CLASS_NAMES}
    for s in samples:
        by_class[s["cls_name"]].append(s)

    out: list[dict] = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        if tumor_only_train and split == "train" and cls_name == "normal":
            continue
        items = list(by_class[cls_name])
        rng = random.Random(seed + cls_idx)
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_frac))
        n_val = max(1, int(len(items) * val_frac))
        if split == "test":
            subset = items[-n_test:]
        elif split == "val":
            subset = items[-(n_test + n_val):-n_test]
        else:
            subset = items[:-(n_test + n_val)]
        out.extend(subset)

    log.info("BUSI %s: %d samples (per-class seed=%d)", split, len(out), seed)
    return out
