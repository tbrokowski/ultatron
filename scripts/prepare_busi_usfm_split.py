#!/usr/bin/env python3
"""
Prepare BUSI in USFM SegBase layout for paper replication (Jiao et al., MedIA 2024).
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ("benign", "malignant", "normal")
SPLIT_DIRS = {
    "train": "training_set",
    "val": "val_set",
    "test": "test_set",
}


def _collect_samples(root):
    samples = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_dir = root / cls_name
        if not cls_dir.exists():
            continue
        for img_path in sorted(cls_dir.glob("*.png")):
            if "_mask" in img_path.name:
                continue
            mask_path = img_path.with_name(img_path.name.replace(".png", "_mask.png"))
            has_mask = mask_path.exists() and cls_name != "normal"
            samples.append(
                {
                    "sample_id": f"{cls_name}_{img_path.stem}",
                    "cls_name": cls_name,
                    "cls_label": cls_idx,
                    "img_path": str(img_path),
                    "mask_path": str(mask_path) if has_mask else None,
                }
            )
    return samples


def _split_samples(
    samples,
    *,
    seed,
    train_ratio,
    val_ratio,
):
    if abs(train_ratio + val_ratio + (1.0 - train_ratio - val_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    by_class = {c: [] for c in CLASS_NAMES}
    for s in samples:
        by_class[s["cls_name"]].append(s)

    splits = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    test_ratio = 1.0 - train_ratio - val_ratio

    for cls_name, cls_samples in by_class.items():
        items = list(cls_samples)
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = max(1, n - n_train - n_val) if n > 0 else 0
        if n_train + n_val + n_test > n:
            n_test = max(0, n - n_train - n_val)

        train_items = items[:n_train]
        val_items = items[n_train : n_train + n_val]
        test_items = items[n_train + n_val : n_train + n_val + n_test]

        splits["train"].extend(train_items)
        splits["val"].extend(val_items)
        splits["test"].extend(test_items)

        print(
            f"  {cls_name:10s}  total={n:4d}  "
            f"train={len(train_items):4d}  val={len(val_items):4d}  test={len(test_items):4d}"
        )

    return splits


def _link_or_copy(src, dst, use_symlinks):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if use_symlinks:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _write_mask(mask_path, src_img, mask_src):
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_src is not None and Path(mask_src).exists():
        arr = np.array(Image.open(mask_src).convert("L"))
        binary = (arr > 127).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(mask_path)
        return

    img = Image.open(src_img).convert("L")
    empty = np.zeros(img.size[::-1], dtype=np.uint8)
    Image.fromarray(empty, mode="L").save(mask_path)


def prepare_split(
    *,
    busi_root,
    out_root,
    seed=42,
    train_ratio=0.70,
    val_ratio=0.15,
    use_symlinks=True,
    force=False,
):
    busi_root = busi_root.resolve()
    out_root = out_root.resolve()
    manifest_path = out_root.parent / "split_manifest.json"

    if out_root.exists() and not force:
        if manifest_path.exists():
            print(f"[prepare_busi_usfm] Reusing existing split at {out_root}")
            return json.loads(manifest_path.read_text())
        raise SystemExit(
            f"[prepare_busi_usfm] {out_root} exists without split_manifest.json — use --force"
        )

    print(f"[prepare_busi_usfm] Source: {busi_root}")
    print(f"[prepare_busi_usfm] Output: {out_root}")
    print(f"[prepare_busi_usfm] Split seed={seed}  ratios train/val/test="
          f"{train_ratio:.0%}/{val_ratio:.0%}/{1-train_ratio-val_ratio:.0%}")

    samples = _collect_samples(busi_root)
    if not samples:
        raise FileNotFoundError(f"No BUSI samples under {busi_root}")

    splits = _split_samples(
        samples, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio,
    )

    if out_root.exists() and force:
        shutil.rmtree(out_root)

    manifest = {
        "source_root": str(busi_root),
        "output_root": str(out_root),
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "task": "tumor_vs_background",
        "num_classes": 2,
        "img_size_train": 512,
        "splits": {},
    }

    for split_name, split_samples in splits.items():
        split_dir = SPLIT_DIRS[split_name]
        img_dir = out_root / split_dir / "image"
        msk_dir = out_root / split_dir / "mask"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        ids = []
        for s in split_samples:
            src_img = Path(s["img_path"])
            out_name = f"{s['sample_id']}.png"
            out_img = img_dir / out_name
            out_msk = msk_dir / out_name
            _link_or_copy(src_img, out_img, use_symlinks=use_symlinks)
            _write_mask(
                out_msk,
                src_img,
                Path(s["mask_path"]) if s["mask_path"] else None,
            )
            ids.append(s["sample_id"])

        manifest["splits"][split_name] = {
            "dir": split_dir,
            "n_samples": len(ids),
            "sample_ids": ids,
        }
        print(f"  {split_name:5s} → {len(ids):4d} samples  ({split_dir})")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[prepare_busi_usfm] Manifest → {manifest_path}")
    return manifest


def main():
    p = argparse.ArgumentParser(description="Prepare BUSI for USFM SegBase training")
    p.add_argument(
        "--busi-root",
        type=Path,
        default="/capstor/store/cscs/swissai/a127/ultrasound/raw/breast/BUSI",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default="dataset_exploration_outputs/busi_usfm/BUSI",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinks")
    p.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_root = args.out_root if args.out_root.is_absolute() else repo / args.out_root

    prepare_split(
        busi_root=args.busi_root,
        out_root=out_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_symlinks=not args.copy,
        force=args.force,
    )


if __name__ == "__main__":
    main()
