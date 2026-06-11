#!/usr/bin/env python3
"""
materialize_us365k.py  ·  Normalize HF US-365K snapshot to Ultatron layout
==========================================================================

Supports the published HF layout (images.zip + split JSONL) and legacy parquet
shards if present. Writes:
  {target}/images/{filename}
  {target}/metadata/{train,val,test}.jsonl

Each JSONL line: {"image": "<abs path>", "caption": "...", "split": "train"}
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "val": "val",
    "valid": "val",
    "test": "test",
}

HF_JSONL_SPLITS = {
    "train": "train.jsonl",
    "val": "valid.jsonl",
    "test": "test.jsonl",
}


def _split_from_path(path: Path) -> str | None:
    name = path.name.lower()
    for token, split in SPLIT_MAP.items():
        if token in name:
            return split
    return None


def _find_parquet_files(snapshot_dir: Path) -> dict[str, list[Path]]:
    by_split: dict[str, list[Path]] = {s: [] for s in ("train", "val", "test")}
    for pq in sorted(snapshot_dir.rglob("*.parquet")):
        split = _split_from_path(pq)
        if split:
            by_split[split].append(pq)
    return by_split


def _image_filename(image_val, row_idx: int, split: str) -> str:
    if isinstance(image_val, str):
        return Path(image_val).name
    if isinstance(image_val, dict):
        if image_val.get("path"):
            return Path(str(image_val["path"])).name
        return f"{split}_{row_idx:08d}.jpg"
    return f"{split}_{row_idx:08d}.jpg"


def _write_image(image_val, dst: Path, snapshot_dir: Path) -> bool:
    if dst.exists():
        return True

    if isinstance(image_val, dict) and image_val.get("bytes"):
        dst.write_bytes(image_val["bytes"])
        return True

    if isinstance(image_val, str):
        fname = Path(image_val).name
        for candidate in (
            snapshot_dir / image_val,
            snapshot_dir / fname,
            snapshot_dir / "images" / fname,
            snapshot_dir / "data" / fname,
        ):
            if candidate.is_file():
                shutil.copy2(candidate, dst)
                return True
        found = next(snapshot_dir.rglob(fname), None)
        if found and found.is_file():
            shutil.copy2(found, dst)
            return True

    return False


def _extract_images_zip(zip_path: Path, images_dir: Path) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir() or not member.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            dst = images_dir / Path(member.filename).name
            if dst.exists():
                continue
            with zf.open(member) as src, dst.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1
    return extracted


def _materialize_from_zip_jsonl(
    target: Path,
    snapshot_dir: Path,
) -> dict[str, int]:
    zip_path = snapshot_dir / "images.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"images.zip not found under {snapshot_dir}")

    images_dir = target / "images"
    meta_dir = target / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    existing_images = sum(1 for _ in images_dir.glob("*")) if images_dir.is_dir() else 0
    if existing_images < 1000:
        print(f"Extracting images from {zip_path} → {images_dir}")
        extracted = _extract_images_zip(zip_path, images_dir)
        print(f"  extracted {extracted:,} new images ({existing_images:,} already present)")
    else:
        print(f"Skipping zip extraction; {existing_images:,} images already in {images_dir}")

    stats: dict[str, int] = {
        "rows_read": 0,
        "metadata_written": 0,
        "skipped_no_image": 0,
        "skipped_no_caption": 0,
    }

    for split, src_name in HF_JSONL_SPLITS.items():
        src_jsonl = snapshot_dir / src_name
        if not src_jsonl.is_file():
            print(f"  warning: missing {src_jsonl}, skipping {split}")
            continue

        out_jsonl = meta_dir / f"{split}.jsonl"
        n_written = 0
        with src_jsonl.open(encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                stats["rows_read"] += 1
                rec = json.loads(line)
                fname = Path(str(rec.get("image", ""))).name
                caption = (rec.get("caption") or "").strip()
                if not fname:
                    stats["skipped_no_image"] += 1
                    continue
                if not caption:
                    stats["skipped_no_caption"] += 1
                    continue

                img_path = images_dir / fname
                if not img_path.is_file():
                    stats["skipped_no_image"] += 1
                    continue

                record = {
                    "image": str(img_path),
                    "caption": caption,
                    "split": split,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

        stats[split] = n_written

    return stats


def _materialize_from_parquet(
    target: Path,
    snapshot_dir: Path,
) -> dict[str, int]:
    import pyarrow.parquet as pq

    images_dir = target / "images"
    meta_dir = target / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    parquet_by_split = _find_parquet_files(snapshot_dir)
    if not any(parquet_by_split.values()):
        raise RuntimeError(f"No parquet files found under {snapshot_dir}")

    stats: dict[str, int] = {
        "rows_read": 0,
        "images_written": 0,
        "skipped_no_image": 0,
        "skipped_no_caption": 0,
    }

    for split, parquet_files in parquet_by_split.items():
        if not parquet_files:
            continue
        out_jsonl = meta_dir / f"{split}.jsonl"
        n_written = 0
        with out_jsonl.open("w", encoding="utf-8") as fout:
            for pq_path in parquet_files:
                table = pq.read_table(pq_path)
                data = table.to_pydict()
                n_rows = len(next(iter(data.values()))) if data else 0
                for i in range(n_rows):
                    stats["rows_read"] += 1
                    image_val = data.get("image", [None] * n_rows)[i]
                    caption = data.get("caption", [""] * n_rows)[i] or ""
                    if not str(caption).strip():
                        stats["skipped_no_caption"] += 1
                        continue

                    fname = _image_filename(image_val, i, split)
                    img_path = images_dir / fname
                    if not _write_image(image_val, img_path, snapshot_dir):
                        stats["skipped_no_image"] += 1
                        continue

                    stats["images_written"] += 1
                    record = {
                        "image": str(img_path),
                        "caption": str(caption),
                        "split": split,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1
        stats[split] = n_written

    return stats


def materialize(target: Path, snapshot_dir: Path | None = None) -> dict[str, int]:
    target = target.resolve()
    snapshot_dir = (snapshot_dir or target / "hf_snapshot").resolve()
    if not snapshot_dir.is_dir():
        raise FileNotFoundError(f"HF snapshot not found: {snapshot_dir}")

    if (snapshot_dir / "images.zip").is_file():
        return _materialize_from_zip_jsonl(target, snapshot_dir)

    parquet_by_split = _find_parquet_files(snapshot_dir)
    if any(parquet_by_split.values()):
        return _materialize_from_parquet(target, snapshot_dir)

    raise RuntimeError(
        f"No supported US-365K layout under {snapshot_dir} "
        "(expected images.zip + JSONL or parquet shards)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize US-365K HF snapshot")
    parser.add_argument(
        "--target",
        default="/capstor/store/cscs/swissai/a127/ultrasound/raw/multi_organ/US-365K",
        help="Dataset root (contains hf_snapshot/, will create images/ and metadata/)",
    )
    parser.add_argument("--snapshot", default=None, help="Override hf_snapshot path")
    args = parser.parse_args()

    target = Path(args.target)
    snapshot = Path(args.snapshot) if args.snapshot else None
    stats = materialize(target, snapshot)

    print("Materialization complete:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
