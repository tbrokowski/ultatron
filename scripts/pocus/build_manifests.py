#!/usr/bin/env python3
"""
build_manifests.py  ·  Stratified POCUS manifests (seed 1234)
=============================================================

Writes:
  enc_images.jsonl       100k US-365K train images, stratified
  enc_videos.jsonl       all train-split videos of CardiacUDC / COVID-BLUES / IUGC
  rl_prompts.jsonl       20k US-365K train images
  rl_heldout.jsonl       1k US-365K val images
  rl_video_prompts.jsonl 200 clips (optional R5)

Also records image:video and cardiac:lung:obstetric mix vs a production
manifest if --production-manifest is given.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pocus.paths import (
    DATASET_DEFAULTS,
    MANIFEST_ROOT,
    N_ENC_IMAGES,
    N_RL_HELDOUT,
    N_RL_PROMPTS,
    N_RL_VIDEO,
    SEED,
)
from scripts.pocus.sampling import (
    clip_plan,
    equal_per_dataset,
    load_jsonl,
    metadata_field_report,
    stratified_sample,
    summarise_clip_fallbacks,
    write_jsonl,
)


def _adapter_entries(dataset_id: str, root: Path, split: Optional[str] = None):
    from data.adapters import ADAPTER_REGISTRY
    cls = ADAPTER_REGISTRY[dataset_id]
    adapter = cls(root, split_override=split) if split else cls(root)
    for e in adapter.iter_entries():
        d = e.to_dict() if hasattr(e, "to_dict") else dict(e)
        d.setdefault("dataset_id", dataset_id)
        yield d


def _us365k_rows(root: Path, split: str) -> List[dict]:
    jsonl = root / "metadata" / f"{split}.jsonl"
    if jsonl.exists():
        rows = []
        for rec in load_jsonl(jsonl):
            rec.setdefault("split", split)
            rec.setdefault("dataset_id", "US-365K")
            rec.setdefault("dataset", "US-365K")
            if "sample_id" not in rec:
                img = rec.get("image") or rec.get("image_fname") or ""
                rec["sample_id"] = Path(str(img)).stem
            rows.append(rec)
        return rows
    return list(_adapter_entries("US-365K", root, split=split))


def _video_rows(dataset_id: str, root: Path) -> List[dict]:
    rows = []
    try:
        for rec in _adapter_entries(dataset_id, root):
            split = rec.get("split", "train")
            if split in ("val", "test"):
                continue
            rec.setdefault("dataset", dataset_id)
            rec.setdefault("n_frames", rec.get("num_frames", 0))
            rec["clip_plan"] = clip_plan(int(rec.get("n_frames") or rec.get("num_frames") or 0))
            rows.append(rec)
    except FileNotFoundError as exc:
        print(f"[WARN] {dataset_id}: {exc}", file=sys.stderr)
        return rows
    return rows


def _mix_report(rows: List[dict], label: str) -> dict:
    ds = Counter(str(r.get("dataset_id") or r.get("dataset") or "?") for r in rows)
    streams = Counter(str(r.get("ssl_stream") or r.get("modality_type") or "?") for r in rows)
    return {"label": label, "n": len(rows), "by_dataset": dict(ds), "by_stream": dict(streams)}


def build(args: argparse.Namespace) -> dict:
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    us_root = Path(args.us365k)
    train = _us365k_rows(us_root, "train")
    val = _us365k_rows(us_root, "val")
    attr_report = metadata_field_report(train[: min(len(train), 8000)])

    enc_images = stratified_sample(train, args.n_images, seed=args.seed)
    rl_prompts = stratified_sample(train, args.n_prompts, seed=args.seed + 1)
    rl_heldout = stratified_sample(val, args.n_heldout, seed=args.seed + 2)

    videos = []
    for ds in ("CardiacUDC", "COVID-BLUES", "IUGC2024"):
        root = Path(getattr(args, ds.lower().replace("-", "_"), DATASET_DEFAULTS[ds]))
        part = _video_rows(ds, root)
        print(f"[info] {ds}: {len(part)} train/unsplit clips from {root}")
        videos.extend(part)
    enc_videos = equal_per_dataset(videos, seed=args.seed)
    clip_fb = summarise_clip_fallbacks([v.get("clip_plan") or clip_plan(0) for v in enc_videos])

    rl_video = []
    if args.n_video > 0 and videos:
        # 200 clips stratified by dataset (equal).
        by_ds = {}
        for v in videos:
            by_ds.setdefault(v.get("dataset_id") or v.get("dataset"), []).append(v)
        per = max(1, args.n_video // max(1, len(by_ds)))
        for ds, items in sorted(by_ds.items()):
            rl_video.extend(stratified_sample(items, per, keyfn=lambda r: r.get("split", "train"), seed=args.seed))
        rl_video = rl_video[: args.n_video]

    paths = {
        "enc_images": out_dir / "enc_images.jsonl",
        "enc_videos": out_dir / "enc_videos.jsonl",
        "rl_prompts": out_dir / "rl_prompts.jsonl",
        "rl_heldout": out_dir / "rl_heldout.jsonl",
        "rl_video_prompts": out_dir / "rl_video_prompts.jsonl",
    }
    counts = {
        "enc_images": write_jsonl(paths["enc_images"], enc_images),
        "enc_videos": write_jsonl(paths["enc_videos"], enc_videos),
        "rl_prompts": write_jsonl(paths["rl_prompts"], rl_prompts),
        "rl_heldout": write_jsonl(paths["rl_heldout"], rl_heldout),
        "rl_video_prompts": write_jsonl(paths["rl_video_prompts"], rl_video),
    }

    mix = {
        "enc_images": _mix_report(enc_images, "enc_images"),
        "enc_videos": _mix_report(enc_videos, "enc_videos"),
        "n_images": counts["enc_images"],
        "n_videos": counts["enc_videos"],
        "image_video_ratio": (
            counts["enc_images"] / counts["enc_videos"] if counts["enc_videos"] else None
        ),
        "cardiac_lung_obstetric": {
            "CardiacUDC": sum(1 for v in enc_videos if (v.get("dataset_id") or v.get("dataset")) == "CardiacUDC"),
            "COVID-BLUES": sum(1 for v in enc_videos if (v.get("dataset_id") or v.get("dataset")) == "COVID-BLUES"),
            "IUGC2024": sum(1 for v in enc_videos if (v.get("dataset_id") or v.get("dataset")) == "IUGC2024"),
        },
    }
    if args.production_manifest and Path(args.production_manifest).exists():
        prod = load_jsonl(Path(args.production_manifest))
        mix["production"] = _mix_report(prod, "production")

    summary = {
        "seed": args.seed,
        "us365k_root": str(us_root),
        "attribute_report": attr_report,
        "counts": counts,
        "paths": {k: str(v) for k, v in paths.items()},
        "mix": mix,
        "clip_fallbacks": clip_fb,
    }
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": counts, "clip_fallbacks": clip_fb, "captions_only": attr_report.get("captions_only")}, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=MANIFEST_ROOT)
    p.add_argument("--us365k", type=Path, default=DATASET_DEFAULTS["US-365K"])
    p.add_argument("--cardiacudc", type=Path, default=DATASET_DEFAULTS["CardiacUDC"])
    p.add_argument("--covid_blues", type=Path, default=DATASET_DEFAULTS["COVID-BLUES"])
    p.add_argument("--iugc2024", type=Path, default=DATASET_DEFAULTS["IUGC2024"])
    p.add_argument("--n-images", type=int, default=N_ENC_IMAGES)
    p.add_argument("--n-prompts", type=int, default=N_RL_PROMPTS)
    p.add_argument("--n-heldout", type=int, default=N_RL_HELDOUT)
    p.add_argument("--n-video", type=int, default=N_RL_VIDEO)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--production-manifest", type=Path, default=None)
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
