#!/usr/bin/env python3
"""
build_shards.py  ·  WebDataset tar shards (~1–2 GB) for POCUS bench
===================================================================

Images: original encoded bytes + JSON sidecar.
Videos: keep encoded bytes (MP4 / AVI / nii.gz); decode in the loader.
Drops unreadable files at build time (spec §2.4).

COVID-BLUES is CC BY-NC-ND: shards stay on the CSCS store and must not be
redistributed.  CardiacUDC is stored in native nii.gz (existing adapter).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pocus.paths import MANIFEST_ROOT, SHARD_ROOT
from scripts.pocus.sampling import clip_plan, load_jsonl


SHARD_BYTES_TARGET = 1_500_000_000  # 1.5 GB
SKIP_SUFFIXES = {".wmv", ".mpeg", ".mpg", ".asf"}


def _primary_path(rec: dict) -> Optional[Path]:
    for key in ("image", "video_path", "path"):
        v = rec.get(key)
        if isinstance(v, str) and v:
            return Path(v)
        if isinstance(v, list) and v:
            return Path(v[0])
    imgs = rec.get("image_paths")
    if isinstance(imgs, list) and imgs:
        return Path(imgs[0])
    return None


def _sidecar(rec: dict, path: Path, raw: bytes, extra: dict) -> dict:
    plan = rec.get("clip_plan")
    n_frames = rec.get("n_frames") or rec.get("num_frames")
    if plan is None and n_frames:
        plan = clip_plan(int(n_frames))
    return {
        "sample_id": rec.get("sample_id") or rec.get("study_id") or path.stem,
        "dataset": rec.get("dataset_id") or rec.get("dataset"),
        "split": rec.get("split", "train"),
        "body_system": rec.get("body_system") or rec.get("anatomy_family"),
        "organ": rec.get("organ"),
        "caption": rec.get("caption") or (rec.get("source_meta") or {}).get("caption")
        or (rec.get("source_meta") or {}).get("report_text"),
        "attributes": rec.get("attributes") or rec.get("source_meta"),
        "H": rec.get("height") or rec.get("H"),
        "W": rec.get("width") or rec.get("W"),
        "bytes": len(raw),
        "n_frames": n_frames,
        "fps": rec.get("fps"),
        "labels": rec.get("labels") or rec.get("source_meta"),
        "path": str(path),
        "clip_plan": plan,
        **extra,
    }


def _open_next_shard(out_dir: Path, idx: int) -> Tuple[tarfile.TarFile, Path]:
    path = out_dir / f"shard_{idx:05d}.tar"
    return tarfile.open(path, "w"), path


def shard_manifest(
    rows: List[dict],
    out_dir: Path,
    prefix: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dropped: List[dict] = []
    kept = 0
    shard_idx = 0
    tar, tar_path = _open_next_shard(out_dir, shard_idx)
    running = 0
    shard_files = [tar_path]

    try:
        for rec in rows:
            path = _primary_path(rec)
            if path is None:
                dropped.append({"reason": "no_path", "sample": rec.get("sample_id")})
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                dropped.append({"reason": "bad_suffix", "path": str(path)})
                continue
            if not path.is_file():
                dropped.append({"reason": "missing", "path": str(path)})
                continue
            try:
                raw = path.read_bytes()
            except OSError as exc:
                dropped.append({"reason": "unreadable", "path": str(path), "error": str(exc)})
                continue
            if len(raw) < 32:
                dropped.append({"reason": "too_small", "path": str(path), "bytes": len(raw)})
                continue
            key = rec.get("sample_id") or path.stem
            sidecar = json.dumps(_sidecar(rec, path, raw, {"prefix": prefix})).encode("utf-8")
            payload = raw + sidecar
            if running + len(payload) > SHARD_BYTES_TARGET and kept > 0:
                tar.close()
                shard_idx += 1
                tar, tar_path = _open_next_shard(out_dir, shard_idx)
                shard_files.append(tar_path)
                running = 0
            ext = path.suffix.lstrip(".") or "bin"
            for name, data in ((f"{key}.{ext}", raw), (f"{key}.json", sidecar)):
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mtime = int(time.time())
                tar.addfile(info, io.BytesIO(data))
            running += len(payload)
            kept += 1
    finally:
        tar.close()

    dropped_path = out_dir / "dropped.jsonl"
    with dropped_path.open("w", encoding="utf-8") as f:
        for d in dropped:
            f.write(json.dumps(d) + "\n")
    summary = {
        "prefix": prefix,
        "kept": kept,
        "dropped": len(dropped),
        "drop_reasons": _count_reasons(dropped),
        "n_shards": len(shard_files),
        "shard_bytes": sum(p.stat().st_size for p in shard_files if p.exists()),
        "out_dir": str(out_dir),
    }
    (out_dir / "shard_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _count_reasons(dropped: List[dict]) -> dict:
    c: Dict[str, int] = {}
    for d in dropped:
        c[d.get("reason", "?")] = c.get(d.get("reason", "?"), 0) + 1
    return c


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", type=Path, default=MANIFEST_ROOT)
    p.add_argument("--out", type=Path, default=SHARD_ROOT)
    p.add_argument("--which", nargs="+", default=["enc_images", "enc_videos"])
    args = p.parse_args()

    dropped_all: Dict[str, Any] = {}
    for name in args.which:
        jsonl = args.manifests / f"{name}.jsonl"
        if not jsonl.exists():
            print(f"[WARN] missing {jsonl}", file=sys.stderr)
            continue
        rows = load_jsonl(jsonl)
        summary = shard_manifest(rows, args.out / name, name)
        dropped_all[name] = summary
        print(json.dumps(summary, indent=2))
    (args.out / "dropped.json").write_text(json.dumps(dropped_all, indent=2) + "\n")


if __name__ == "__main__":
    main()
