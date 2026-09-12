#!/usr/bin/env python3
"""
data_facts.py  ·  Data and I/O facts for the feasibility review (spec §2.5)
===========================================================================

Per dataset and per manifest: sample/file counts, bytes (raw and sharded),
mean and 95th-percentile bytes per image/clip, histograms of resolution,
frames/clip and fps, licence and sensitivity.

Output: data_facts.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pocus.paths import LICENCES, MANIFEST_ROOT, RAW_ROOT, SHARD_ROOT
from scripts.pocus.sampling import load_jsonl


def _percentile(xs: Sequence[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return float(ys[0])
    k = (len(ys) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(ys[int(k)])
    return float(ys[f] * (c - k) + ys[c] * (k - f))


def _hist(values: Sequence[Any], bins: Optional[List] = None) -> dict:
    if not values:
        return {}
    if bins is None:
        return dict(Counter(values))
    c: Counter = Counter()
    for v in values:
        placed = False
        for b in bins:
            if v <= b:
                c[f"≤{b}"] += 1
                placed = True
                break
        if not placed:
            c[f">{bins[-1]}"] += 1
    return dict(c)


def _file_size(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    p = Path(path)
    try:
        return p.stat().st_size if p.is_file() else None
    except OSError:
        return None


def _walk_bytes(root: Path, max_files: int = 2_000_000) -> dict:
    n_files = 0
    total = 0
    if not root.exists():
        return {"exists": False, "n_files": 0, "bytes": 0}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        n_files += 1
        try:
            total += p.stat().st_size
        except OSError:
            pass
        if n_files >= max_files:
            break
    return {"exists": True, "n_files": n_files, "bytes": total, "root": str(root)}


def facts_for_manifest(path: Path, dataset_hint: str = "") -> dict:
    rows = load_jsonl(path) if path.exists() else []
    byte_sizes: List[int] = []
    heights: List[int] = []
    widths: List[int] = []
    n_frames: List[int] = []
    fps: List[float] = []
    unreadable = 0
    for rec in rows:
        paths = []
        for key in ("image", "image_paths", "video_path", "path"):
            v = rec.get(key)
            if isinstance(v, list):
                paths.extend(v)
            elif isinstance(v, str):
                paths.append(v)
        sz = None
        for p in paths:
            s = _file_size(p)
            if s is not None:
                sz = (sz or 0) + s
        if sz is None:
            # sidecar may already store bytes
            if rec.get("bytes"):
                sz = int(rec["bytes"])
            else:
                unreadable += 1
        else:
            byte_sizes.append(sz)
        h = rec.get("height") or rec.get("H")
        w = rec.get("width") or rec.get("W")
        if h:
            heights.append(int(h))
        if w:
            widths.append(int(w))
        nf = rec.get("n_frames") or rec.get("num_frames")
        if nf:
            n_frames.append(int(nf))
        f = rec.get("fps")
        if f:
            fps.append(float(f))
    ds_ids = Counter(str(r.get("dataset_id") or r.get("dataset") or dataset_hint) for r in rows)
    return {
        "path": str(path),
        "sample_count": len(rows),
        "file_count_readable": len(byte_sizes),
        "unreadable_or_missing": unreadable,
        "bytes_sum": int(sum(byte_sizes)),
        "bytes_mean": (sum(byte_sizes) / len(byte_sizes)) if byte_sizes else None,
        "bytes_p95": _percentile(byte_sizes, 0.95) if byte_sizes else None,
        "resolution_hist_h": _hist(heights, [256, 512, 640, 800, 1024, 1280]),
        "resolution_hist_w": _hist(widths, [256, 512, 640, 800, 1024, 1280]),
        "frames_hist": _hist(n_frames, [16, 32, 64, 100, 200, 500]),
        "fps_hist": _hist([round(x) for x in fps], [15, 24, 25, 30, 50, 60]),
        "by_dataset": dict(ds_ids),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifests", type=Path, default=MANIFEST_ROOT)
    p.add_argument("--raw", type=Path, default=RAW_ROOT)
    p.add_argument("--shards", type=Path, default=SHARD_ROOT)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    facts: Dict[str, Any] = {"datasets": {}, "manifests": {}, "shards": {}}
    for ds, lic in LICENCES.items():
        raw = args.raw / ds
        facts["datasets"][ds] = {
            **lic,
            "raw": _walk_bytes(raw),
        }
    if args.manifests.exists():
        for jsonl in sorted(args.manifests.glob("*.jsonl")):
            facts["manifests"][jsonl.stem] = facts_for_manifest(jsonl)
    if args.shards.exists():
        for ds_dir in sorted(p for p in args.shards.iterdir() if p.is_dir()):
            tars = list(ds_dir.glob("*.tar"))
            facts["shards"][ds_dir.name] = {
                "n_shards": len(tars),
                "bytes": sum(t.stat().st_size for t in tars if t.is_file()),
                "file_count": len(tars),
            }
        dropped = args.shards / "dropped.json"
        if dropped.exists():
            facts["dropped"] = json.loads(dropped.read_text(encoding="utf-8"))
    out = args.out or (args.manifests / "data_facts.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
