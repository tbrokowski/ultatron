"""
scripts/pocus/sampling.py  ·  Stratified manifests and clip-length fallbacks
============================================================================

Seed 1234 (spec §2.3).  Image strata: body_system × organ when those fields
exist; otherwise a single 'unknown/unknown' stratum (uniform draw).
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


ATTRIBUTE_KEYS = (
    "body_system", "body-system", "system",
    "organ", "organs",
    "attributes", "structured_attributes",
    "diagnosis", "findings", "view", "probe",
)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def first_key(rec: dict, *names: str) -> Any:
    for n in names:
        if n in rec and rec[n] not in (None, "", []):
            return rec[n]
        meta = rec.get("source_meta") or {}
        if isinstance(meta, dict) and n in meta and meta[n] not in (None, "", []):
            return meta[n]
    return None


def stratum_key(rec: dict) -> Tuple[str, str]:
    bs = first_key(rec, "body_system", "body-system", "system", "anatomy_family")
    organ = first_key(rec, "organ", "organs")
    return (str(bs or "unknown"), str(organ or "unknown"))


def metadata_field_report(rows: Sequence[dict]) -> dict:
    """Which of the 9 structured attribute fields appear in US-365K records."""
    counts: Counter = Counter()
    key_union: Counter = Counter()
    n = 0
    for rec in rows:
        n += 1
        keys = set(rec.keys())
        meta = rec.get("source_meta") if isinstance(rec.get("source_meta"), dict) else {}
        keys |= set(meta.keys())
        for k in keys:
            key_union[k] += 1
        for k in ATTRIBUTE_KEYS:
            if k in keys:
                counts[k] += 1
    structured = [k for k in ATTRIBUTE_KEYS if counts[k] > 0]
    return {
        "n_records": n,
        "attribute_key_counts": dict(counts),
        "all_key_counts": dict(key_union.most_common(40)),
        "has_structured_attributes": bool(structured),
        "structured_keys_present": structured,
        "captions_only": (not structured) and (key_union.get("caption", 0) > 0 or key_union.get("report_text", 0) > 0),
    }


def stratified_sample(
    rows: Sequence[dict],
    n: int,
    *,
    keyfn: Callable[[dict], Any] = stratum_key,
    seed: int = 1234,
) -> List[dict]:
    """Proportional stratified sample without replacement. Falls back to uniform."""
    if n <= 0:
        return []
    if n >= len(rows):
        rng = random.Random(seed)
        out = list(rows)
        rng.shuffle(out)
        return out
    groups: Dict[Any, List[dict]] = defaultdict(list)
    for r in rows:
        groups[keyfn(r)].append(r)
    rng = random.Random(seed)
    total = len(rows)
    alloc: Dict[Any, int] = {}
    # Largest remainder method.
    remainders = []
    taken = 0
    for k, items in groups.items():
        raw = n * len(items) / total
        base = int(math.floor(raw))
        alloc[k] = min(base, len(items))
        taken += alloc[k]
        remainders.append((raw - base, k))
    remainders.sort(reverse=True)
    i = 0
    while taken < n and remainders:
        _, k = remainders[i % len(remainders)]
        if alloc[k] < len(groups[k]):
            alloc[k] += 1
            taken += 1
        i += 1
        if i > n * len(remainders) + 8:
            break
    sampled: List[dict] = []
    leftover: List[dict] = []
    for k, items in groups.items():
        rng.shuffle(items)
        k_n = alloc.get(k, 0)
        sampled.extend(items[:k_n])
        leftover.extend(items[k_n:])
    if len(sampled) < n:
        rng.shuffle(leftover)
        sampled.extend(leftover[: n - len(sampled)])
    rng.shuffle(sampled)
    return sampled[:n]


def equal_per_dataset(rows: Sequence[dict], *, seed: int = 1234) -> List[dict]:
    """Round-robin mix so each dataset is sampled equally per step."""
    by_ds: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        ds = str(r.get("dataset_id") or r.get("dataset") or "unknown")
        by_ds[ds].append(r)
    rng = random.Random(seed)
    for items in by_ds.values():
        rng.shuffle(items)
    keys = sorted(by_ds)
    idxs = {k: 0 for k in keys}
    out: List[dict] = []
    remaining = True
    while remaining:
        remaining = False
        for k in keys:
            i = idxs[k]
            if i < len(by_ds[k]):
                out.append(by_ds[k][i])
                idxs[k] = i + 1
                remaining = True
    return out


def clip_plan(n_frames: int, n_out: int = 32, span: int = 64) -> dict:
    """
    Production clip rule (spec §2.4): 32 frames at stride 2 (64-frame span).

    * n_frames ≥ 64 → stride 2
    * 32 ≤ n_frames < 64 → stride 1
    * n_frames < 32 → stride 1 and pad by frame repetition
    """
    n_frames = int(n_frames or 0)
    if n_frames >= span:
        return {
            "n_frames": n_frames,
            "n_out": n_out,
            "stride": 2,
            "pad_repeat": False,
            "fallback": "none",
        }
    if n_frames >= n_out:
        return {
            "n_frames": n_frames,
            "n_out": n_out,
            "stride": 1,
            "pad_repeat": False,
            "fallback": "stride1",
        }
    return {
        "n_frames": n_frames,
        "n_out": n_out,
        "stride": 1,
        "pad_repeat": True,
        "fallback": "pad_repeat",
    }


def summarise_clip_fallbacks(plans: Sequence[dict]) -> dict:
    c = Counter(p.get("fallback", "none") for p in plans)
    n = len(plans) or 1
    return {
        "n_clips": len(plans),
        "counts": dict(c),
        "fractions": {k: v / n for k, v in c.items()},
    }
