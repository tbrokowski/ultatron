#!/usr/bin/env python3
"""
scripts/inventory.py  ·  Dataset inventory and data-count assessment
====================================================================

Produces a Markdown and JSON report of every registered dataset covering:
  - Store presence and file count
  - Scratch presence and file count
  - Approximate disk size on Store
  - Entry count in the most recent manifest (if available)
  - Whether the dataset has a registered adapter

Usage:
    python3 scripts/inventory.py
    python3 scripts/inventory.py --out inventory.json
    python3 scripts/inventory.py --manifest dataset_exploration_outputs/run1/run1_train.jsonl
    python3 scripts/inventory.py --anatomy fetal musculoskeletal

This script is intentionally standalone (no torch required) so it can be run
on CSCS login nodes or any Python ≥ 3.10 environment.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Bootstrap: avoid triggering data/__init__.py (needs torch) ───────────────
REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import types as _types

if "data" not in sys.modules:
    _stub = _types.ModuleType("data")
    _stub.__path__ = [str(REPO / "data")]  # type: ignore[attr-defined]
    _stub.__package__ = "data"
    sys.modules["data"] = _stub

from data.infra.storage import DATASET_STORE_MAP, StorageConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_count(path: Path) -> int:
    """Recursively count files under path (fast, no Python recursion)."""
    if not path.exists():
        return 0
    try:
        result = subprocess.run(
            ["find", str(path), "-type", "f"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.count("\n")
    except Exception:
        return -1


def _dir_size_gb(path: Path) -> Optional[float]:
    """Return approximate GB used by path (du -sk)."""
    if not path.exists():
        return None
    try:
        result = subprocess.run(
            ["du", "-sk", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        kb = int(result.stdout.split()[0])
        return round(kb / 1_048_576, 2)  # KB → GB
    except Exception:
        return None


def _load_manifest_counts(manifest_path: Path) -> Dict[str, int]:
    """Return {dataset_id: entry_count} from a JSONL manifest."""
    counts: Dict[str, int] = defaultdict(int)
    if not manifest_path.exists():
        return counts
    try:
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ds_id = obj.get("dataset_id") or obj.get("dataset") or ""
                    if ds_id:
                        counts[ds_id] += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return dict(counts)


def _adapter_registered(dataset_id: str) -> bool:
    """Check if dataset_id has a registered adapter (without importing adapters)."""
    adapters_init = REPO / "data" / "adapters" / "__init__.py"
    if not adapters_init.exists():
        return False
    try:
        content = adapters_init.read_text()
        return f'"{dataset_id}"' in content or f"'{dataset_id}'" in content
    except Exception:
        return False


# ── Core inventory ────────────────────────────────────────────────────────────

def build_inventory(
    cfg: StorageConfig,
    anatomy_filter: Optional[List[str]] = None,
    manifest_path: Optional[Path] = None,
    no_sizes: bool = False,
) -> List[dict]:
    manifest_counts = _load_manifest_counts(manifest_path) if manifest_path else {}

    rows = []
    for dataset_id, (anatomy, subdir) in sorted(
        DATASET_STORE_MAP.items(), key=lambda x: (x[1][0], x[0])
    ):
        if anatomy_filter and anatomy not in anatomy_filter:
            continue

        store_path = cfg.store_root / "raw" / anatomy / subdir
        store_exists = store_path.exists()
        store_files = (0 if no_sizes else _file_count(store_path)) if store_exists else 0
        store_gb = (None if no_sizes else _dir_size_gb(store_path)) if store_exists else None

        scratch_path: Optional[Path] = None
        scratch_exists = False
        scratch_files = 0
        if cfg.scratch_root:
            scratch_path = cfg.scratch_root / "raw" / anatomy / subdir
            scratch_exists = (
                scratch_path.exists() and any(scratch_path.iterdir())
                if scratch_path.exists() else False
            )
            scratch_files = (0 if no_sizes else _file_count(scratch_path)) if scratch_exists else 0

        manifest_entries = manifest_counts.get(dataset_id, 0)
        adapter_ok = _adapter_registered(dataset_id)

        rows.append({
            "dataset_id":       dataset_id,
            "anatomy":          anatomy,
            "store_subdir":     subdir,
            "store_exists":     store_exists,
            "store_files":      store_files,
            "store_gb":         store_gb,
            "scratch_exists":   scratch_exists,
            "scratch_files":    scratch_files,
            "manifest_entries": manifest_entries,
            "adapter":          adapter_ok,
            "store_path":       str(store_path),
            "scratch_path":     str(scratch_path) if scratch_path else None,
        })
    return rows


# ── Output formatting ─────────────────────────────────────────────────────────

def _status(exists: bool) -> str:
    return "✓" if exists else "✗"


def print_markdown_report(rows: List[dict], manifest_path: Optional[Path]) -> None:
    total = len(rows)
    in_store = sum(1 for r in rows if r["store_exists"])
    in_scratch = sum(1 for r in rows if r["scratch_exists"])
    total_files = sum(r["store_files"] for r in rows if r["store_files"] > 0)
    total_gb = sum(r["store_gb"] for r in rows if r["store_gb"] is not None)
    total_manifest = sum(r["manifest_entries"] for r in rows)
    with_adapter = sum(1 for r in rows if r["adapter"])

    print("# Ultatron Dataset Inventory")
    print()
    print("## Summary")
    print()
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Datasets in DATASET_STORE_MAP | {total} |")
    print(f"| Datasets in Store             | {in_store} / {total} |")
    print(f"| Datasets staged to Scratch    | {in_scratch} / {total} |")
    print(f"| Datasets with adapter         | {with_adapter} / {total} |")
    print(f"| Total files in Store          | {total_files:,} |")
    print(f"| Approximate Store size        | {total_gb:.1f} GB |")
    if manifest_path:
        print(f"| Manifest entries ({manifest_path.name}) | {total_manifest:,} |")
    print()

    # Per-anatomy summary
    by_anatomy: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_anatomy[r["anatomy"]].append(r)

    print("## By Anatomy Family")
    print()
    print(f"| Anatomy | Datasets | Store | Scratch | Files (Store) | Size (GB) | Manifest entries |")
    print(f"|---------|----------|-------|---------|---------------|-----------|-----------------|")
    for anatomy in sorted(by_anatomy):
        ds = by_anatomy[anatomy]
        a_store = sum(1 for r in ds if r["store_exists"])
        a_scratch = sum(1 for r in ds if r["scratch_exists"])
        a_files = sum(r["store_files"] for r in ds if r["store_files"] > 0)
        a_gb = sum(r["store_gb"] for r in ds if r["store_gb"] is not None)
        a_manifest = sum(r["manifest_entries"] for r in ds)
        print(f"| {anatomy:<20} | {len(ds):>8} | {a_store:>5} | {a_scratch:>7} | {a_files:>13,} | {a_gb:>9.1f} | {a_manifest:>16,} |")
    print()

    # Per-dataset table
    print("## Dataset Detail")
    print()
    print(f"| Dataset | Anatomy | Store | Scratch | Files | Size(GB) | Manifest | Adapter |")
    print(f"|---------|---------|-------|---------|-------|----------|----------|---------|")
    for r in rows:
        gb = f"{r['store_gb']:.2f}" if r["store_gb"] is not None else "—"
        mf = f"{r['manifest_entries']:,}" if r["manifest_entries"] else "—"
        files = f"{r['store_files']:,}" if r["store_files"] > 0 else "—"
        print(
            f"| {r['dataset_id']:<45} | {r['anatomy']:<20} "
            f"| {_status(r['store_exists'])} "
            f"| {_status(r['scratch_exists'])} "
            f"| {files:>9} "
            f"| {gb:>8} "
            f"| {mf:>8} "
            f"| {_status(r['adapter'])} |"
        )
    print()

    # Missing-from-store list
    missing = [r for r in rows if not r["store_exists"]]
    if missing:
        print("## Not Yet Downloaded (missing from Store)")
        print()
        for r in missing:
            print(f"- **{r['dataset_id']}** ({r['anatomy']}) — expected: `{r['store_path']}`")
        print()

    # Staged vs not-staged
    not_staged = [r for r in rows if r["store_exists"] and not r["scratch_exists"]]
    if not_staged:
        print("## In Store but NOT Staged to Scratch")
        print()
        for r in not_staged:
            print(f"- **{r['dataset_id']}** ({r['anatomy']})")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Dataset inventory and data-count report.")
    p.add_argument("--out", default=None, help="Write JSON output to this path.")
    p.add_argument(
        "--manifest", default=None,
        help="JSONL manifest to count entries per dataset. "
             "Default: dataset_exploration_outputs/run1/run1_train.jsonl",
    )
    p.add_argument(
        "--anatomy", nargs="+", default=None,
        help="Limit report to these anatomy families (e.g. fetal musculoskeletal).",
    )
    p.add_argument("--no-sizes", action="store_true", help="Skip du/find calls (faster).")
    args = p.parse_args()

    cfg = StorageConfig()

    # Default manifest
    manifest_path: Optional[Path] = None
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        default_mf = REPO / "dataset_exploration_outputs" / "run1" / "run1_train.jsonl"
        if default_mf.exists():
            manifest_path = default_mf

    print(f"Store   : {cfg.store_root}", file=sys.stderr)
    print(f"Scratch : {cfg.scratch_root}", file=sys.stderr)
    if manifest_path:
        print(f"Manifest: {manifest_path}", file=sys.stderr)
    print(file=sys.stderr)

    rows = build_inventory(
        cfg,
        anatomy_filter=args.anatomy,
        manifest_path=manifest_path,
        no_sizes=args.no_sizes,
    )

    print_markdown_report(rows, manifest_path)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "store_root": str(cfg.store_root),
                    "scratch_root": str(cfg.scratch_root),
                    "manifest": str(manifest_path) if manifest_path else None,
                    "datasets": rows,
                },
                f, indent=2,
            )
        print(f"\nJSON written to {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
