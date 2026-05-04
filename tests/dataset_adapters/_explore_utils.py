"""
_explore_utils.py  ·  Shared helpers for manual real-data adapter exploration
==============================================================================
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Type

from data.adapters.base import BaseAdapter
from data.schema.manifest import ManifestWriter, manifest_stats


def get_root(default_root: Path, env_var: str) -> Path:
    env = os.environ.get(env_var)
    root = Path(env) if env else default_root
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found at {root}")
    return root


def get_out_dir(default_out_dir: Path, env_var: str) -> Path:
    env = os.environ.get(env_var)
    out_dir = Path(env) if env else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_adapter_explore(
    *,
    dataset_name: str,
    adapter_cls: Type[BaseAdapter],
    default_root: Path,
    default_out_dir: Path,
    root_env_var: str,
    out_dir_env_var: str,
) -> None:
    root = get_root(default_root, root_env_var)
    out_dir = get_out_dir(default_out_dir, out_dir_env_var)
    manifest_path = out_dir / f"{dataset_name.lower().replace('-', '_')}_manifest.jsonl"

    entries = list(adapter_cls(root).iter_entries())
    if not entries:
        raise RuntimeError(f"No entries produced for {root}")

    with ManifestWriter(manifest_path) as writer:
        for entry in entries:
            writer.write(entry)

    print(f"[{dataset_name}] root: {root}")
    print(f"[{dataset_name}] manifest: {manifest_path}")
    print(f"[{dataset_name}] entries: {len(entries)}")
    for key, value in manifest_stats(entries).items():
        print(f"  {key}: {value}")

    preview = entries[:3]
    for idx, entry in enumerate(preview):
        print(
            f"[{dataset_name}] sample[{idx}]"
            f" split={entry.split}"
            f" modality={entry.modality_type}"
            f" task={entry.task_type}"
            f" study={entry.study_id}"
            f" path={entry.image_paths[0]}"
        )

