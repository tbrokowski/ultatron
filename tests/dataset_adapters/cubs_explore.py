"""
cubs_explore.py  ·  Real-data exploration for the CUBS carotid adapter
=======================================================================

Usage:
    python -m tests.dataset_adapters.cubs_explore
"""
from __future__ import annotations

import os
from pathlib import Path

from data.adapters.cubs import CUBSAdapter
from data.pipeline.dataset import ImageSSLDataset
from data.pipeline.downstream_dataset import DownstreamDataset
from data.pipeline.transforms import ImageSSLTransformConfig
from data.schema.manifest import ManifestWriter, load_manifest, manifest_stats


DEFAULT_ROOT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/vascular-carotid/CUBS"
)
DEFAULT_OUT_DIR = Path("dataset_exploration_outputs/cubs")


def get_root() -> Path:
    env = os.environ.get("US_CUBS_ROOT")
    root = Path(env) if env else DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"CUBS root not found at {root}")
    return root


def get_out_dir() -> Path:
    env = os.environ.get("US_CUBS_OUT_DIR")
    out_dir = Path(env) if env else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    root = get_root()
    out_dir = get_out_dir()
    manifest_path = out_dir / "cubs_manifest.jsonl"

    entries = list(CUBSAdapter(root).iter_entries())
    if not entries:
        raise RuntimeError(f"No entries produced for {root}")

    with ManifestWriter(manifest_path) as writer:
        for entry in entries:
            writer.write(entry)

    print(f"[CUBS] root: {root}")
    print(f"[CUBS] manifest: {manifest_path}")
    print(f"[CUBS] entries: {len(entries)}")
    for key, value in manifest_stats(entries).items():
        print(f"  {key}: {value}")

    values = [e.instances[0].measurement_mm for e in entries if e.instances]
    values = [v for v in values if v is not None]
    if values:
        print(
            f"[CUBS] IMT mm range: min={min(values):.3f}"
            f" mean={sum(values) / len(values):.3f}"
            f" max={max(values):.3f}"
        )

    reloaded = load_manifest(manifest_path)
    img_ds = ImageSSLDataset(
        reloaded[:4],
        cfg=ImageSSLTransformConfig(
            n_global_crops=2,
            n_local_crops=2,
            max_global_crop_px=128,
            min_crop_px=32,
        ),
    )
    for idx in range(min(2, len(img_ds))):
        sample = img_ds[idx]
        print(
            f"[CUBS][ImageSSL] idx={idx}"
            f" global0={tuple(sample['global_crops'][0].shape)}"
            f" locals={len(sample['local_crops'])}"
            f" task={sample['task_type']}"
        )

    sample = DownstreamDataset([reloaded[0]])[0]
    print(
        f"[CUBS][Downstream] image={tuple(sample['image'].shape)}"
        f" targets={[t.head_id for t in sample['label_targets']]}"
        f" promptable={sample['is_promptable']}"
    )
    print("[cubs_explore] Complete.")


if __name__ == "__main__":
    main()
