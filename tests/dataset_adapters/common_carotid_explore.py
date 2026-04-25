"""
common_carotid_explore.py  ·  Real-data exploration for the expert-mask carotid adapter
=========================================================================================

Usage:
    python -m tests.dataset_adapters.common_carotid_explore
"""
from __future__ import annotations

import os
from pathlib import Path

from data.adapters.common_carotid import CommonCarotidArteryImagesAdapter
from data.pipeline.dataset import ImageSSLDataset
from data.pipeline.downstream_dataset import DownstreamDataset
from data.pipeline.transforms import ImageSSLTransformConfig
from data.schema.manifest import ManifestWriter, load_manifest, manifest_stats


DEFAULT_ROOT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/vascular-carotid/Common-Carotid-Artery-Ultrasound-Images"
)
DEFAULT_OUT_DIR = Path("dataset_exploration_outputs/common_carotid")


def get_root() -> Path:
    env = os.environ.get("US_COMMON_CAROTID_ROOT")
    root = Path(env) if env else DEFAULT_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Common carotid root not found at {root}")
    return root


def get_out_dir() -> Path:
    env = os.environ.get("US_COMMON_CAROTID_OUT_DIR")
    out_dir = Path(env) if env else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    root = get_root()
    out_dir = get_out_dir()
    manifest_path = out_dir / "common_carotid_manifest.jsonl"

    entries = list(CommonCarotidArteryImagesAdapter(root).iter_entries())
    if not entries:
        raise RuntimeError(f"No entries produced for {root}")

    with ManifestWriter(manifest_path) as writer:
        for entry in entries:
            writer.write(entry)

    print(f"[Common-Carotid] root: {root}")
    print(f"[Common-Carotid] manifest: {manifest_path}")
    print(f"[Common-Carotid] entries: {len(entries)}")
    for key, value in manifest_stats(entries).items():
        print(f"  {key}: {value}")

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
            f"[Common-Carotid][ImageSSL] idx={idx}"
            f" global0={tuple(sample['global_crops'][0].shape)}"
            f" locals={len(sample['local_crops'])}"
            f" task={sample['task_type']}"
        )

    sample = DownstreamDataset([reloaded[0]])[0]
    print(
        f"[Common-Carotid][Downstream] image={tuple(sample['image'].shape)}"
        f" targets={[t.head_id for t in sample['label_targets']]}"
        f" promptable={sample['is_promptable']}"
    )
    print("[common_carotid_explore] Complete.")


if __name__ == "__main__":
    main()
