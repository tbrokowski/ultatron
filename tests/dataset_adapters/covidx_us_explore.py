"""
covidx_us_explore.py  ·  Real-data exploration for COVIDx-US lung ultrasound
===========================================================================

Usage (from project root):

    python -m tests.dataset_adapters.covidx_us_explore

This script:
  1. Uses the COVIDxUSAdapter on the real COVIDx-US dataset on CSCS.
  2. Writes a small manifest JSONL with all COVIDx-US entries.
  3. Runs a VideoSSL dataset (or equivalent) on a subset.
  4. Saves PNG snapshots of raw frames and transformed clips.

It is an exploratory tool and is NOT part of the automated pytest suite.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from data.adapters.dataset_adapters import COVIDxUSAdapter
from data.schema.manifest import (
    ManifestWriter,
    USManifestEntry,
    load_manifest,
    manifest_stats,
)
from data.labels.label_spec import TaskConfig
from data.pipeline.dataset import VideoSSLDataset, load_video_frames
from data.pipeline.transforms import VideoSSLTransformConfig
from data.infra.storage import configure_storage


# ── Configuration ──────────────────────────────────────────────────────────────

# Default COVIDx-US root on CSCS Store; can be overridden via US_COVIDX_ROOT.
DEFAULT_COVIDX_ROOT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/COVIDx-US"
)

# Output directory for manifests and PNGs (relative to repo root by default)
DEFAULT_OUT_DIR = Path("covidx_exploration_outputs")

# How many clips / frames to visualise
N_VIS_CLIPS = 8
N_FRAMES_PER_CLIP = 3  # first / middle / last


def get_covidx_root() -> Path:
    """Resolve COVIDx-US root from env or fall back to default path."""
    env = os.environ.get("US_COVIDX_ROOT")
    root = Path(env) if env else DEFAULT_COVIDX_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"COVIDx-US root not found at {root}. "
            "Set US_COVIDX_ROOT to override or verify the dataset path."
        )
    return root


def ensure_out_dir(base: Path = DEFAULT_OUT_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    (base / "raw_frames").mkdir(exist_ok=True)
    (base / "ssl_frames").mkdir(exist_ok=True)
    return base


# ── Manifest building ─────────────────────────────────────────────────────────

def build_covidx_manifest(root: Path, out_dir: Path) -> Path:
    """Run COVIDxUSAdapter over root and write a manifest JSONL."""
    manifest_path = out_dir / "covidx_explore_manifest.jsonl"
    adapter = COVIDxUSAdapter(root)
    entries: List[USManifestEntry] = list(adapter.iter_entries())

    if not entries:
        raise RuntimeError(f"No COVIDx-US entries found under {root}")

    with ManifestWriter(manifest_path) as w:
        for e in entries:
            w.write(e)

    stats = manifest_stats(entries)
    print(f"[COVIDx-US] Wrote {len(entries)} entries to {manifest_path}")
    print("[COVIDx-US] Manifest stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Basic label distribution
    label_counts = {0: 0, 1: 0, 2: 0}
    for e in entries:
        for inst in e.instances:
            if inst.classification_label is not None:
                label_counts[int(inst.classification_label)] = (
                    label_counts.get(int(inst.classification_label), 0) + 1
                )
    print("[COVIDx-US] Label distribution (0/1/2):", label_counts)

    return manifest_path


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _to_pil_image(arr: np.ndarray) -> Image.Image:
    """Convert HxW or HxWxC uint8 array to a single-channel or RGB PIL image."""
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        return Image.fromarray(arr if arr.shape[2] == 3 else arr[:, :, 0], mode="L")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(arr[:, :, :3])
    raise ValueError(f"Unsupported array shape for image conversion: {arr.shape}")


def save_raw_frames(entries: Iterable[USManifestEntry], out_dir: Path) -> None:
    """Save a few raw frames per clip for inspection."""
    for idx, e in enumerate(entries):
        if idx >= N_VIS_CLIPS:
            break

        # Each entry should have a single video path
        video_path = e.image_paths[0]
        frames = load_video_frames(video_path)
        if not frames:
            print(f"[COVIDx-US] No frames loaded for {video_path}")
            continue

        indices = [0, len(frames) // 2, len(frames) - 1]
        indices = [i for i in indices if 0 <= i < len(frames)]

        for fi, frame_idx in enumerate(indices[:N_FRAMES_PER_CLIP]):
            frame = frames[frame_idx]
            frame_img = _to_pil_image(frame.astype(np.uint8))
            out = out_dir / "raw_frames" / f"{idx:04d}_frame{fi}.png"
            frame_img.save(out)

        print(
            f"[COVIDx-US] Saved {len(indices[:N_FRAMES_PER_CLIP])} raw frames "
            f"for sample {e.sample_id} → idx={idx}"
        )


def save_ssl_frames(entries: List[USManifestEntry], out_dir: Path) -> None:
    """Run VideoSSLTransform on a subset of entries and save a few frames."""
    cfg = VideoSSLTransformConfig(
        clip_frames=32,
        crop_size=224,
    )

    # Use store paths directly; disable scratch remap for this exploration.
    configure_storage(use_scratch=False)

    ds = VideoSSLDataset(entries, cfg=cfg, patch_size=16)

    for idx in range(min(N_VIS_CLIPS, len(ds))):
        item = ds[idx]
        full_clip = item["full_clip"]  # Tensor(T,3,H,W)
        sample_id = item["sample_id"]

        # Save first, middle, last frame from transformed clip
        T = full_clip.shape[0]
        indices = [0, T // 2, T - 1]
        indices = [i for i in indices if 0 <= i < T]

        for fi, frame_idx in enumerate(indices[:N_FRAMES_PER_CLIP]):
            frame = full_clip[frame_idx]
            f_np = (frame.clamp(0, 1).numpy() * 255).astype(np.uint8)
            f_np = np.transpose(f_np, (1, 2, 0))
            f_img = _to_pil_image(f_np)
            out = out_dir / "ssl_frames" / f"{idx:04d}_ssl_frame{fi}.png"
            f_img.save(out)

        print(
            f"[COVIDx-US] Saved SSL frames for sample {sample_id} "
            f"→ dataset index {idx}"
        )


def downstream_classification_smoke(entries: List[USManifestEntry]) -> None:
    """
    Optional downstream classification smoke test.

    Builds a video downstream dataset (if/when available) or prints labels
    directly from entries for a quick sanity check.
    """
    # For now, just verify labels / splits from manifest entries.
    if not entries:
        print("[COVIDx-US] No entries available for downstream smoke test.")
        return

    splits = {}
    labels = {0: 0, 1: 0, 2: 0}
    for e in entries:
        splits[e.split] = splits.get(e.split, 0) + 1
        for inst in e.instances:
            if inst.classification_label is not None:
                labels[int(inst.classification_label)] = (
                    labels.get(int(inst.classification_label), 0) + 1
                )

    print("[COVIDx-US] Split counts:", splits)
    print("[COVIDx-US] Label counts:", labels)

    # Placeholder for future DownstreamDataset-based smoke if needed:
    try:
        from data.pipeline.downstream_dataset import DownstreamDataset  # type: ignore[attr-defined]

        tc = TaskConfig.multiclass_classification("lung")
        ds = DownstreamDataset(entries, task_config=tc, image_size=224)
        print(f"[COVIDx-US] DownstreamDataset size: {len(ds)}")
        for i in range(min(4, len(ds))):
            item = ds[i]
            x = item["image"]
            y = item.get("cls_label", None)
            print(
                f"[COVIDx-US] Downstream sample {i}: image.shape={tuple(x.shape)}, "
                f"label={y}"
            )
    except Exception as exc:  # noqa: BLE001
        print(
            "[COVIDx-US] DownstreamDataset not available or failed to construct; "
            f"continuing with manifest-only checks. Error: {exc}"
        )


def main() -> None:
    root = get_covidx_root()
    out_dir = ensure_out_dir()

    print(f"[COVIDx-US] Using COVIDx-US root: {root}")
    print(f"[COVIDx-US] Outputs will be written to: {out_dir.resolve()}")

    manifest_path = build_covidx_manifest(root, out_dir)

    # Reload manifest entries (round-trip coverage and future filtering)
    entries = load_manifest(manifest_path)
    print(f"[COVIDx-US] Loaded {len(entries)} entries from manifest.")

    # Visualise raw frames for a subset
    save_raw_frames(entries, out_dir)

    # Visualise SSL frames for a subset
    save_ssl_frames(entries, out_dir)

    # Optional downstream smoke test
    downstream_classification_smoke(entries)

    print("[COVIDx-US] Exploration complete.")


if __name__ == "__main__":
    main()

