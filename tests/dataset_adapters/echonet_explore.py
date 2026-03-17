"""
echonet_explore.py  ·  Real-data exploration for EchoNet-Dynamic
===============================================================

Usage (from project root):

    python -m tests.dataset_adapters.echonet_explore

This script:
  1. Uses the EchoNetDynamicAdapter on the real EchoNet-Dynamic dataset on CSCS.
  2. Writes a small manifest JSONL with all EchoNet-Dynamic entries.
  3. Runs VideoSSL / downstream regression datasets on a subset.
  4. Saves PNG snapshots of raw frames and SSL-transformed clips.
  5. Saves side-by-side debug PNGs comparing freq / spatial / both masking.
  6. Saves a summary panel per sample showing raw frames and all SSL variants.

It is an exploratory tool and is NOT part of the automated pytest suite.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image

from data.adapters.echonet import EchoNetDynamicAdapter
from data.schema.manifest import (
    ManifestWriter,
    USManifestEntry,
    load_manifest,
    manifest_stats,
)
from data.pipeline.dataset import VideoSSLDataset, load_video_frames
from data.pipeline.downstream_dataset import DownstreamDataset
from data.pipeline.transforms import (
    VideoSSLTransform,
    VideoSSLTransformConfig,
    MASK_STRATEGY_FREQ,
    MASK_STRATEGY_SPATIAL,
    MASK_STRATEGY_BOTH,
)


# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_ECHONET_ROOT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/EchoNet-Dynamic"
)
DEFAULT_OUT_DIR = Path("dataset_exploration_outputs/echonet")

N_VIS_CLIPS = 8
N_DEBUG_CLIPS = 4
N_FRAMES_PER_CLIP = 3  # first, middle, last


def get_echonet_root() -> Path:
    env = os.environ.get("US_ECHONET_ROOT")
    root = Path(env) if env else DEFAULT_ECHONET_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"EchoNet-Dynamic root not found at {root}. "
            "Set US_ECHONET_ROOT to override or verify the dataset path."
        )
    return root


def ensure_out_dir(base: Path = DEFAULT_OUT_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    for sub in (
        "raw_frames",
        "ssl_frames/full",
        "ssl_frames/visible",
        "mask_debug",
        "summary_panels",
    ):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


# ── Helpers ────────────────────────────────────────────────────────────────────

def _frame_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert HxW or HxWxC numpy array in [0,255] to PIL RGB."""
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0].astype(np.uint8)).convert("RGB")
        return Image.fromarray(arr[:, :, :3].astype(np.uint8)).convert("RGB")
    raise ValueError(f"Unsupported frame shape: {arr.shape}")


def _tensor_clip_to_frames(t):
    """
    Convert a (T, C, H, W) float tensor in [0,1] to a list of PIL frames.
    """
    import torch

    if isinstance(t, torch.Tensor):
        arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    else:
        arr = (np.clip(t, 0, 1) * 255).astype(np.uint8)
    # (T, C, H, W) -> (T, H, W, C)
    arr = np.transpose(arr, (0, 2, 3, 1))
    return [_frame_to_pil(f) for f in arr]


# ── Manifest building ─────────────────────────────────────────────────────────

def build_echonet_manifest(root: Path, out_dir: Path) -> Path:
    manifest_path = out_dir / "echonet_explore_manifest.jsonl"
    adapter = EchoNetDynamicAdapter(root)
    entries: List[USManifestEntry] = list(adapter.iter_entries())

    if not entries:
        raise RuntimeError(f"No EchoNet-Dynamic entries found under {root}")

    with ManifestWriter(manifest_path) as w:
        for e in entries:
            w.write(e)

    stats = manifest_stats(entries)
    print(f"[ECHONET] Wrote {len(entries)} entries to {manifest_path}")
    print("[ECHONET] Manifest stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Split distribution and EF stats
    from statistics import mean

    splits = {}
    efs: List[float] = []
    for e in entries:
        splits[e.split] = splits.get(e.split, 0) + 1
        ef = float(e.source_meta.get("ef", 0.0))
        efs.append(ef)
    print("[ECHONET] Split distribution:", splits)
    if efs:
        print(
            "[ECHONET] EF stats: "
            f"min={min(efs):.2f}, mean={mean(efs):.2f}, max={max(efs):.2f}"
        )

    return manifest_path


# ── Raw frame exports ─────────────────────────────────────────────────────────

def save_raw_frames(entries: Iterable[USManifestEntry], out_dir: Path) -> None:
    for idx, e in enumerate(entries):
        if idx >= N_VIS_CLIPS:
            break
        clip = load_video_frames(e.image_paths[0])
        # clip is a list of (H, W, 3) uint8 frames
        if not clip:
            continue
        T = len(clip)
        frame_indices = [0, T // 2, max(T - 1, 0)]
        for j, fi in enumerate(frame_indices):
            frame = clip[fi]
            _frame_to_pil(frame).save(
                out_dir / "raw_frames" / f"echonet_{idx:04d}_{j}.png"
            )
        print(f"[ECHONET] Saved raw frames for idx={idx} id={e.sample_id}")


# ── Video SSL views ───────────────────────────────────────────────────────────

def save_video_ssl_views(entries: List[USManifestEntry], out_dir: Path) -> None:
    cfg = VideoSSLTransformConfig(
        n_frames=16,
        max_crop_px=224,
        min_crop_px=64,
        mask_strategy=MASK_STRATEGY_FREQ,
    )
    ds = VideoSSLDataset(entries, cfg=cfg)

    for idx in range(min(N_VIS_CLIPS, len(ds))):
        item = ds[idx]
        full_clip = item["full_clip"]      # (T, C, H, W)
        visible_clip = item["visible_clip"]

        full_frames = _tensor_clip_to_frames(full_clip)
        visible_frames = _tensor_clip_to_frames(visible_clip)

        # pick a few frames (first/middle/last)
        T_full = len(full_frames)
        sel = [0, T_full // 2, max(T_full - 1, 0)]
        for j, fi in enumerate(sel):
            full_frames[fi].save(
                out_dir / "ssl_frames/full" / f"echonet_{idx:04d}_{j}.png"
            )
            visible_frames[fi].save(
                out_dir / "ssl_frames/visible" / f"echonet_{idx:04d}_{j}.png"
            )

        print(f"[ECHONET] SSL frames saved → idx={idx} id={item['sample_id']}")


# ── Masking debug (freq / spatial / both) ─────────────────────────────────────

def save_masking_debug(entries: List[USManifestEntry], out_dir: Path) -> None:
    for idx, e in enumerate(entries[:N_DEBUG_CLIPS]):
        clip = load_video_frames(e.image_paths[0])

        for tag, strategy in (
            ("freq", MASK_STRATEGY_FREQ),
            ("spatial", MASK_STRATEGY_SPATIAL),
            ("both", MASK_STRATEGY_BOTH),
        ):
            cfg = VideoSSLTransformConfig(
                n_frames=16,
                max_crop_px=224,
                min_crop_px=64,
                mask_strategy=strategy,
            )
            views = VideoSSLTransform(cfg)(clip)
            full_clip = views["full"]
            visible_clip = views["visible"]

            full_frames = _tensor_clip_to_frames(full_clip)
            visible_frames = _tensor_clip_to_frames(visible_clip)

            if not full_frames or not visible_frames:
                continue

            # save first frame of each
            full_frames[0].save(
                out_dir
                / "mask_debug"
                / f"echonet_{idx:04d}_{tag}_full.png"
            )
            visible_frames[0].save(
                out_dir
                / "mask_debug"
                / f"echonet_{idx:04d}_{tag}_visible.png"
            )

        print(f"[ECHONET] Masking debug saved → idx={idx} id={e.sample_id}")


# ── Summary panels ────────────────────────────────────────────────────────────

def save_summary_panels(entries: List[USManifestEntry], out_dir: Path) -> None:
    """
    For the first N_DEBUG_CLIPS entries, build a panel showing:

      Row 0: raw frames (first, mid, last)
      Row 1: SSL freq full vs visible
      Row 2: SSL spatial full vs visible
      Row 3: SSL both full vs visible
    """
    from PIL import ImageDraw, ImageFont

    thumb_w, thumb_h = 112, 112
    pad = 8

    def _resize_thumb(img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), (0, 0, 0))
        x = (thumb_w - img.width) // 2
        y = (thumb_h - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas

    def _label(img: Image.Image, text: str) -> Image.Image:
        bar_h = 18
        bar = Image.new("RGB", (img.width, bar_h), (30, 30, 30))
        draw = ImageDraw.Draw(bar)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
            )
        except Exception:
            font = ImageFont.load_default()
        draw.text((4, 2), text, fill=(220, 220, 220), font=font)
        canvas = Image.new("RGB", (img.width, img.height + bar_h))
        canvas.paste(bar, (0, 0))
        canvas.paste(img, (0, bar_h))
        return canvas

    for idx, e in enumerate(entries[:N_DEBUG_CLIPS]):
        clip = load_video_frames(e.image_paths[0])
        if not clip:
            continue
        T = len(clip)
        raw_frames = [
            _frame_to_pil(clip[0]),
            _frame_to_pil(clip[T // 2]),
            _frame_to_pil(clip[max(T - 1, 0)]),
        ]
        raw_row = [_label(_resize_thumb(f), "raw") for f in raw_frames]

        strategy_rows = []
        for tag, strategy in (
            ("freq", MASK_STRATEGY_FREQ),
            ("spatial", MASK_STRATEGY_SPATIAL),
            ("both", MASK_STRATEGY_BOTH),
        ):
            cfg = VideoSSLTransformConfig(
                n_frames=16,
                max_crop_px=224,
                min_crop_px=64,
                mask_strategy=strategy,
            )
            views = VideoSSLTransform(cfg)(clip)
            full_clip = views["full"]
            visible_clip = views["visible"]
            full_frames = _tensor_clip_to_frames(full_clip)
            visible_frames = _tensor_clip_to_frames(visible_clip)
            if not full_frames or not visible_frames:
                continue
            row = [
                _label(_resize_thumb(full_frames[0]), f"{tag} full"),
                _label(_resize_thumb(visible_frames[0]), f"{tag} vis"),
            ]
            strategy_rows.append(row)

        rows = [raw_row] + strategy_rows

        cell_w = thumb_w
        cell_h = thumb_h + 18
        n_cols = max(len(r) for r in rows)
        n_rows = len(rows)
        canvas_w = n_cols * (cell_w + pad) + pad
        canvas_h = n_rows * (cell_h + pad) + pad
        canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))

        for ri, row in enumerate(rows):
            y = pad + ri * (cell_h + pad)
            for ci, img in enumerate(row):
                x = pad + ci * (cell_w + pad)
                canvas.paste(img, (x, y))

        out = out_dir / "summary_panels" / f"echonet_{idx:04d}_panel.png"
        canvas.save(out)
        print(f"[ECHONET] Summary panel saved → {out}")


# ── Downstream EF regression smoke ────────────────────────────────────────────

def downstream_ef_regression_smoke(entries: List[USManifestEntry]) -> None:
    if not entries:
        print("[ECHONET] No entries available for downstream smoke test.")
        return

    cfg = VideoSSLTransformConfig(
        n_frames=16,
        max_crop_px=224,
        min_crop_px=64,
        mask_strategy=MASK_STRATEGY_FREQ,
    )
    # DownstreamDataset may expect cfg.global_crop_size in some code paths; add shim.
    cfg.global_crop_size = cfg.max_crop_px  # type: ignore[attr-defined]

    ds = DownstreamDataset(entries, cfg=cfg, training_mode="supervised")
    print(f"[ECHONET] DownstreamDataset size: {len(ds)}")
    for i in range(min(4, len(ds))):
        item = ds[i]
        clip = item["image"]
        cls_label = item.get("cls_label")
        label_targets = item.get("label_targets")
        print(
            f"[ECHONET] Downstream sample {i}: clip.shape={tuple(clip.shape)}, "
            f"cls_label={cls_label}, label_targets={label_targets}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    root = get_echonet_root()
    out_dir = ensure_out_dir()

    print(f"[ECHONET] Using EchoNet-Dynamic root: {root}")
    print(f"[ECHONET] Outputs will be written to: {out_dir.resolve()}")

    manifest_path = build_echonet_manifest(root, out_dir)
    entries = load_manifest(manifest_path)
    print(f"[ECHONET] Loaded {len(entries)} entries from manifest.")

    save_raw_frames(entries, out_dir)
    save_video_ssl_views(entries, out_dir)
    save_masking_debug(entries, out_dir)
    save_summary_panels(entries, out_dir)
    downstream_ef_regression_smoke(entries)

    print("[ECHONET] Exploration complete.")
    print("Output layout:")
    print(f"  {out_dir}/raw_frames/          → raw frames (first/mid/last)")
    print(f"  {out_dir}/ssl_frames/full/     → SSL full clips (few frames)")
    print(f"  {out_dir}/ssl_frames/visible/  → SSL visible clips (few frames)")
    print(f"  {out_dir}/mask_debug/          → freq/spatial/both comparison")
    print(f"  {out_dir}/summary_panels/      → per-sample overview panels")


if __name__ == "__main__":
    main()

