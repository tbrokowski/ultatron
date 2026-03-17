"""
busi_explore.py  ·  Real-data exploration for BUSI breast ultrasound
====================================================================

Usage (from project root):

    python -m tests.dataset_adapters.busi_explore

This script:
  1. Uses the BUSIAdapter on the real BUSI dataset on CSCS.
  2. Writes a small manifest JSONL with all BUSI entries.
  3. Runs ImageSSL / downstream classification datasets on a subset.
  4. Saves PNG snapshots of raw images, masks, and transformed crops.
  5. Saves side-by-side debug PNGs comparing freq / spatial / both masking.
  6. Saves a summary panel per sample showing the original image, global
     crops, local crops, and all three masking strategies together.

It is an exploratory tool and is NOT part of the automated pytest suite.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from data.adapters.busi import BUSIAdapter
from data.schema.manifest import (
    ManifestWriter,
    USManifestEntry,
    load_manifest,
    manifest_stats,
)
from data.pipeline.dataset import ImageSSLDataset
from data.pipeline.downstream_dataset import DownstreamDataset
from data.pipeline.transforms import (
    ImageSSLTransform,
    ImageSSLTransformConfig,
    MASK_STRATEGY_FREQ,
    MASK_STRATEGY_SPATIAL,
    MASK_STRATEGY_BOTH,
)
from data.infra.storage import configure_storage


# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_BUSI_ROOT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/breast/BUSI"
)
DEFAULT_OUT_DIR = Path("dataset_exploration_outputs/busi")

N_VIS_SAMPLES = 16
N_DEBUG_SAMPLES = 4   # how many samples get the full side-by-side panel
PANEL_THUMB_SIZE = 224  # each thumbnail in the summary panel (pixels)
PANEL_PAD = 8           # padding between thumbnails (pixels)
LABEL_BAR_H = 22        # height of the text label bar above each thumbnail


def get_busi_root() -> Path:
    env = os.environ.get("US_BUSI_ROOT")
    root = Path(env) if env else DEFAULT_BUSI_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"BUSI root not found at {root}. "
            "Set US_BUSI_ROOT to override or verify the dataset path."
        )
    return root


def ensure_out_dir(base: Path = DEFAULT_OUT_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    for sub in ("raw", "masks", "crops_global", "crops_local",
                "crops_debug", "summary_panels"):
        (base / sub).mkdir(exist_ok=True)
    return base


# ── Manifest building ─────────────────────────────────────────────────────────

def build_busi_manifest(root: Path, out_dir: Path) -> Path:
    manifest_path = out_dir / "busi_explore_manifest.jsonl"
    adapter = BUSIAdapter(root)
    entries: List[USManifestEntry] = list(adapter.iter_entries())

    if not entries:
        raise RuntimeError(f"No BUSI entries found under {root}")

    with ManifestWriter(manifest_path) as w:
        for e in entries:
            w.write(e)

    stats = manifest_stats(entries)
    print(f"[BUSI] Wrote {len(entries)} entries to {manifest_path}")
    print("[BUSI] Manifest stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Per-class label distribution
    label_counts: Dict[str, int] = {}
    for e in entries:
        for inst in e.instances:
            label_counts[inst.label_raw] = label_counts.get(inst.label_raw, 0) + 1
    print("[BUSI] Label distribution:", label_counts)

    return manifest_path


# ── Image helpers ─────────────────────────────────────────────────────────────

def _to_pil_image(arr: np.ndarray) -> Image.Image:
    """Convert HxW or HxWxC uint8 array to PIL (RGB or L)."""
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return Image.fromarray(arr[:, :, 0].astype(np.uint8)).convert("RGB")
        if arr.shape[2] >= 3:
            return Image.fromarray(arr[:, :, :3].astype(np.uint8)).convert("RGB")
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def _tensor_to_pil(t) -> Image.Image:
    """Convert a (C, H, W) float32 tensor in [0,1] to a PIL RGB image."""
    np_img = (t.clamp(0, 1).numpy() * 255).astype(np.uint8)
    np_img = np.transpose(np_img, (1, 2, 0))
    return _to_pil_image(np_img)


def _resize_thumb(img: Image.Image, size: int) -> Image.Image:
    """Resize to square thumbnail, keeping aspect ratio with black padding."""
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _add_label(img: Image.Image, text: str, bar_h: int = LABEL_BAR_H) -> Image.Image:
    """Add a black label bar with white text above the image."""
    bar = Image.new("RGB", (img.width, bar_h), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 3), text, fill=(220, 220, 220), font=font)
    combined = Image.new("RGB", (img.width, img.height + bar_h))
    combined.paste(bar, (0, 0))
    combined.paste(img, (0, bar_h))
    return combined


# ── Per-sample raw + mask savers ──────────────────────────────────────────────

def save_raw_and_mask_images(entries: Iterable[USManifestEntry], out_dir: Path) -> None:
    from data.pipeline.dataset import load_image, load_mask

    for idx, e in enumerate(entries):
        if idx >= N_VIS_SAMPLES:
            break

        img_arr = load_image(e.image_paths[0])
        img = _to_pil_image(img_arr)
        img.save(out_dir / "raw" / f"busi_{idx:04d}.png")

        mask_saved = False
        for inst in e.instances:
            if inst.mask_path:
                try:
                    mask_arr = load_mask(inst.mask_path)
                    _to_pil_image(mask_arr.astype(np.uint8) * 255).save(
                        out_dir / "masks" / f"busi_{idx:04d}_mask.png"
                    )
                    mask_saved = True
                except Exception as exc:  # noqa: BLE001
                    print(f"[BUSI] Mask load error for {e.sample_id}: {exc}")
                break

        print(f"[BUSI] raw+mask idx={idx}  mask={mask_saved}  id={e.sample_id}")


# ── SSL crops + masking comparisons ──────────────────────────────────────────

def save_ssl_crops(entries: List[USManifestEntry], out_dir: Path) -> None:
    """
    For each of N_VIS_SAMPLES:
      - Save standard global crops (crops_global/) and local crops (crops_local/).

    For the first N_DEBUG_SAMPLES additionally:
      - Run ImageSSLTransform three times with mask_strategy="freq", "spatial", "both".
      - For each strategy save the masked student view (global[0]) and clean teacher
        view (global[1]) to crops_debug/.
    """
    from data.pipeline.dataset import load_image

    cfg = ImageSSLTransformConfig(
        n_global_crops=2,
        n_local_crops=2,
        max_global_crop_px=224,
        min_crop_px=64,
        mask_strategy=MASK_STRATEGY_FREQ,
    )
    configure_storage(use_scratch=False)
    ds = ImageSSLDataset(entries, cfg=cfg)

    for idx in range(min(N_VIS_SAMPLES, len(ds))):
        item = ds[idx]
        sample_id = item["sample_id"]

        # Standard global crops
        for gi, g in enumerate(item["global_crops"]):
            _tensor_to_pil(g).save(
                out_dir / "crops_global" / f"{idx:04d}_{gi}_global.png"
            )

        # Standard local crops
        for li, l in enumerate(item["local_crops"]):
            _tensor_to_pil(l).save(
                out_dir / "crops_local" / f"{idx:04d}_{li}_local.png"
            )

        print(f"[BUSI] SSL crops saved → idx={idx}  id={sample_id}")

        # ── Debug masking comparisons (first N_DEBUG_SAMPLES only) ────────────
        if idx < N_DEBUG_SAMPLES:
            img_arr = load_image(entries[idx].image_paths[0])

            for tag, strategy in (
                ("freq",    MASK_STRATEGY_FREQ),
                ("spatial", MASK_STRATEGY_SPATIAL),
                ("both",    MASK_STRATEGY_BOTH),
            ):
                dcfg = ImageSSLTransformConfig(
                    n_global_crops=2,
                    n_local_crops=0,
                    max_global_crop_px=224,
                    min_crop_px=64,
                    mask_strategy=strategy,
                )
                views = ImageSSLTransform(dcfg)(img_arr)
                # global[0] = masked student view, global[1] = clean teacher view
                _tensor_to_pil(views["global"][0]).save(
                    out_dir / "crops_debug" / f"{idx:04d}_{tag}_masked.png"
                )
                _tensor_to_pil(views["global"][1]).save(
                    out_dir / "crops_debug" / f"{idx:04d}_{tag}_clean.png"
                )

            print(f"[BUSI] Debug masking crops saved → idx={idx}")


# ── Summary panel ─────────────────────────────────────────────────────────────

def save_summary_panels(entries: List[USManifestEntry], out_dir: Path) -> None:
    """
    For the first N_DEBUG_SAMPLES entries, build a single PNG that shows:

    Row 0 (original):  Original image | Mask overlay (if any)
    Row 1 (SSL crops): Global crop 0 (masked) | Global crop 1 (clean) |
                       Local crop 0 | Local crop 1
    Row 2 (freq):      freq masked | freq clean
    Row 3 (spatial):   spatial masked | spatial clean
    Row 4 (both):      both masked | both clean

    Each thumbnail is PANEL_THUMB_SIZE × PANEL_THUMB_SIZE with a text label.
    """
    from data.pipeline.dataset import load_image, load_mask

    T = PANEL_THUMB_SIZE
    P = PANEL_PAD
    LH = LABEL_BAR_H

    def _thumb(img: Image.Image, label: str) -> Image.Image:
        return _add_label(_resize_thumb(img, T), label)

    for idx in range(min(N_DEBUG_SAMPLES, len(entries))):
        e = entries[idx]
        img_arr = load_image(e.image_paths[0])
        raw_pil = _to_pil_image(img_arr)

        # ── Row 0: original + mask ─────────────────────────────────────────
        row0 = [_thumb(raw_pil.copy(), "original")]
        for inst in e.instances:
            if inst.mask_path:
                try:
                    mask_arr = load_mask(inst.mask_path)
                    mask_rgb = Image.fromarray(mask_arr.astype(np.uint8) * 255).convert("RGB")
                    row0.append(_thumb(mask_rgb, "mask"))
                except Exception:  # noqa: BLE001
                    pass
                break

        # ── Row 1: standard SSL crops (2 global + 2 local) ────────────────
        ssl_cfg = ImageSSLTransformConfig(
            n_global_crops=2, n_local_crops=2,
            max_global_crop_px=T, min_crop_px=32,
            mask_strategy=MASK_STRATEGY_FREQ,
        )
        ssl_views = ImageSSLTransform(ssl_cfg)(img_arr)
        row1 = [
            _thumb(_tensor_to_pil(ssl_views["global"][0]), "global_0 (masked)"),
            _thumb(_tensor_to_pil(ssl_views["global"][1]), "global_1 (clean)"),
            _thumb(_tensor_to_pil(ssl_views["local"][0]),  "local_0"),
            _thumb(_tensor_to_pil(ssl_views["local"][1]),  "local_1"),
        ]

        # ── Rows 2-4: masking strategy comparison ─────────────────────────
        strategy_rows: List[Tuple[str, List[Image.Image]]] = []
        for tag, strategy in (
            ("freq",    MASK_STRATEGY_FREQ),
            ("spatial", MASK_STRATEGY_SPATIAL),
            ("both",    MASK_STRATEGY_BOTH),
        ):
            dcfg = ImageSSLTransformConfig(
                n_global_crops=2, n_local_crops=0,
                max_global_crop_px=T, min_crop_px=32,
                mask_strategy=strategy,
            )
            v = ImageSSLTransform(dcfg)(img_arr)
            strategy_rows.append((tag, [
                _thumb(_tensor_to_pil(v["global"][0]), f"{tag} masked"),
                _thumb(_tensor_to_pil(v["global"][1]), f"{tag} clean"),
            ]))

        # ── Assemble all rows into one canvas ─────────────────────────────
        all_rows: List[List[Image.Image]] = [row0, row1] + [r for _, r in strategy_rows]
        cell_h = T + LH
        n_cols = max(len(r) for r in all_rows)

        canvas_w = n_cols * (T + P) + P
        canvas_h = len(all_rows) * (cell_h + P) + P
        canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))

        # Row labels on left edge
        row_labels = ["original", "ssl_crops", "freq", "spatial", "both"]
        try:
            font_row = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13
            )
        except Exception:
            font_row = ImageFont.load_default()

        for ri, row in enumerate(all_rows):
            y = P + ri * (cell_h + P)
            for ci, thumb in enumerate(row):
                x = P + ci * (T + P)
                canvas.paste(thumb, (x, y))

        # Metadata label at bottom
        cls_label = (
            e.instances[0].label_raw if e.instances else "unknown"
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (P, canvas_h - 18),
            f"sample {idx}  id={e.sample_id[:12]}  class={cls_label}  "
            f"has_mask={e.has_mask}  tier={e.curriculum_tier}",
            fill=(180, 180, 180),
            font=font_row,
        )

        out = out_dir / "summary_panels" / f"busi_{idx:04d}_panel.png"
        canvas.save(out)
        print(f"[BUSI] Summary panel saved → {out}")


# ── Downstream smoke ──────────────────────────────────────────────────────────

def downstream_classification_smoke(entries: List[USManifestEntry]) -> None:
    if not entries:
        print("[BUSI] No entries available for downstream smoke test.")
        return

    cfg = ImageSSLTransformConfig(
        max_global_crop_px=224,
        min_crop_px=64,
        n_global_crops=1,
        n_local_crops=0,
    )
    cfg.global_crop_size = cfg.max_global_crop_px  # type: ignore[attr-defined]
    ds = DownstreamDataset(entries, cfg=cfg, training_mode="supervised")

    print(f"[BUSI] DownstreamDataset size: {len(ds)}")
    for i in range(min(4, len(ds))):
        item = ds[i]
        img = item["image"]
        label = item.get("cls_label", None)
        print(
            f"[BUSI] Downstream sample {i}: image.shape={tuple(img.shape)}, "
            f"label={label}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    root = get_busi_root()
    out_dir = ensure_out_dir()

    print(f"[BUSI] Using BUSI root: {root}")
    print(f"[BUSI] Outputs will be written to: {out_dir.resolve()}")

    manifest_path = build_busi_manifest(root, out_dir)

    entries = load_manifest(manifest_path)
    print(f"[BUSI] Loaded {len(entries)} entries from manifest.")

    # Individual per-category saves
    save_raw_and_mask_images(entries, out_dir)
    save_ssl_crops(entries, out_dir)

    # Organised summary panels (original + crops + all masking strategies)
    save_summary_panels(entries, out_dir)

    # Downstream smoke test
    downstream_classification_smoke(entries)

    print("[BUSI] Exploration complete.")
    print(f"[BUSI] Output layout:")
    print(f"  {out_dir}/raw/                  → raw input images")
    print(f"  {out_dir}/masks/                → binary segmentation masks")
    print(f"  {out_dir}/crops_global/         → standard SSL global crops")
    print(f"  {out_dir}/crops_local/          → standard SSL local crops")
    print(f"  {out_dir}/crops_debug/          → freq / spatial / both side-by-side")
    print(f"  {out_dir}/summary_panels/       → full per-sample overview panels")


if __name__ == "__main__":
    main()
