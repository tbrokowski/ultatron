#!/usr/bin/env python3
"""
scripts/data_prep_analysis.py  ·  Ultatron data pipeline verification report
=============================================================================

Generates a self-contained HTML report that exercises every layer of the
data pipeline: manifest loading, image/video I/O, all three masking
strategies, adaptive curriculum tiers, and canonical label spaces.

Sections
--------
1. Manifest Summary          — per-dataset and per-anatomy statistics
2. Label Spaces              — canonical class vocabularies per anatomy
3. Image Loading Test        — one sample thumbnail per dataset (all in manifest)
4. Video Loading Test        — filmstrip per dataset with video entries
5. Segmentation Mask Test    — image + mask overlay for annotated entries
6. Masking Strategy Comparison — freq / spatial / both side-by-side
7. Adaptive Curriculum Masking — tier-1/2/3 mask ratios (40% / 65% / 80%)
8. Video Tube Masking        — teacher vs student clip + tube-mask heatmap

Usage
-----
    python scripts/data_prep_analysis.py                        # defaults
    python scripts/data_prep_analysis.py \\
        --manifest dataset_exploration_outputs/run1/run1_train_v3.jsonl \\
        --out      dataset_exploration_outputs/data_prep_analysis.html \\
        --n-per-dataset 1 \\
        --n-video-datasets 8
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from torch import Tensor

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Optional tqdm ─────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, desc="", **kw):  # type: ignore[misc]
        print(f"[{desc}]" if desc else "", flush=True)
        return it

# ── Project imports ───────────────────────────────────────────────────────────
from data.schema.manifest import USManifestEntry, load_manifest
from data.pipeline.dataset import (
    load_image, load_video_frames, load_mask,
    media_path_exists, image_path_extension, resolve_media_path,
    _VOLUME_SLICE_EXTS,
)
from data.pipeline.transforms import (
    ImageSSLTransformConfig,
    VideoSSLTransform, VideoSSLTransformConfig,
    FreqMaskConfig, to_canonical_tensor,
    freq_mask_image, spatial_patch_mask,
    MASK_STRATEGY_FREQ, MASK_STRATEGY_SPATIAL, MASK_STRATEGY_BOTH,
)
from data.labels.label_interface import (
    ANATOMY_LABEL_SPACES, ANATOMY_DEFAULT_HEAD,
)

log = logging.getLogger(__name__)

# Populated in main() from StorageConfig; maps store → scratch roots.
_ROOT_REMAP: Dict[str, str] = {}


def _remap_path(path: str) -> str:
    for old, new in _ROOT_REMAP.items():
        if path.startswith(old):
            return path.replace(old, new, 1)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CURRICULUM_TIERS = {
    1: {"mask_ratio": 0.40, "label": "Tier 1 — mask 40%", "color": "#4CAF50"},
    2: {"mask_ratio": 0.65, "label": "Tier 2 — mask 65%", "color": "#FF9800"},
    3: {"mask_ratio": 0.80, "label": "Tier 3 — mask 80%", "color": "#F44336"},
}

MASKING_STRATEGIES = [
    (MASK_STRATEGY_FREQ,    "Freq (spectral)"),
    (MASK_STRATEGY_SPATIAL, "Spatial (black)"),
    (MASK_STRATEGY_BOTH,    "Both combined"),
]

# ─────────────────────────────────────────────────────────────────────────────
# HTML Report writer
# ─────────────────────────────────────────────────────────────────────────────

_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2d3148;
    --text: #e2e8f0; --muted: #8892a4; --accent: #6366f1;
    --green: #4ade80; --yellow: #fbbf24; --red: #f87171;
    --mono: "JetBrains Mono", "Fira Code", monospace;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
    font-size: 14px; line-height: 1.6;
  }}
  #toc {{
    position: fixed; top: 0; left: 0; width: 240px; height: 100vh;
    background: var(--surface); border-right: 1px solid var(--border);
    overflow-y: auto; padding: 16px; z-index: 100;
  }}
  #toc h3 {{ color: var(--accent); margin-bottom: 12px; font-size: 13px;
              letter-spacing: .05em; text-transform: uppercase; }}
  #toc a {{ display: block; color: var(--muted); text-decoration: none;
             padding: 4px 0; font-size: 12px; border-left: 2px solid transparent;
             padding-left: 8px; transition: .15s; }}
  #toc a:hover {{ color: var(--text); border-color: var(--accent); }}
  #main {{ margin-left: 240px; padding: 32px 40px; max-width: 1400px; }}
  h1 {{ color: var(--text); font-size: 26px; margin-bottom: 8px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 32px; font-size: 13px; }}
  h2 {{ color: var(--accent); font-size: 20px; margin: 40px 0 16px;
        padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
  h3 {{ color: var(--text); font-size: 15px; margin: 24px 0 10px; }}
  h4 {{ color: var(--muted); font-size: 13px; margin: 16px 0 6px;
        text-transform: uppercase; letter-spacing: .05em; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 12px; }}
  th {{ background: var(--surface); color: var(--muted); text-transform: uppercase;
        font-size: 11px; letter-spacing: .05em; padding: 8px 10px;
        border-bottom: 2px solid var(--border); text-align: left; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:hover td {{ background: rgba(99,102,241,.05); }}
  .badge {{
    display: inline-block; padding: 2px 7px; border-radius: 10px;
    font-size: 10px; font-weight: 600; letter-spacing: .03em;
  }}
  .badge-image  {{ background: #1e3a5f; color: #60a5fa; }}
  .badge-video  {{ background: #1e3d30; color: #4ade80; }}
  .badge-both   {{ background: #3d2b1e; color: #fb923c; }}
  .badge-t1     {{ background: #1e3d30; color: #4ade80; }}
  .badge-t2     {{ background: #3d3a1e; color: #fbbf24; }}
  .badge-t3     {{ background: #3d1e1e; color: #f87171; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
                gap: 12px; margin: 16px 0 24px; }}
  .stat-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
  }}
  .stat-card .val {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
  .stat-card .lbl {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  figure {{ margin: 16px 0 24px; }}
  figure img {{ max-width: 100%; border-radius: 6px;
                border: 1px solid var(--border); display: block; }}
  figcaption {{ color: var(--muted); font-size: 11px; margin-top: 6px;
                font-family: var(--mono); }}
  .warning {{
    background: #3d2008; color: #fb923c; border-left: 3px solid #fb923c;
    padding: 8px 12px; margin: 8px 0; border-radius: 0 4px 4px 0;
    font-size: 12px; font-family: var(--mono);
  }}
  .dataset-section {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px; margin: 16px 0;
  }}
  .label-space {{ margin: 8px 0; }}
  .label-space td:first-child {{ font-family: var(--mono); color: var(--accent);
                                  width: 3em; text-align: right; }}
  .label-space td:last-child  {{ font-family: var(--mono); }}
  code {{ font-family: var(--mono); font-size: 12px; color: #a5b4fc;
           background: rgba(99,102,241,.1); padding: 1px 5px; border-radius: 3px; }}
  .thumb-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin: 12px 0;
  }}
  .thumb-card {{
    background: #12151e;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}
  .thumb-card img {{
    width: 100%;
    aspect-ratio: 1;
    object-fit: contain;
    background: #000;
    display: block;
  }}
  .thumb-card .thumb-meta {{
    padding: 6px 8px;
    font-size: 10px;
    font-family: var(--mono);
    color: var(--muted);
    line-height: 1.5;
  }}
  .thumb-card .thumb-ds {{ font-weight: 700; color: var(--text); font-size: 11px; }}
</style>
</head>
<body>
<nav id="toc">
  <h3>Contents</h3>
  {toc}
</nav>
<div id="main">
<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>
"""

_HTML_TAIL = """
</div>
</body>
</html>
"""


class HTMLReport:
    """Incremental HTML report builder with base64 figure embedding."""

    def __init__(self, path: str, title: str, subtitle: str = ""):
        self.path = path
        self._sections: List[Tuple[str, str]] = []
        self._buffer: List[str] = []
        self._title = title
        self._subtitle = subtitle
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def section(self, anchor: str, title: str):
        self._sections.append((anchor, title))
        self._buffer.append(f'<h2 id="{anchor}">{title}</h2>\n')

    def h3(self, text: str):
        self._buffer.append(f"<h3>{text}</h3>\n")

    def h4(self, text: str):
        self._buffer.append(f"<h4>{text}</h4>\n")

    def html(self, content: str):
        self._buffer.append(content + "\n")

    def warning(self, msg: str):
        self._buffer.append(
            f'<div class="warning">&#9888; {_esc(msg)}</div>\n'
        )

    def stat_grid(self, stats: Dict[str, str]):
        cards = "".join(
            f'<div class="stat-card"><div class="val">{v}</div>'
            f'<div class="lbl">{k}</div></div>'
            for k, v in stats.items()
        )
        self._buffer.append(f'<div class="stat-grid">{cards}</div>\n')

    def table(self, headers: List[str], rows: List[List[str]]):
        ths = "".join(f"<th>{h}</th>" for h in headers)
        trs = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
            for row in rows
        )
        self._buffer.append(f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>\n")

    def figure(self, fig: plt.Figure, caption: str = "", dpi: int = 90):
        b64 = _fig_to_b64(fig, dpi=dpi)
        cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
        self._buffer.append(
            f'<figure><img src="{b64}" loading="lazy">{cap}</figure>\n'
        )

    def thumb_grid_open(self):
        self._buffer.append('<div class="thumb-grid">\n')

    def thumb_grid_close(self):
        self._buffer.append("</div>\n")

    def thumb_card(self, fig: plt.Figure, dataset_id: str, meta_lines: List[str], dpi: int = 72):
        b64 = _fig_to_b64(fig, dpi=dpi)
        meta_html = "".join(f"{_esc(l)}<br>" for l in meta_lines)
        self._buffer.append(
            f'<div class="thumb-card">'
            f'<img src="{b64}" loading="lazy">'
            f'<div class="thumb-meta">'
            f'<div class="thumb-ds">{_esc(dataset_id)}</div>'
            f'{meta_html}'
            f"</div></div>\n"
        )

    def save(self):
        toc_links = "".join(
            f'<a href="#{a}">{t}</a>' for a, t in self._sections
        )
        head = _HTML_HEAD.format(
            title=_esc(self._title),
            subtitle=_esc(self._subtitle),
            toc=toc_links,
        )
        with open(self.path, "w") as fh:
            fh.write(head)
            fh.writelines(self._buffer)
            fh.write(_HTML_TAIL)
        log.info("Report saved → %s", self.path)

    # Context-manager support so we can save on exit
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.save()


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fig_to_b64(fig: plt.Figure, dpi: int = 90) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _tensor_to_np(t: Tensor) -> np.ndarray:
    """(3,H,W) float32 [0,1] → (H,W,3) uint8."""
    return (t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def _pad_to_patch(t: Tensor, ps: int = 16) -> Tensor:
    """Crop to largest patch-aligned H×W."""
    _, H, W = t.shape
    return t[:, : (H // ps) * ps, : (W // ps) * ps]


def _load_entry_image(entry: USManifestEntry) -> Optional[np.ndarray]:
    """Load a representative frame from an entry, return None on failure."""
    try:
        path = resolve_media_path(entry.image_paths[0], root_remap=_ROOT_REMAP)
        if path is None:
            log.warning("load_entry_image: file not found for %s: %s",
                        entry.sample_id, entry.image_paths[0])
            return None
        ext = image_path_extension(path)
        meta = entry.source_meta or {}
        if "frame_idx" in meta and meta["frame_idx"] is not None:
            frame_idx = int(meta["frame_idx"])
        elif ext in _VOLUME_SLICE_EXTS or entry.modality_type == "volume":
            frame_idx = -1
        else:
            frame_idx = 0
        if ext in _VOLUME_SLICE_EXTS:
            return load_image(path, frame_idx=frame_idx)
        if ext == ".dcm":
            return load_image(path, frame_idx=-1)
        if (
            entry.modality_type in ("video", "pseudo_video", "volume")
            and ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif", ".h5", ".hdf5")
        ):
            frames = load_video_frames(path, max_frames=4)
            return frames[len(frames) // 2] if frames else None
        return load_image(path, frame_idx=frame_idx)
    except Exception as exc:
        log.warning("load_entry_image failed for %s (%s): %s",
                    entry.sample_id, entry.image_paths[0], exc)
        return None


def _load_entry_video(entry: USManifestEntry, max_frames: int = 8) -> Optional[List]:
    """Load video frames from an entry, return None on failure."""
    try:
        path = resolve_media_path(entry.image_paths[0], root_remap=_ROOT_REMAP)
        if path is None:
            log.warning("load_entry_video: file not found for %s: %s",
                        entry.sample_id, entry.image_paths[0])
            return None
        ext = image_path_extension(path)
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".gif", ".dcm", ".h5", ".hdf5"):
            frames = load_video_frames(path, max_frames=max_frames)
        elif ext in _VOLUME_SLICE_EXTS and len(entry.image_paths) == 1:
            frames = load_video_frames(path, max_frames=max_frames)
        else:
            # Multi-frame image sequence (e.g. CAMUS ED→ES pseudo-video)
            frames = []
            for p in entry.image_paths:
                resolved = resolve_media_path(p, root_remap=_ROOT_REMAP)
                if resolved is not None:
                    frames.append(load_image(resolved, frame_idx=0))
            frames = frames[:max_frames]
        return frames if frames else None
    except Exception as exc:
        log.warning("load_entry_video failed for %s (%s): %s",
                    entry.sample_id, entry.image_paths[0], exc)
        return None


def _overlay_mask(img_np: np.ndarray, mask: Tensor, ps: int = 16) -> np.ndarray:
    """Overlay patch mask (red tint) on float [0,1] RGB numpy array."""
    overlay = img_np.copy().astype(np.float32)
    ph, pw = mask.shape
    for pi in range(ph):
        for pj in range(pw):
            if mask[pi, pj]:
                r0, r1 = pi * ps, (pi + 1) * ps
                c0, c1 = pj * ps, (pj + 1) * ps
                overlay[r0:r1, c0:c1, 0] = np.clip(overlay[r0:r1, c0:c1, 0] * 0.4 + 0.6, 0, 1)
                overlay[r0:r1, c0:c1, 1] *= 0.3
                overlay[r0:r1, c0:c1, 2] *= 0.3
    return np.clip(overlay, 0, 1)


def _dark_fig(*args, **kwargs) -> plt.Figure:
    fig = plt.figure(*args, **kwargs)
    fig.patch.set_facecolor("#0f1117")
    return fig


def _dark_ax(ax: plt.Axes):
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#8892a4")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2d3148")
    return ax


def _imshow_dark(ax: plt.Axes, img_np: np.ndarray, title: str = ""):
    _dark_ax(ax)
    ax.imshow(img_np, interpolation="nearest", vmin=0, vmax=1 if img_np.dtype == np.float32 else 255)
    if title:
        ax.set_title(title, color="#e2e8f0", fontsize=9, pad=4)
    ax.axis("off")


def _to_rgb_np(img_np: np.ndarray) -> np.ndarray:
    """Ensure (H, W, 3) float display array."""
    if img_np.ndim == 2:
        return np.stack([img_np, img_np, img_np], axis=-1)
    return img_np[..., :3]


def _freq_change_map(orig_np: np.ndarray, deg_np: np.ndarray) -> np.ndarray:
    """Per-pixel |Δluma| scaled to ~[0, 1] for display (p99 normalization)."""
    orig = _to_rgb_np(orig_np).astype(np.float32)
    deg  = _to_rgb_np(deg_np).astype(np.float32)
    diff = np.abs(orig.mean(axis=-1) - deg.mean(axis=-1))
    if diff.max() <= 1e-8:
        return diff
    p99 = float(np.percentile(diff, 99))
    return np.clip(diff / max(p99, 1e-6), 0.0, 1.0)


def _imshow_freq_degradation_triple(
    ax: plt.Axes,
    orig_np: np.ndarray,
    deg_np: np.ndarray,
    title: str = "",
):
    """Original | degraded | amplified spectral residual (diff is often subtle in pixel space)."""
    orig = _to_rgb_np(orig_np)
    deg  = _to_rgb_np(deg_np)
    diff = _freq_change_map(orig_np, deg_np)
    diff_rgb = plt.cm.inferno(diff)[..., :3]
    triplet = np.concatenate([orig, deg, diff_rgb], axis=1)
    _dark_ax(ax)
    ax.imshow(triplet, interpolation="bilinear", vmin=0, vmax=1)
    w = orig.shape[1]
    ax.axvline(w - 0.5, color="#64748b", linewidth=0.8, alpha=0.7)
    ax.axvline(2 * w - 0.5, color="#64748b", linewidth=0.8, alpha=0.7)
    mean_d = float(np.abs(_to_rgb_np(orig_np).astype(np.float32).mean(-1)
                        - _to_rgb_np(deg_np).astype(np.float32).mean(-1)).mean())
    ax.set_title(
        title or f"Original | degraded | |Δ|×scale  (mean Δ={mean_d:.3f})",
        color="#e2e8f0", fontsize=9,
    )
    ax.axis("off")


def _imshow_degradation_pair(
    ax: plt.Axes,
    orig_np: np.ndarray,
    deg_np: np.ndarray,
    title: str = "Original | Degraded",
):
    """Side-by-side original and spectrally degraded views (no patch overlay)."""
    _imshow_freq_degradation_triple(ax, orig_np, deg_np, title=title)


def _imshow_spatial_mask_overlay(
    ax: plt.Axes,
    img_np: np.ndarray,
    mask: Tensor,
    patch_size: int,
    title: str,
):
    """Red patch overlay — meaningful for spatial / both strategies."""
    _dark_ax(ax)
    img = _to_rgb_np(img_np)
    ax.imshow(img, interpolation="nearest", vmin=0, vmax=1)
    ph, pw = mask.shape
    H, W = img.shape[:2]
    for pi in range(ph):
        for pj in range(pw):
            if mask[pi, pj]:
                rect = mpatches.Rectangle(
                    (pj * patch_size - 0.5, pi * patch_size - 0.5),
                    patch_size, patch_size,
                    linewidth=0, facecolor="#f87171", alpha=0.45,
                )
                ax.add_patch(rect)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(title, color="#e2e8f0", fontsize=9)
    ax.axis("off")


def _badge_html(text: str, kind: str) -> str:
    cls_map = {
        "image": "badge-image", "video": "badge-video", "both": "badge-both",
        "t1": "badge-t1", "t2": "badge-t2", "t3": "badge-t3",
    }
    cls = cls_map.get(kind, "badge-image")
    return f'<span class="badge {cls}">{_esc(text)}</span>'


# modality_type on each manifest entry (set by adapters via BaseAdapter._make_entry)
_IMAGE_MODALITIES = frozenset({"image"})
_VIDEO_MODALITIES = frozenset({"video", "pseudo_video"})


def _modality_counts(entries: List[USManifestEntry]) -> Dict[str, int]:
    """Count loadable samples by modality_type (not ssl_stream routing)."""
    return {
        "image": sum(1 for e in entries if e.modality_type in _IMAGE_MODALITIES),
        "video": sum(1 for e in entries if e.modality_type in _VIDEO_MODALITIES),
        "volume": sum(1 for e in entries if e.modality_type == "volume"),
    }


def _is_image_entry(entry: USManifestEntry) -> bool:
    return entry.modality_type in _IMAGE_MODALITIES


def _is_video_entry(entry: USManifestEntry) -> bool:
    return entry.modality_type in _VIDEO_MODALITIES


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def section_manifest_summary(report: HTMLReport, entries: List[USManifestEntry]):
    report.section("s1", "1. Manifest Summary")

    # High-level stats
    datasets  = {e.dataset_id for e in entries}
    families  = {e.anatomy_family for e in entries}
    streams   = defaultdict(int)
    tiers     = defaultdict(int)
    modalities = _modality_counts(entries)
    for e in entries:
        streams[e.ssl_stream] += 1
        tiers[e.curriculum_tier] += 1

    n_mask  = sum(1 for e in entries if e.has_mask)
    n_label = sum(1 for e in entries if e.task_type != "ssl_only")

    stat_grid = {
        "Total entries":    f"{len(entries):,}",
        "Datasets":         str(len(datasets)),
        "Anatomy families": str(len(families)),
        "Images":           f"{modalities['image']:,}",
        "Videos":           f"{modalities['video']:,}",
        "With annotation":  f"{n_mask + n_label:,}",
        "SSL image-only":   f"{streams['image']:,}",
        "SSL video-only":   f"{streams['video']:,}",
        "SSL both":         f"{streams['both']:,}",
        "Tier 1":           f"{tiers[1]:,}",
        "Tier 2":           f"{tiers[2]:,}",
        "Tier 3":           f"{tiers[3]:,}",
    }
    if modalities["volume"]:
        stat_grid["Volumes"] = f"{modalities['volume']:,}"
    report.stat_grid(stat_grid)

    report.html(
        "<p><b>Images / Videos</b> count manifest entries by "
        "<code>modality_type</code> (what each sample actually is on disk). "
        "<b>SSL *</b> counts show how entries are routed to image vs video "
        "training streams via <code>ssl_stream</code>; a single dataset can "
        "appear in both columns when entries are tagged "
        "<code>ssl_stream=&quot;both&quot;</code>.</p>"
    )

    # Per-dataset table
    report.h3("Per-dataset breakdown")
    try:
        from data.infra.storage import DATASET_STORE_MAP
    except ImportError:
        DATASET_STORE_MAP = {}

    ds_map: Dict[str, List[USManifestEntry]] = defaultdict(list)
    for e in entries:
        ds_map[e.dataset_id].append(e)

    has_volumes = any(_modality_counts(es)["volume"] for es in ds_map.values())
    rows = []
    for ds_id in sorted(ds_map):
        es = ds_map[ds_id]
        n  = len(es)
        fam = es[0].anatomy_family
        mc = _modality_counts(es)
        n_msk = sum(1 for e in es if e.has_mask)
        t1 = sum(1 for e in es if e.curriculum_tier == 1)
        t2 = sum(1 for e in es if e.curriculum_tier == 2)
        t3 = sum(1 for e in es if e.curriculum_tier == 3)
        task_types = ", ".join(sorted({e.task_type for e in es}))
        store_slug = DATASET_STORE_MAP.get(ds_id, (None, ds_id))[1]
        row = [
            f"<code>{_esc(ds_id)}</code>",
            f"<code>{_esc(store_slug)}</code>",
            _esc(fam),
            f"{n:,}",
            f"{mc['image']:,}",
            f"{mc['video']:,}",
            f"{n_msk:,}",
            f"{t1:,} / {t2:,} / {t3:,}",
            _esc(task_types),
        ]
        if has_volumes:
            row.insert(6, f"{mc['volume']:,}")
        rows.append(row)

    table_cols = ["Dataset ID", "Store dir", "Anatomy Family", "Entries", "Images", "Videos"]
    if has_volumes:
        table_cols.append("Volumes")
    table_cols += ["Masked", "Tier 1/2/3", "Task Types"]
    report.table(table_cols, rows)

    # Per-anatomy-family summary plot
    fam_counts = defaultdict(int)
    for e in entries:
        fam_counts[e.anatomy_family] += 1
    fams_sorted = sorted(fam_counts, key=fam_counts.get, reverse=True)
    counts = [fam_counts[f] for f in fams_sorted]

    fig = _dark_fig(figsize=(14, 5))
    ax  = fig.add_subplot(111)
    _dark_ax(ax)
    bars = ax.barh(fams_sorted[::-1], counts[::-1], color="#6366f1", alpha=0.8)
    ax.set_xlabel("Entry count", color="#8892a4")
    ax.set_title("Entries per anatomy family", color="#e2e8f0", fontsize=11)
    for bar, val in zip(bars, counts[::-1]):
        ax.text(val + len(entries) * 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", color="#8892a4", fontsize=8)
    fig.tight_layout()
    report.figure(fig, "Entry counts by anatomy family")

    # Modality + SSL stream + curriculum tier plots
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
    fig2.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2, ax3):
        _dark_ax(ax)
    mod_labels, mod_values, mod_colors = ["image", "video"], [
        modalities["image"], modalities["video"],
    ], ["#60a5fa", "#4ade80"]
    if modalities["volume"]:
        mod_labels.append("volume")
        mod_values.append(modalities["volume"])
        mod_colors.append("#c084fc")
    ax1.pie(
        mod_values, labels=mod_labels, colors=mod_colors,
        autopct="%1.1f%%", textprops={"color": "#e2e8f0", "fontsize": 9},
    )
    ax1.set_title("Modality distribution", color="#e2e8f0", fontsize=10)
    stream_colors = ["#60a5fa", "#4ade80", "#fb923c"]
    ax2.pie(
        [streams["image"], streams["video"], streams["both"]],
        labels=["image", "video", "both"],
        colors=stream_colors, autopct="%1.1f%%",
        textprops={"color": "#e2e8f0", "fontsize": 9},
    )
    ax2.set_title("SSL stream routing", color="#e2e8f0", fontsize=10)
    tier_colors = ["#4ade80", "#fbbf24", "#f87171"]
    ax3.pie(
        [tiers[1], tiers[2], tiers[3]],
        labels=["Tier 1", "Tier 2", "Tier 3"],
        colors=tier_colors, autopct="%1.1f%%", textprops={"color": "#e2e8f0", "fontsize": 9},
    )
    ax3.set_title("Curriculum tier distribution", color="#e2e8f0", fontsize=10)
    fig2.tight_layout()
    report.figure(fig2, "Modality, SSL stream, and curriculum tier breakdown")


def section_label_spaces(report: HTMLReport):
    report.section("s2", "2. Label Spaces")
    report.html(
        "<p>Canonical class vocabularies from "
        "<code>data/labels/label_interface.py</code>. "
        "These are the integer class IDs used throughout the pipeline.</p>"
    )

    for anatomy, classes in sorted(ANATOMY_LABEL_SPACES.items()):
        default_head = ANATOMY_DEFAULT_HEAD.get(anatomy, "—")
        report.h3(anatomy)
        report.html(
            f"<p>Default head: <code>{_esc(str(default_head))}</code> &nbsp;|&nbsp; "
            f"{len(classes)} classes</p>"
        )
        rows = [[str(i), _esc(cls_name)] for i, cls_name in enumerate(classes)]
        report.html('<table class="label-space">'
                    '<thead><tr><th>ID</th><th>Class name</th></tr></thead><tbody>'
                    + "".join(f"<tr><td>{r[0]}</td><td>{r[1]}</td></tr>" for r in rows)
                    + "</tbody></table>")


def section_image_loading(
    report: HTMLReport,
    entries: List[USManifestEntry],
    n_per_dataset: int = 1,
):
    report.section("s3", "3. Image Loading Test")
    report.html(
        "<p>One representative thumbnail per dataset. Images loaded via "
        "<code>load_image()</code> and converted to RGB via "
        "<code>to_canonical_tensor()</code>. Datasets are grouped by anatomy family.</p>"
    )

    # Build dataset → entries map, grouped by anatomy family
    ds_by_family: Dict[str, Dict[str, List[USManifestEntry]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        ds_by_family[e.anatomy_family][e.dataset_id].append(e)

    loaded_for_masking: List[Tuple[str, str, np.ndarray]] = []  # (dataset_id, family, raw_np)

    for fam in sorted(ds_by_family):
        report.h3(f"Anatomy: {fam}")
        report.thumb_grid_open()

        for ds_id in sorted(ds_by_family[fam]):
            ds_entries = ds_by_family[fam][ds_id]
            # Sample actual image modalities; fall back to volumes then any entry.
            img_entries = [e for e in ds_entries if _is_image_entry(e)]
            if not img_entries:
                img_entries = [e for e in ds_entries if e.modality_type == "volume"]
            if not img_entries:
                img_entries = ds_entries
            max_tries = min(20, len(img_entries))
            candidates = random.sample(img_entries, max_tries)

            raw = None
            entry = candidates[0]
            for cand in candidates:
                raw = _load_entry_image(cand)
                if raw is not None:
                    entry = cand
                    break

            if raw is None:
                # Show error thumb
                fig = _dark_fig(figsize=(2, 2))
                ax  = fig.add_subplot(111)
                _dark_ax(ax)
                resolved = resolve_media_path(entry.image_paths[0], root_remap=_ROOT_REMAP)
                err_label = "LOAD\nFAILED" if resolved else "FILE\nNOT FOUND"
                ax.text(0.5, 0.5, err_label, ha="center", va="center",
                        color="#f87171", fontsize=9, transform=ax.transAxes)
                ax.axis("off")
                err_meta = "⚠ load failed" if resolved else "⚠ file not found"
                report.thumb_card(
                    fig, ds_id,
                    [f"family: {fam}", err_meta,
                     f"path: ...{entry.image_paths[0][-40:]}"],
                )
                continue

            # Convert to display tensor
            try:
                tensor = to_canonical_tensor(raw)
            except ValueError as exc:
                log.warning(
                    "to_canonical_tensor failed for %s (%s): %s",
                    ds_id, entry.image_paths[0], exc,
                )
                fig = _dark_fig(figsize=(2, 2))
                ax  = fig.add_subplot(111)
                _dark_ax(ax)
                ax.text(0.5, 0.5, "LOAD\nFAILED", ha="center", va="center",
                        color="#f87171", fontsize=9, transform=ax.transAxes)
                ax.axis("off")
                report.thumb_card(
                    fig, ds_id,
                    [f"family: {fam}", "⚠ load failed",
                     f"shape: {getattr(raw, 'shape', '?')}",
                     f"path: ...{entry.image_paths[0][-40:]}"],
                )
                continue
            img_np = tensor.permute(1, 2, 0).clamp(0, 1).numpy()
            H, W   = img_np.shape[:2]

            # Store for masking sections (max 2 per family)
            if len([x for x in loaded_for_masking if x[1] == fam]) < 2:
                loaded_for_masking.append((ds_id, fam, raw))

            # Create thumbnail figure
            fig = _dark_fig(figsize=(2, 2))
            ax  = fig.add_subplot(111)
            _dark_ax(ax)
            ax.imshow(img_np, interpolation="bilinear",
                      vmin=0, vmax=1)
            ax.axis("off")
            fig.tight_layout(pad=0)

            report.thumb_card(
                fig, ds_id,
                [
                    f"family: {fam}",
                    f"modality: {entry.modality_type}",
                    f"task: {entry.task_type}",
                    f"ssl: {entry.ssl_stream}",
                    f"tier: {entry.curriculum_tier}",
                    f"shape: {H}×{W}",
                    f"frames: {entry.num_frames}",
                ],
            )

        report.thumb_grid_close()

    # Store for later sections
    return loaded_for_masking


def section_video_loading(
    report: HTMLReport,
    entries: List[USManifestEntry],
    n_datasets: int = 8,
):
    report.section("s4", "4. Video Loading Test")
    report.html(
        "<p>Filmstrip (up to 6 evenly-spaced frames) per dataset with video entries. "
        "Loaded via <code>load_video_frames()</code>.</p>"
    )

    # One representative video-modality entry per dataset
    video_entries: Dict[str, USManifestEntry] = {}
    for e in entries:
        if e.dataset_id not in video_entries and _is_video_entry(e):
            video_entries[e.dataset_id] = e
        if len(video_entries) >= n_datasets * 3:
            break

    sampled_ds = random.sample(list(video_entries.keys()),
                               min(n_datasets, len(video_entries)))

    loaded_video_entries: List[Tuple[str, str, List]] = []  # (ds_id, fam, frames)

    for ds_id in sampled_ds:
        entry  = video_entries[ds_id]
        frames = _load_entry_video(entry, max_frames=8)

        if frames is None:
            report.warning(f"[{ds_id}] Could not load video frames: {entry.image_paths[0]}")
            continue

        n_show = min(6, len(frames))
        idxs   = np.linspace(0, len(frames) - 1, n_show, dtype=int)
        show_frames = [frames[i] for i in idxs]

        tensors = [to_canonical_tensor(f) for f in show_frames]
        imgs_np = [t.permute(1, 2, 0).clamp(0, 1).numpy() for t in tensors]
        H, W    = imgs_np[0].shape[:2]

        fig, axes = plt.subplots(1, n_show, figsize=(n_show * 2.5, 2.8))
        fig.patch.set_facecolor("#0f1117")
        if n_show == 1:
            axes = [axes]
        for ax, img, fi in zip(axes, imgs_np, idxs):
            _imshow_dark(ax, img, title=f"f{fi}")

        fps_str = f"{entry.fps:.1f} fps" if entry.fps else "? fps"
        fig.suptitle(
            f"{ds_id}  ·  {entry.anatomy_family}  ·  {len(frames)} frames  ·  {fps_str}  ·  {H}×{W}",
            color="#e2e8f0", fontsize=9, y=1.01,
        )
        fig.tight_layout()
        report.figure(fig,
                      f"{ds_id} | {entry.anatomy_family} | {len(frames)} total frames shown {n_show}")

        loaded_video_entries.append((ds_id, entry.anatomy_family, frames))

    if not loaded_video_entries:
        report.warning("No video entries with accessible files found.")

    return loaded_video_entries


def section_mask_loading(
    report: HTMLReport,
    entries: List[USManifestEntry],
    n_samples: int = 12,
):
    report.section("s5", "5. Segmentation Mask Loading Test")
    report.html(
        "<p>Image with mask overlay (red channel, α=0.5) for entries with "
        "<code>has_mask=True</code>. Loaded via <code>load_mask()</code>. "
        "Sampled across anatomy families.</p>"
    )

    # One representative masked entry per (anatomy, dataset) — scan the full manifest
    # so late datasets (e.g. FASS at ~640k lines) are not skipped by an early break.
    mask_entries: Dict[str, USManifestEntry] = {}
    for e in entries:
        if not (e.has_mask and e.instances):
            continue
        inst = e.instances[0]
        if not inst.mask_path:
            continue
        key = f"{e.anatomy_family}::{e.dataset_id}"
        if key not in mask_entries:
            mask_entries[key] = e

    selected = list(mask_entries.values())
    random.shuffle(selected)
    selected = selected[:n_samples]

    if not selected:
        report.warning("No mask entries found in the loaded manifest slice.")
        return

    n_cols = 3
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 5, n_rows * 3))
    fig.patch.set_facecolor("#0f1117")
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    shown = 0
    for row in range(n_rows):
        for col in range(n_cols):
            ax_img  = axes[row][col * 2]
            ax_mask = axes[row][col * 2 + 1]
            for ax in (ax_img, ax_mask):
                _dark_ax(ax)
                ax.axis("off")

            idx = row * n_cols + col
            if idx >= len(selected):
                continue

            entry = selected[idx]
            inst  = entry.instances[0]

            raw = _load_entry_image(entry)
            if raw is None:
                ax_img.text(0.5, 0.5, "no image", ha="center", va="center",
                            color="#f87171", transform=ax_img.transAxes)
                continue

            tensor  = to_canonical_tensor(raw)
            img_np  = tensor.permute(1, 2, 0).clamp(0, 1).numpy()
            H, W    = img_np.shape[:2]

            try:
                mask_arr = load_mask(
                    inst.mask_path,
                    frame_idx=0,
                    mask_channel=inst.mask_channel,
                )
                # Resize mask to match image if needed
                if mask_arr.shape != (H, W):
                    from PIL import Image as PILImage
                    mask_pil  = PILImage.fromarray((mask_arr * 255).astype(np.uint8))
                    mask_pil  = mask_pil.resize((W, H), PILImage.NEAREST)
                    mask_arr  = np.array(mask_pil).astype(np.uint8)

                # Overlay: image + red mask
                overlay = img_np.copy()
                msk_bin = mask_arr > 0
                overlay[msk_bin, 0] = np.clip(overlay[msk_bin, 0] * 0.5 + 0.5, 0, 1)
                overlay[msk_bin, 1] *= 0.3
                overlay[msk_bin, 2] *= 0.3

                ax_img.imshow(img_np, vmin=0, vmax=1)
                ax_img.set_title(f"{entry.dataset_id}", color="#e2e8f0", fontsize=7, pad=2)
                ax_mask.imshow(overlay, vmin=0, vmax=1)
                ax_mask.set_title(
                    f"{entry.anatomy_family} | {int(msk_bin.sum()/(H*W)*100)}% masked",
                    color="#e2e8f0", fontsize=7, pad=2,
                )
                shown += 1
            except Exception as exc:
                ax_img.imshow(img_np, vmin=0, vmax=1)
                ax_img.set_title(entry.dataset_id, color="#e2e8f0", fontsize=7)
                ax_mask.text(0.5, 0.5, f"mask error:\n{exc}", ha="center", va="center",
                             color="#f87171", fontsize=7, transform=ax_mask.transAxes,
                             wrap=True)

    fig.suptitle("Segmentation mask overlays", color="#e2e8f0", fontsize=10)
    fig.tight_layout()
    report.figure(fig, f"{shown}/{len(selected)} masks loaded successfully")


def section_masking_strategies(
    report: HTMLReport,
    loaded_images: List[Tuple[str, str, np.ndarray]],
    patch_size: int = 16,
    mask_ratio: float = 0.40,
    freq_cfg: Optional[FreqMaskConfig] = None,
):
    report.section("s6", "6. Masking Strategy Comparison")
    report.html(
        "<p>Same image passed through all three masking strategies. "
        "<b>Column 1</b>: original. "
        "<b>Column 2</b>: freq (spectral band zeroed in Fourier space). "
        "<b>Column 3</b>: spatial/black (random patches zeroed in pixel space). "
        "<b>Column 4</b>: both (freq first, then spatial on top). "
        "Row 2 for freq/both shows <b>original | degraded | |Δ| (amplified)</b> — "
        "spectral band removal is often subtle in pixel space, so the diff panel "
        "uses p99 scaling to make the change visible. "
        "Spatial column keeps a red patch overlay.</p>"
    )

    if freq_cfg is None:
        freq_cfg = FreqMaskConfig(mask_ratio=mask_ratio)

    # Use up to 8 representative images (diverse families)
    used_families = set()
    representative: List[Tuple[str, str, np.ndarray]] = []
    for ds_id, fam, raw in loaded_images:
        if fam not in used_families:
            representative.append((ds_id, fam, raw))
            used_families.add(fam)
        if len(representative) >= 8:
            break

    if not representative:
        report.warning("No loaded images available for masking comparison.")
        return

    for ds_id, fam, raw in tqdm(representative, desc="Masking strategy"):
        tensor = _pad_to_patch(to_canonical_tensor(raw), patch_size)
        _, H, W = tensor.shape
        img_np  = tensor.permute(1, 2, 0).clamp(0, 1).numpy()

        try:
            f_masked, f_mask = freq_mask_image(tensor, freq_cfg, patch_size)
            f_np = f_masked.permute(1, 2, 0).clamp(0, 1).numpy()
        except Exception as exc:
            log.warning("freq_mask_image failed for %s: %s", ds_id, exc)
            f_masked, f_mask, f_np = tensor, torch.zeros(H // patch_size, W // patch_size, dtype=torch.bool), img_np

        try:
            s_masked, s_mask = spatial_patch_mask(tensor, patch_size, mask_ratio)
            s_np = s_masked.permute(1, 2, 0).clamp(0, 1).numpy()
        except Exception as exc:
            log.warning("spatial_patch_mask failed for %s: %s", ds_id, exc)
            s_masked, s_mask, s_np = tensor, torch.zeros(H // patch_size, W // patch_size, dtype=torch.bool), img_np

        try:
            b_freq_masked, b_f_mask = freq_mask_image(tensor, freq_cfg, patch_size)
            b_masked, b_s_mask      = spatial_patch_mask(b_freq_masked, patch_size, mask_ratio)
            b_mask = b_f_mask | b_s_mask
            b_np   = b_masked.permute(1, 2, 0).clamp(0, 1).numpy()
        except Exception as exc:
            log.warning("both masking failed for %s: %s", ds_id, exc)
            b_masked, b_mask, b_np = tensor, torch.zeros(H // patch_size, W // patch_size, dtype=torch.bool), img_np

        # Build 2-row × 4-col figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.patch.set_facecolor("#0f1117")

        # Row 0: masked images
        _imshow_dark(axes[0][0], img_np,  "Original")
        _imshow_dark(axes[0][1], f_np,    f"Freq degraded ({int(f_mask.float().mean()*100)}% target patches)")
        _imshow_dark(axes[0][2], s_np,    f"Spatial ({int(s_mask.float().mean()*100)}% patches)")
        _imshow_dark(axes[0][3], b_np,    f"Both ({int(b_mask.float().mean()*100)}% patches)")

        # Row 1: detail views (degradation pairs for freq; overlay for spatial)
        axes[1][0].set_facecolor("#0f1117")
        axes[1][0].text(0.5, 0.5, f"{fam}\n{ds_id}\n{H}×{W}",
                        ha="center", va="center", color="#8892a4", fontsize=8,
                        transform=axes[1][0].transAxes)
        axes[1][0].axis("off")

        _imshow_freq_degradation_triple(axes[1][1], img_np, f_np, "Freq: orig | degraded | |Δ|")
        _imshow_spatial_mask_overlay(
            axes[1][2], s_np, s_mask, patch_size,
            f"Spatial overlay ({int(s_mask.float().mean()*100)}% patches)",
        )
        _imshow_freq_degradation_triple(axes[1][3], img_np, b_np, "Both: orig | student | |Δ|")

        fig.suptitle(f"{ds_id}  ·  {fam}  ·  {H}×{W}  ·  mask_ratio={mask_ratio}",
                     color="#e2e8f0", fontsize=10)
        fig.tight_layout()
        report.figure(fig, f"{ds_id} | {fam} — freq / spatial / both comparison")


def section_curriculum_masking(
    report: HTMLReport,
    entries: List[USManifestEntry],
    loaded_images: List[Tuple[str, str, np.ndarray]],
    patch_size: int = 16,
    freq_cfg: Optional[FreqMaskConfig] = None,
):
    report.section("s7", "7. Adaptive Curriculum Masking")
    report.html(
        "<p>Three curriculum stages applied using the <b>freq</b> masking strategy. "
        "Images are drawn from the manifest by tier (where possible) to also show the "
        "kind of data at each stage. Mask ratios: "
        f"Tier 1 = 40% · Tier 2 = 65% · Tier 3 = 80%.</p>"
    )

    # Pick one image per tier from the manifest (diverse families)
    tier_images: Dict[int, Optional[Tuple[str, str, np.ndarray]]] = {1: None, 2: None, 3: None}

    # First try to get tier-specific images from the manifest
    tier_entries: Dict[int, List[USManifestEntry]] = defaultdict(list)
    for e in entries:
        tier_entries[e.curriculum_tier].append(e)

    for tier in (1, 2, 3):
        for e in random.sample(tier_entries[tier], min(20, len(tier_entries[tier]))):
            raw = _load_entry_image(e)
            if raw is not None:
                tier_images[tier] = (e.dataset_id, e.anatomy_family, raw)
                break

    # Fall back to loaded_images if tier-specific load failed
    fallbacks = list(loaded_images)
    random.shuffle(fallbacks)
    for tier in (1, 2, 3):
        if tier_images[tier] is None and fallbacks:
            tier_images[tier] = fallbacks.pop()

    if all(v is None for v in tier_images.values()):
        report.warning("No images available for curriculum masking section.")
        return

    # For each image that we have, show tier 1/2/3 masking side by side
    # Show the image from its own tier + the other tier mask ratios for comparison
    # Layout: one row per tier-image source, 3 cols for T1/T2/T3 mask ratios

    # Also: show a comparison using the SAME image at all 3 mask ratios
    report.h3("Same image at increasing mask ratios (Tier 1 → 2 → 3)")

    # Pick one reference image
    ref = None
    for t in (1, 2, 3):
        if tier_images[t]:
            ref = tier_images[t]
            break

    if ref:
        ref_ds, ref_fam, ref_raw = ref
        ref_tensor = _pad_to_patch(to_canonical_tensor(ref_raw), patch_size)
        _, H, W    = ref_tensor.shape
        ref_np     = ref_tensor.permute(1, 2, 0).clamp(0, 1).numpy()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.patch.set_facecolor("#0f1117")
        _imshow_dark(axes[0][0], ref_np, "Original")
        axes[1][0].set_facecolor("#0f1117")
        axes[1][0].axis("off")

        for col, (tier, info) in enumerate(CURRICULUM_TIERS.items(), start=1):
            ratio = info["mask_ratio"]
            t_cfg = replace(freq_cfg, mask_ratio=ratio) if freq_cfg else FreqMaskConfig(mask_ratio=ratio)
            try:
                masked, mask = freq_mask_image(ref_tensor, t_cfg, patch_size)
                m_np = masked.permute(1, 2, 0).clamp(0, 1).numpy()
            except Exception as exc:
                log.warning("curriculum freq_mask failed tier=%d: %s", tier, exc)
                m_np = ref_np
                mask = torch.zeros(H // patch_size, W // patch_size, dtype=torch.bool)

            _imshow_dark(axes[0][col], m_np, info["label"])
            _imshow_freq_degradation_triple(
                axes[1][col], ref_np, m_np,
                f"Tier {tier}: orig | degraded | |Δ|",
            )

        fig.suptitle(
            f"Curriculum scaling — {ref_ds} · {ref_fam} · {H}×{W}",
            color="#e2e8f0", fontsize=10,
        )
        fig.tight_layout()
        report.figure(fig, "Same image — Tier 1 (40%) · Tier 2 (65%) · Tier 3 (80%) mask ratios")

    # Show tier-representative images (one from each tier) each at their own mask ratio
    report.h3("Tier-representative images at their native mask ratio")

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))
    fig2.patch.set_facecolor("#0f1117")

    for col, (tier, info) in enumerate(CURRICULUM_TIERS.items()):
        ax_img  = axes2[0][col]
        ax_mask = axes2[1][col]
        _dark_ax(ax_img)
        _dark_ax(ax_mask)

        entry_data = tier_images[tier]
        if entry_data is None:
            ax_img.text(0.5, 0.5, f"No tier-{tier} image loaded",
                        ha="center", va="center", color="#f87171",
                        transform=ax_img.transAxes)
            ax_img.axis("off")
            ax_mask.axis("off")
            continue

        ds_id, fam, raw = entry_data
        tensor = _pad_to_patch(to_canonical_tensor(raw), patch_size)
        _, H, W = tensor.shape
        img_np  = tensor.permute(1, 2, 0).clamp(0, 1).numpy()

        ratio = info["mask_ratio"]
        t_cfg = replace(freq_cfg, mask_ratio=ratio) if freq_cfg else FreqMaskConfig(mask_ratio=ratio)
        try:
            masked, mask = freq_mask_image(tensor, t_cfg, patch_size)
            m_np = masked.permute(1, 2, 0).clamp(0, 1).numpy()
        except Exception as exc:
            log.warning("curriculum freq_mask failed tier=%d: %s", tier, exc)
            m_np, mask = img_np, torch.zeros(H // patch_size, W // patch_size, dtype=torch.bool)

        ax_img.imshow(m_np, vmin=0, vmax=1)
        ax_img.set_title(
            f"{info['label']}\n{ds_id} · {fam}",
            color="#e2e8f0", fontsize=9, pad=4,
        )
        ax_img.axis("off")

        _imshow_freq_degradation_triple(
            ax_mask, img_np, m_np,
            f"Tier {tier}: orig | degraded | |Δ|",
        )

    fig2.suptitle("Tier-representative images at their curriculum mask ratio",
                  color="#e2e8f0", fontsize=11)
    fig2.tight_layout()
    report.figure(fig2, "Tier 1 / 2 / 3 representative images with freq masking")


def section_video_masking(
    report: HTMLReport,
    loaded_videos: List[Tuple[str, str, List]],
    patch_size: int = 16,
):
    report.section("s8", "8. Video Tube Masking Test")
    report.html(
        "<p>For each video clip: <b>Row 1</b> = teacher (clean full clip); "
        "<b>Row 2</b> = student (freq-tube-masked visible clip); "
        "<b>Row 3</b> = tube mask heatmap per frame "
        "(red = masked spatiotemporal tube). "
        "Strategy: <code>freq</code> · tube_mask_ratio=0.75 · tube_size=2.</p>"
    )

    if not loaded_videos:
        report.warning("No video clips loaded — skipping video masking section.")
        return

    ps = patch_size

    for ds_id, fam, raw_frames in tqdm(loaded_videos, desc="Video masking"):
        n_show   = min(8, len(raw_frames))
        idxs     = np.linspace(0, len(raw_frames) - 1, n_show, dtype=int)
        frames   = [raw_frames[i] for i in idxs]

        # Build clip tensor (T, 3, H, W)
        tensors  = [_pad_to_patch(to_canonical_tensor(f), ps) for f in frames]
        # Ensure same spatial size (take min H, W)
        min_H    = min(t.shape[1] for t in tensors)
        min_W    = min(t.shape[2] for t in tensors)
        min_H    = (min_H // ps) * ps
        min_W    = (min_W // ps) * ps
        tensors  = [t[:, :min_H, :min_W] for t in tensors]
        clip     = torch.stack(tensors)           # (T, C, H, W)
        T, C, H, W = clip.shape
        ph, pw   = H // ps, W // ps

        # Apply freq tube masking
        from data.pipeline.transforms import freq_mask_video
        freq_cfg = FreqMaskConfig(mask_ratio=0.75, use_alp_bias=True)
        try:
            visible_clip, tube_mask = freq_mask_video(
                clip, freq_cfg, ps, mask_ratio_override=0.75, tube_size=2
            )
        except Exception as exc:
            report.warning(f"[{ds_id}] Video freq masking failed: {exc}")
            visible_clip = clip
            tube_mask    = torch.zeros(T, ph, pw, dtype=torch.bool)

        # Apply spatial tube masking for comparison
        from data.pipeline.transforms import spatial_tube_mask
        try:
            spatial_visible, spatial_tube = spatial_tube_mask(clip, ps, 0.75, tube_size=2)
        except Exception as exc:
            log.warning("spatial_tube_mask failed for %s: %s", ds_id, exc)
            spatial_visible = clip
            spatial_tube    = torch.zeros(T, ph, pw, dtype=torch.bool)

        # Full clip (teacher)
        full_np    = [clip[t].permute(1, 2, 0).clamp(0, 1).numpy()     for t in range(T)]
        # Freq masked (student)
        vis_np     = [visible_clip[t].permute(1, 2, 0).clamp(0, 1).numpy() for t in range(T)]
        # Spatial masked
        sp_np      = [spatial_visible[t].permute(1, 2, 0).clamp(0, 1).numpy() for t in range(T)]

        fig, axes = plt.subplots(4, T, figsize=(T * 2.2, 4 * 2.4), squeeze=False)
        fig.patch.set_facecolor("#0f1117")

        row_labels = ["Teacher (full)", "Student (freq)", "Student (spatial)", "Tube mask (freq)"]
        for row, (row_imgs, label) in enumerate(zip(
            [full_np, vis_np, sp_np, None], row_labels
        )):
            for col in range(T):
                ax = axes[row][col]
                _dark_ax(ax)
                ax.axis("off")
                if row < 3:
                    ax.imshow(row_imgs[col], vmin=0, vmax=1)
                    if col == 0:
                        ax.set_ylabel(label, color="#e2e8f0", fontsize=8,
                                      rotation=0, labelpad=60, va="center")
                else:
                    # Tube mask heatmap
                    mask_frame = tube_mask[col].float().numpy()
                    ax.imshow(mask_frame, cmap="Reds", vmin=0, vmax=1,
                              interpolation="nearest",
                              extent=[0, pw, ph, 0])
                    if col == 0:
                        ax.set_ylabel("Tube mask", color="#e2e8f0", fontsize=8,
                                      rotation=0, labelpad=60, va="center")
                if row == 0:
                    ax.set_title(f"f{int(idxs[col])}", color="#8892a4", fontsize=8)

        pct = int(tube_mask.float().mean().item() * 100)
        fig.suptitle(
            f"{ds_id}  ·  {fam}  ·  {T} frames  ·  {H}×{W}  "
            f"·  freq tube masking  ·  {pct}% masked",
            color="#e2e8f0", fontsize=10, y=1.01,
        )
        fig.tight_layout()
        report.figure(fig,
                      f"{ds_id} | {fam} — teacher / freq-student / spatial-student / tube-mask")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Ultatron data pipeline analysis report")
    ap.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "dataset_exploration_outputs/run1/run1_train_v3.jsonl"),
        help="Path to the JSONL manifest",
    )
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "dataset_exploration_outputs/data_prep_analysis.html"),
        help="Output HTML path",
    )
    ap.add_argument(
        "--n-per-dataset", type=int, default=1,
        help="Thumbnail samples per dataset in Section 3",
    )
    ap.add_argument(
        "--n-video-datasets", type=int, default=8,
        help="Max video datasets to sample in Sections 4 and 8",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling",
    )
    ap.add_argument(
        "--patch-size", type=int, default=16,
        help="Patch size used for all masking operations",
    )
    ap.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING"),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(message)s",
    )

    from scripts.ensure_deps import ensure_deps
    ensure_deps()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    print(f"Loading manifest: {manifest_path}", flush=True)
    entries = load_manifest(manifest_path)
    print(f"  → {len(entries):,} entries loaded", flush=True)

    global _ROOT_REMAP
    try:
        from data.infra.storage import StorageConfig
        _ROOT_REMAP = StorageConfig().build_root_remap()
        if _ROOT_REMAP:
            print(f"  → root remap active: store → scratch", flush=True)
    except Exception as exc:
        log.warning("Could not build root remap: %s", exc)
        _ROOT_REMAP = {}

    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    freq_cfg = FreqMaskConfig(
        r_inner_min=0.10, r_outer_max=0.95,
        band_width_min=0.15, band_width_max=0.40,
        mask_ratio=0.40, use_alp_bias=True,
        n_bands=1,
    )

    with HTMLReport(
        path=args.out,
        title="Ultatron — Data Pipeline Analysis",
        subtitle=(
            f"Manifest: {manifest_path.name}  ·  "
            f"{len(entries):,} entries  ·  generated {now_str}"
        ),
    ) as report:

        # ── Section 1: Manifest summary ───────────────────────────────────────
        print("Section 1: Manifest summary …", flush=True)
        section_manifest_summary(report, entries)
        report.save()  # save incrementally

        # ── Section 2: Label spaces ───────────────────────────────────────────
        print("Section 2: Label spaces …", flush=True)
        section_label_spaces(report)
        report.save()

        # ── Section 3: Image loading ──────────────────────────────────────────
        print("Section 3: Image loading (all datasets) …", flush=True)
        loaded_images = section_image_loading(
            report, entries, n_per_dataset=args.n_per_dataset
        )
        report.save()

        # ── Section 4: Video loading ──────────────────────────────────────────
        print("Section 4: Video loading …", flush=True)
        loaded_videos = section_video_loading(
            report, entries, n_datasets=args.n_video_datasets
        )
        report.save()

        # ── Section 5: Mask loading ───────────────────────────────────────────
        print("Section 5: Segmentation mask loading …", flush=True)
        section_mask_loading(report, entries, n_samples=12)
        report.save()

        # ── Section 6: Masking strategy comparison ────────────────────────────
        print("Section 6: Masking strategy comparison …", flush=True)
        section_masking_strategies(
            report, loaded_images,
            patch_size=args.patch_size, mask_ratio=0.40, freq_cfg=freq_cfg,
        )
        report.save()

        # ── Section 7: Adaptive curriculum masking ────────────────────────────
        print("Section 7: Adaptive curriculum masking …", flush=True)
        section_curriculum_masking(
            report, entries, loaded_images,
            patch_size=args.patch_size, freq_cfg=freq_cfg,
        )
        report.save()

        # ── Section 8: Video tube masking ─────────────────────────────────────
        print("Section 8: Video tube masking …", flush=True)
        section_video_masking(
            report, loaded_videos, patch_size=args.patch_size
        )

    print(f"\nReport written → {args.out}", flush=True)


if __name__ == "__main__":
    main()
