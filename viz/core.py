"""
viz/core.py  ·  Shared visualisation utilities
====================================================

Provides the low-level building blocks used by every other viz module:
  - Device / dtype normalisation
  - Patch token → 2-D spatial map reshaping
  - Colourmap helpers (jet, viridis, seismic for difference maps)
  - Figure save with metadata
  - Image denormalisation and numpy conversion
  - Batch singleton extraction

All functions are pure (no side effects on model state).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# ── Matplotlib lazy import ────────────────────────────────────────────────────
# Import lazily so the module can be imported in headless/CSCS environments
# where matplotlib may not be configured. All viz functions import plt locally.

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe on CSCS
    import matplotlib.pyplot as plt
    return plt


# ── Tensor / numpy utilities ──────────────────────────────────────────────────

def to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert (3, H, W) or (1, H, W) float [0,1] tensor → (H, W, C) uint8 numpy.
    """
    if x.ndim == 4:
        x = x[0]
    x = x.detach().cpu().float().clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return arr


def tokens_to_spatial(
    tokens: torch.Tensor,   # (B, N, D) or (N, D)
    ph: Optional[int] = None,
    pw: Optional[int] = None,
) -> torch.Tensor:
    """
    Reshape flat patch tokens back to a spatial grid.

    If ph/pw not given, assumes a square grid (ph = pw = sqrt(N)).

    Returns (B, D, ph, pw) or (D, ph, pw).
    """
    if tokens.ndim == 2:
        tokens = tokens.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, N, D = tokens.shape
    if ph is None:
        ph = pw = int(N ** 0.5)
        assert ph * pw == N, f"N={N} is not a perfect square; supply ph, pw explicitly."

    spatial = tokens.permute(0, 2, 1).reshape(B, D, ph, pw)
    return spatial[0] if squeeze else spatial


def scalar_map_to_heatmap(
    values: np.ndarray,      # (ph, pw) float
    cmap:   str  = "jet",
    vmin:   Optional[float] = None,
    vmax:   Optional[float] = None,
) -> np.ndarray:
    """
    Convert a scalar map → (H, W, 3) uint8 RGB heatmap using matplotlib colormap.
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    vmin = vmin if vmin is not None else float(values.min())
    vmax = vmax if vmax is not None else float(values.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    rgba   = mapper.to_rgba(values)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def overlay_heatmap(
    image_rgb:  np.ndarray,   # (H, W, 3) or (H, W) uint8
    heatmap:    np.ndarray,   # (ph, pw) float or (H, W) float
    alpha:      float = 0.55,
    cmap:       str   = "jet",
    upsample:   bool  = True,
) -> np.ndarray:
    """
    Blend a heatmap onto an image.

    Returns (H, W, 3) uint8.
    """
    H, W = image_rgb.shape[:2]
    if image_rgb.ndim == 2:
        image_rgb = np.stack([image_rgb] * 3, axis=-1)

    # Upsample heatmap to image resolution if needed
    if upsample and heatmap.shape != (H, W):
        heat_t = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
        heat_t = F.interpolate(heat_t, size=(H, W), mode="bilinear", align_corners=False)
        heatmap = heat_t.squeeze().numpy()

    heat_rgb = scalar_map_to_heatmap(heatmap, cmap=cmap)
    blended  = (
        (1 - alpha) * image_rgb.astype(np.float32)
        + alpha     * heat_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    return blended


def mask_overlay(
    image_rgb:    np.ndarray,         # (H, W, 3) uint8
    mask:         np.ndarray,         # (H, W) binary
    colour:       Tuple[int,int,int] = (255, 80, 80),
    alpha:        float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask on an image with a solid colour."""
    out = image_rgb.astype(np.float32).copy()
    idx = mask.astype(bool)
    out[idx] = (
        (1 - alpha) * out[idx]
        + alpha     * np.array(colour, dtype=np.float32)
    )
    return out.clip(0, 255).astype(np.uint8)


def save_figure(
    fig,
    path:     Union[str, Path],
    dpi:      int  = 150,
    metadata: dict = None,
):
    """Save a matplotlib figure to disk, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                metadata=metadata or {})
    return path


# ── Model inference helpers ───────────────────────────────────────────────────

@torch.no_grad()
def extract_patch_features(
    img_branch,
    image_rgb: torch.Tensor,          # (3, H, W) float [0,1]
    device:    str = "cuda",
) -> dict:
    """
    Single-image forward through teacher backbone.
    Returns {cls, patch_tokens, register_tokens, ph, pw, H, W}.
    """
    x = image_rgb.unsqueeze(0).to(device)
    out = img_branch.forward_teacher(x)
    N   = out["patch_tokens"].shape[1]
    ph  = pw = int(N ** 0.5)
    return {
        "cls":             out["cls"][0],                     # (D,)
        "patch_tokens":    out["patch_tokens"][0],            # (N, D)
        "register_tokens": out.get("register_tokens", [None])[0],
        "ph": ph, "pw": pw,
        "H":  image_rgb.shape[1], "W": image_rgb.shape[2],
    }


@torch.no_grad()
def extract_attentions(
    img_branch,
    image_rgb: torch.Tensor,
    device:    str = "cuda",
) -> Optional[torch.Tensor]:
    """
    Extract last-layer attention weights from the teacher backbone.
    Returns (n_heads, seq_len, seq_len) or None if model doesn't expose attentions.

    HuggingFace DINOv3: pass output_attentions=True.
    Attentions shape per layer: (B, n_heads, seq, seq)
    seq = 1 (CLS) + n_reg (registers) + N (patches)
    """
    x  = image_rgb.unsqueeze(0).to(device)
    rgb = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x

    vit = img_branch.teacher._vit if hasattr(img_branch.teacher, "_vit") \
          else img_branch.teacher.vit
    n_reg = getattr(vit.config, "num_register_tokens", 0)

    out = vit(pixel_values=rgb, output_attentions=True)
    if not hasattr(out, "attentions") or out.attentions is None:
        return None, n_reg

    last_attn = out.attentions[-1][0]   # (n_heads, seq, seq)
    return last_attn, n_reg
