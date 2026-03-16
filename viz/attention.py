"""
viz/attention.py  ·  Attention map visualisation
======================================================

The most interpretable window into what the backbone has learned.

Three views are produced:
  1. Per-head CLS attention   — what each of the 16 attention heads focuses
                                on when computing the global representation.
  2. Mean CLS attention        — average across heads; reveals consensus.
  3. Head diversity map        — std across heads; highlights where heads disagree
                                (often semantically rich boundaries).
  4. Last-layer full attention — the (N+1+R, N+1+R) full attention matrix
                                for the CLS token row, useful for debugging.

DINOv3 attention: last_hidden_state comes from a transformer where
  seq = [CLS, reg_0, ..., reg_R, patch_0, ..., patch_N-1]
  CLS attention to patches: attentions[-1][:, h, 0, 1+R:]  → (B, heads, N)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from viz.core import (
    to_numpy_image, scalar_map_to_heatmap, overlay_heatmap,
    extract_attentions, save_figure, _get_plt,
)


def extract_cls_attention(
    img_branch,
    image_rgb: torch.Tensor,   # (3, H, W) float [0,1]
    device:    str = "cuda",
) -> Optional[dict]:
    """
    Extract CLS-to-patch attention weights from the last transformer layer.

    Returns dict:
        per_head  : (n_heads, ph, pw) float  — CLS attention per head
        mean      : (ph, pw)          float  — mean across heads
        std       : (ph, pw)          float  — std across heads (diversity)
        n_heads   : int
        ph, pw    : int
    or None if model doesn't expose attention weights.
    """
    attn, n_reg = extract_attentions(img_branch, image_rgb, device)
    if attn is None:
        return None

    n_heads, seq_len, _ = attn.shape
    N                    = seq_len - 1 - n_reg          # number of patches
    ph = pw              = int(N ** 0.5)

    # CLS row: attn[h, 0, :] — slice patch positions
    cls_attn = attn[:, 0, 1 + n_reg:]                   # (n_heads, N)
    cls_attn = cls_attn.float().cpu().numpy()
    cls_attn = cls_attn.reshape(n_heads, ph, pw)

    return {
        "per_head": cls_attn,                            # (H_heads, ph, pw)
        "mean":     cls_attn.mean(axis=0),               # (ph, pw)
        "std":      cls_attn.std(axis=0),                # (ph, pw) diversity
        "n_heads":  n_heads,
        "ph": ph,   "pw": pw,
    }


def plot_attention_maps(
    img_branch,
    image_rgb:   torch.Tensor,   # (3, H, W) float [0,1]
    device:      str   = "cuda",
    max_heads:   int   = 16,
    alpha:       float = 0.55,
    save_path:   Optional[str] = None,
):
    """
    Plot per-head CLS attention maps overlaid on the input image.

    Layout: 4 rows × 4 columns (for 16-head DINOv3-L)
      Row 0–3: individual attention head maps
      Bottom row extra: mean attention, head diversity, side-by-side

    Returns a matplotlib Figure.
    """
    plt = _get_plt()

    attn_data = extract_cls_attention(img_branch, image_rgb, device)
    if attn_data is None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Model does not expose attention weights",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    img_np   = to_numpy_image(image_rgb)
    per_head = attn_data["per_head"]     # (n_heads, ph, pw)
    mean_map = attn_data["mean"]
    std_map  = attn_data["std"]
    n_heads  = min(attn_data["n_heads"], max_heads)

    # Grid layout: all heads + mean + diversity
    n_cols  = 4
    n_rows  = (n_heads + n_cols - 1) // n_cols + 1  # extra row for mean+std
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3.2))
    axes = axes.flatten()

    for h in range(n_heads):
        ax  = axes[h]
        vis = overlay_heatmap(img_np, per_head[h], alpha=alpha, cmap="inferno")
        ax.imshow(vis)
        ax.set_title(f"Head {h}", fontsize=9)
        ax.axis("off")

    # Mean and diversity in remaining slots
    n_extra = len(axes) - n_heads
    if n_extra >= 1:
        ax = axes[n_heads]
        ax.imshow(overlay_heatmap(img_np, mean_map, alpha=alpha, cmap="inferno"))
        ax.set_title("Mean (all heads)", fontsize=9, fontweight="bold")
        ax.axis("off")
    if n_extra >= 2:
        ax = axes[n_heads + 1]
        ax.imshow(overlay_heatmap(img_np, std_map, alpha=alpha, cmap="plasma"))
        ax.set_title("Head diversity (std)", fontsize=9)
        ax.axis("off")

    # Turn off unused axes
    for ax in axes[n_heads + 2:]:
        ax.axis("off")

    fig.suptitle("CLS Attention Maps (last layer)", fontsize=12, y=1.01)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_attention_evolution(
    img_branch,
    image_rgb:   torch.Tensor,
    device:      str   = "cuda",
    layers:      Optional[list] = None,
    save_path:   Optional[str] = None,
):
    """
    Show how CLS attention evolves across transformer layers.

    layers: list of layer indices to visualise (default: [0, 6, 11] for 12-layer)
    """
    plt = _get_plt()

    vit   = img_branch.teacher._vit if hasattr(img_branch.teacher, "_vit") \
            else img_branch.teacher.vit
    n_reg = getattr(vit.config, "num_register_tokens", 0)

    rgb   = image_rgb.unsqueeze(0).to(device)
    if rgb.shape[1] == 1:
        rgb = rgb.expand(-1, 3, -1, -1)

    with torch.no_grad():
        out = vit(pixel_values=rgb, output_attentions=True,
                  output_hidden_states=False)

    if not hasattr(out, "attentions") or out.attentions is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No attention weights available",
                ha="center", va="center")
        return fig

    all_attns = out.attentions   # tuple of (1, H, seq, seq) per layer
    n_layers  = len(all_attns)
    layers    = layers or [0, n_layers // 4, n_layers // 2,
                            3 * n_layers // 4, n_layers - 1]
    layers    = [l for l in layers if 0 <= l < n_layers]

    img_np = to_numpy_image(image_rgb)
    fig, axes = plt.subplots(1, len(layers), figsize=(len(layers) * 3.5, 3.5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer_idx in zip(axes, layers):
        attn   = all_attns[layer_idx][0]           # (H, seq, seq)
        N      = attn.shape[-1] - 1 - n_reg
        ph = pw = int(N ** 0.5)
        mean_attn = attn[:, 0, 1 + n_reg:].float().cpu().numpy().mean(0)
        mean_attn = mean_attn.reshape(ph, pw)
        ax.imshow(overlay_heatmap(img_np, mean_attn, alpha=0.6, cmap="inferno"))
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.axis("off")

    fig.suptitle("CLS Attention across layers (mean over heads)", fontsize=11)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig
