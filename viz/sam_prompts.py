"""
viz/sam_prompts.py  ·  MedSAM prompt and refinement visualisation
=======================================================================

Visualises the agent loop from agents/agent_loop.py:
  1. Attention → prompt points  — which patches trigger SAM prompt placement
  2. Iterative mask refinement  — overlay masks from each SAM iteration
  3. Prompt confidence          — which prompt points the model is confident about
  4. Mask quality over iters    — Dice vs iteration number

All functions accept pre-computed data (attention maps, masks) rather than
running the agent — keeping viz purely a presentation layer.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from viz.core import (
    to_numpy_image, mask_overlay, overlay_heatmap,
    save_figure, _get_plt,
)


def plot_prompt_placement(
    image_rgb:       np.ndarray,        # (H, W, 3) uint8
    attention_map:   np.ndarray,        # (ph, pw) float — CLS attention
    prompt_points:   np.ndarray,        # (N_pts, 2) float — (row, col) in image coords
    prompt_labels:   Optional[np.ndarray] = None,   # (N_pts,) 1=foreground, 0=background
    gt_mask:         Optional[np.ndarray] = None,   # (H, W)
    title:           str   = "SAM prompt placement",
    save_path:       Optional[str] = None,
):
    """
    Show how attention maps guide SAM prompt point placement.

    Left:  raw image + gt mask outline (if available)
    Middle: attention heatmap (source of prompt signal)
    Right: image + prompt points (green=foreground, red=background)
    """
    plt = _get_plt()

    H, W = image_rgb.shape[:2]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    # Panel 1: image + GT
    ax = axes[0]
    if gt_mask is not None:
        vis = mask_overlay(image_rgb, gt_mask > 0, colour=(80, 220, 80), alpha=0.35)
    else:
        vis = image_rgb
    ax.imshow(vis)
    ax.set_title("Image" + (" + GT" if gt_mask is not None else ""), fontsize=10)
    ax.axis("off")

    # Panel 2: attention heatmap
    axes[1].imshow(overlay_heatmap(image_rgb, attention_map, alpha=0.65, cmap="inferno"))
    axes[1].set_title("CLS attention (prompt signal)", fontsize=10)
    axes[1].axis("off")

    # Panel 3: prompt points
    axes[2].imshow(image_rgb)
    if len(prompt_points) > 0:
        fg_idx = np.ones(len(prompt_points), dtype=bool)
        if prompt_labels is not None:
            fg_idx = prompt_labels.astype(bool)
        for is_fg, pts in [(True, prompt_points[fg_idx]),
                            (False, prompt_points[~fg_idx])]:
            if len(pts) == 0:
                continue
            colour = "lime" if is_fg else "red"
            marker = "*"    if is_fg else "x"
            axes[2].scatter(pts[:, 1], pts[:, 0],
                            c=colour, marker=marker, s=100,
                            edgecolors="white", linewidths=0.5, zorder=5,
                            label="Foreground" if is_fg else "Background")
    axes[2].set_title("Prompt points", fontsize=10)
    axes[2].legend(fontsize=8, loc="lower right")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_iterative_refinement(
    image_rgb:   np.ndarray,            # (H, W, 3)
    masks:       list[np.ndarray],      # list of (H, W) binary masks, one per iter
    gt_mask:     Optional[np.ndarray] = None,
    dice_scores: Optional[list[float]] = None,
    title:       str = "SAM iterative refinement",
    save_path:   Optional[str] = None,
):
    """
    Show how the segmentation mask evolves across SAM refinement iterations.

    Each panel = one iteration.  Title shows iteration Dice vs GT if available.
    """
    plt = _get_plt()
    from eval.metrics import dice_score as dice_fn

    n     = len(masks)
    n_cols= min(n, 6)
    n_rows= (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    colours = [
        (255, 80,  80),    # iter 0: red
        (255, 140, 0),     # iter 1: orange
        (50,  200, 80),    # iter 2: green
        (50,  150, 255),   # iter 3: blue
        (180, 80,  220),   # iter 4: purple
        (255, 220, 50),    # iter 5: yellow
    ]

    for i, mask in enumerate(masks):
        ax  = axes_flat[i]
        col = colours[i % len(colours)]
        vis = mask_overlay(image_rgb, mask > 0, colour=col, alpha=0.50)

        if gt_mask is not None:
            # Overlay GT contour
            from scipy.ndimage import binary_erosion
            contour = gt_mask > 0.5
            inner   = binary_erosion(contour, iterations=2)
            outline = contour & ~inner
            vis[outline] = [255, 255, 255]

        ax.imshow(vis)
        iter_title = f"Iter {i}"
        if dice_scores and i < len(dice_scores):
            iter_title += f"  Dice={dice_scores[i]:.3f}"
        elif gt_mask is not None:
            d = dice_fn(mask.astype(float), (gt_mask > 0).astype(float))
            iter_title += f"  Dice={d:.3f}"
        ax.set_title(iter_title, fontsize=9)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_mask_quality_curve(
    dice_per_iter:   list[float],
    title:           str = "Mask quality vs SAM iteration",
    save_path:       Optional[str] = None,
):
    """Line plot of Dice vs iteration number across multiple samples."""
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(dice_per_iter)), dice_per_iter, "o-",
            color="steelblue", linewidth=2, markersize=6)
    ax.axhline(dice_per_iter[0], color="grey", linestyle=":",
               alpha=0.5, label=f"Iter 0: {dice_per_iter[0]:.3f}")
    ax.axhline(max(dice_per_iter), color="green", linestyle="--",
               alpha=0.7, label=f"Best: {max(dice_per_iter):.3f}")
    ax.set_xlabel("SAM refinement iteration")
    ax.set_ylabel("Dice vs ground truth")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(dice_per_iter)))
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig
