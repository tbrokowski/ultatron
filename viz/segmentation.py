"""
viz/segmentation.py  ·  Segmentation prediction visualisation
==================================================================

Shows predictions vs ground truth, error maps, and per-dataset Dice distributions.

Plots:
  1. Overlay panel         — image | GT mask | prediction | error map
  2. Dice histogram        — distribution of per-sample Dice scores
  3. Failure analysis      — worst-performing samples in a grid
  4. Anatomy stratification — boxplot of Dice per anatomy family
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from viz.core import (
    to_numpy_image, mask_overlay, overlay_heatmap,
    save_figure, _get_plt,
)
from eval.metrics import dice_score


def plot_segmentation_overlay(
    image:       np.ndarray,          # (H, W, 3) or (H, W) uint8
    pred_mask:   np.ndarray,          # (H, W) float or binary
    gt_mask:     Optional[np.ndarray] = None,   # (H, W)
    threshold:   float = 0.5,
    title:       str   = "",
    save_path:   Optional[str] = None,
):
    """
    4-panel visualisation: image | GT | prediction | difference map.
    If gt_mask is None, shows only image + prediction.
    """
    plt = _get_plt()

    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    pred_bin = (pred_mask >= threshold).astype(np.uint8)

    n_panels = 4 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 3.5, 3.8))

    axes[0].imshow(image)
    axes[0].set_title("Image", fontsize=10)
    axes[0].axis("off")

    if gt_mask is not None:
        axes[1].imshow(mask_overlay(image, gt_mask >= threshold,
                                     colour=(80, 200, 80)))
        dc = dice_score(pred_bin, (gt_mask >= threshold).astype(float))
        axes[1].set_title(f"Ground truth  (Dice={dc:.3f})", fontsize=10)
        axes[1].axis("off")

        pred_panel = axes[2]
        diff_panel = axes[3]
    else:
        pred_panel = axes[1]
        diff_panel = None

    pred_panel.imshow(mask_overlay(image, pred_bin, colour=(255, 80, 80)))
    pred_panel.set_title("Prediction", fontsize=10)
    pred_panel.axis("off")

    if diff_panel is not None and gt_mask is not None:
        gt_bin  = (gt_mask  >= threshold).astype(np.float32)
        diff    = pred_bin.astype(np.float32) - gt_bin
        # Red = false positive, Blue = false negative
        diff_rgb = np.zeros((*diff.shape, 3), dtype=np.uint8)
        diff_rgb[diff  > 0] = [255, 80,  80]   # FP: red
        diff_rgb[diff  < 0] = [80,  80, 255]   # FN: blue
        # Background: faint image
        bg = (image.astype(float) * 0.3).astype(np.uint8)
        bg[diff != 0] = diff_rgb[diff != 0]
        diff_panel.imshow(bg)
        diff_panel.set_title("Error  [red=FP, blue=FN]", fontsize=10)
        diff_panel.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_segmentation_grid(
    images:      list[np.ndarray],      # list of (H, W, 3) or (H, W)
    pred_masks:  list[np.ndarray],
    gt_masks:    Optional[list[np.ndarray]] = None,
    sample_ids:  Optional[list[str]] = None,
    n_cols:      int = 4,
    threshold:   float = 0.5,
    title:       str   = "Segmentation results",
    save_path:   Optional[str] = None,
):
    """
    Grid of overlay images.  Each cell shows image + predicted mask.
    If gt_masks given, the title of each cell shows Dice score.
    """
    plt = _get_plt()

    n      = min(len(images), 32)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.0, n_rows * 3.2))
    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i in range(n):
        ax  = axes_flat[i]
        img = images[i]
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        p   = (pred_masks[i] >= threshold).astype(np.uint8)
        vis = mask_overlay(img, p, colour=(255, 100, 100))
        ax.imshow(vis)

        cell_title = sample_ids[i] if sample_ids else f"#{i}"
        if gt_masks is not None:
            g  = (gt_masks[i] >= threshold).astype(float)
            dc = dice_score(p.astype(float), g)
            cell_title += f"\nDice={dc:.3f}"
        ax.set_title(cell_title, fontsize=7)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_dice_distribution(
    dice_scores:     np.ndarray,           # (N,) float
    anatomy_labels:  Optional[list[str]] = None,
    title:           str = "Dice score distribution",
    save_path:       Optional[str] = None,
):
    """Histogram + optional anatomy-stratified boxplot of Dice scores."""
    plt = _get_plt()

    has_anatomy = anatomy_labels is not None and len(set(anatomy_labels)) > 1

    if has_anatomy:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    else:
        fig, ax1 = plt.subplots(figsize=(6, 4.5))

    ax1.hist(dice_scores, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
    ax1.axvline(dice_scores.mean(), color="red", linestyle="--",
                label=f"Mean={dice_scores.mean():.3f}")
    ax1.axvline(np.median(dice_scores), color="orange", linestyle="--",
                label=f"Median={np.median(dice_scores):.3f}")
    ax1.set_title(title, fontsize=11)
    ax1.set_xlabel("Dice coefficient")
    ax1.set_ylabel("Count")
    ax1.set_xlim(0, 1)
    ax1.legend()

    if has_anatomy:
        families = sorted(set(anatomy_labels))
        data     = [dice_scores[np.array(anatomy_labels) == f] for f in families]
        ax2.boxplot(data, labels=families, patch_artist=True,
                    boxprops=dict(facecolor="steelblue", alpha=0.6))
        ax2.set_xticklabels(families, rotation=30, ha="right", fontsize=8)
        ax2.set_title("Dice by anatomy family", fontsize=11)
        ax2.set_ylabel("Dice coefficient")
        ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig
