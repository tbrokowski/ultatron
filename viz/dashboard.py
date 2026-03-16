"""
viz/dashboard.py  ·  Multi-panel summary dashboard
========================================================

A single-call function that produces a comprehensive A3-format summary
figure for one training checkpoint, useful for experiment tracking and
slide decks.

Dashboard layout (4 rows × 4 cols):
  Row 0: [Input image] [Attention mean] [Prototype map] [Seg overlay]
  Row 1: [AUC curve]   [Per-anatomy bar] [Loss total]   [Loss components]
  Row 2: [UMAP by anatomy] [UMAP by dataset] [Cross-branch] [Gram loss]
  Row 3: [EF scatter]  [EF Bland-Altman]  [Prompt placement] [SAM iters]

Any panel that cannot be computed (missing data) is replaced by a
grey "N/A" panel so the layout remains consistent.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from viz.core import (
    to_numpy_image, save_figure, _get_plt,
    extract_patch_features, extract_attentions,
)
from viz import (
    attention as attn_viz,
    prototypes as proto_viz,
    features   as feat_viz,
    segmentation as seg_viz,
    training   as train_viz,
    regression as reg_viz,
    probes     as probe_viz,
)


def _na_panel(ax, label: str = "N/A"):
    """Fill an axis with a grey N/A placeholder."""
    ax.set_facecolor("#f0f0f0")
    ax.text(0.5, 0.5, label, ha="center", va="center",
            fontsize=10, color="#888888", transform=ax.transAxes)
    ax.axis("off")


def build_dashboard(
    # Required
    img_branch,
    device:             str,

    # Optional — provide whatever is available; missing panels become N/A
    image_rgb:          Optional[torch.Tensor] = None,    # (3, H, W)
    pred_mask:          Optional[np.ndarray]   = None,    # (H, W)
    gt_mask:            Optional[np.ndarray]   = None,    # (H, W)
    proto_head                                  = None,
    metrics_jsonl:      Optional[str]           = None,
    probe_steps:        Optional[list[int]]     = None,
    probe_macro_aucs:   Optional[list[float]]   = None,
    probe_per_anatomy:  Optional[dict]          = None,
    features_array:     Optional[np.ndarray]    = None,   # (N, D) CLS tokens
    anatomy_labels:     Optional[list[str]]     = None,
    ef_pred:            Optional[np.ndarray]    = None,
    ef_true:            Optional[np.ndarray]    = None,
    prompt_points:      Optional[np.ndarray]    = None,   # (N, 2)
    sam_masks:          Optional[list]          = None,
    global_step:        int  = 0,
    title_suffix:       str  = "",
    save_path:          Optional[str] = None,
    dpi:                int  = 120,
):
    """
    Build a comprehensive 4×4 summary dashboard figure.

    Returns the matplotlib Figure.  All panels gracefully degrade to N/A
    if their data is not provided.
    """
    plt = _get_plt()

    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    title = f"Ultatron Training Dashboard  —  Step {global_step:,}"
    if title_suffix:
        title += f"  |  {title_suffix}"
    fig.suptitle(title, fontsize=14, y=1.005)

    # ── Row 0: Image-level analysis ───────────────────────────────────────────
    if image_rgb is not None:
        img_np = to_numpy_image(image_rgb)

        # [0,0] Raw image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Input image", fontsize=10)
        axes[0, 0].axis("off")

        # [0,1] Mean CLS attention
        attn_data = attn_viz.extract_cls_attention(img_branch, image_rgb, device)
        if attn_data:
            from viz.core import overlay_heatmap
            vis = overlay_heatmap(img_np, attn_data["mean"], alpha=0.6, cmap="inferno")
            axes[0, 1].imshow(vis)
            axes[0, 1].set_title("Mean CLS attention", fontsize=10)
        else:
            _na_panel(axes[0, 1], "Attention\nN/A")
        axes[0, 1].axis("off")

        # [0,2] Prototype map
        if proto_head is not None:
            sub_fig = proto_viz.plot_prototype_map(
                img_branch, proto_head, image_rgb, device
            )
            axes[0, 2].imshow(sub_fig.axes[1].get_images()[0].get_array())
            plt.close(sub_fig)
            axes[0, 2].set_title("Prototype map", fontsize=10)
        else:
            _na_panel(axes[0, 2], "Prototype map\n(no head provided)")
        axes[0, 2].axis("off")

        # [0,3] Segmentation overlay
        if pred_mask is not None:
            vis = seg_viz.plot_segmentation_overlay(
                img_np, pred_mask, gt_mask, threshold=0.5
            )
            axes[0, 3].imshow(vis.axes[2 if gt_mask is None else 2].get_images()[0].get_array())
            plt.close(vis)
            axes[0, 3].set_title("Segmentation", fontsize=10)
        else:
            _na_panel(axes[0, 3], "Segmentation\n(no prediction)")
        axes[0, 3].axis("off")
    else:
        for col in range(4):
            _na_panel(axes[0, col], "Image\nnot provided")

    # ── Row 1: Training metrics ────────────────────────────────────────────────
    # [1,0] AUC curve
    if probe_steps and probe_macro_aucs:
        axes[1, 0].plot(probe_steps, probe_macro_aucs, "ko-", linewidth=2, markersize=5)
        axes[1, 0].axhline(0.5, color="grey", linestyle=":", alpha=0.5)
        axes[1, 0].set_ylim(0.45, 1.02)
        axes[1, 0].set_title("Linear probe AUC", fontsize=10)
        axes[1, 0].set_xlabel("Step", fontsize=8)
        axes[1, 0].grid(True, alpha=0.25)
    else:
        _na_panel(axes[1, 0], "Probe AUC\ncurve N/A")

    # [1,1] Per-anatomy AUC bar
    if probe_per_anatomy:
        latest = {k: v[-1] for k, v in probe_per_anatomy.items() if v}
        families = sorted(latest, key=lambda x: latest[x], reverse=True)
        aucs_   = [latest[f] for f in families]
        y_      = np.arange(len(families))
        axes[1, 1].barh(y_, aucs_,
                        color=["tomato" if a < 0.7 else "steelblue" for a in aucs_],
                        alpha=0.8)
        axes[1, 1].set_yticks(y_)
        axes[1, 1].set_yticklabels(families, fontsize=7)
        axes[1, 1].axvline(0.7, color="orange", linestyle="--", alpha=0.7)
        axes[1, 1].set_xlim(0.4, 1.0)
        axes[1, 1].invert_yaxis()
        axes[1, 1].set_title("Per-anatomy AUC", fontsize=10)
    else:
        _na_panel(axes[1, 1], "Per-anatomy AUC\nN/A")

    # [1,2] & [1,3] Loss curves (total and gram)
    if metrics_jsonl:
        try:
            rows   = train_viz.load_metrics(metrics_jsonl)
            steps_ = np.array([r["step"] for r in rows])
            loss_  = np.array([r.get("loss", np.nan) for r in rows])
            valid  = ~np.isnan(loss_)
            axes[1, 2].plot(steps_[valid], loss_[valid],
                            color="steelblue", alpha=0.3, linewidth=0.7)
            if valid.sum() > 50:
                sm = train_viz._smooth(loss_[valid], 100)
                axes[1, 2].plot(steps_[valid][99:], sm, color="steelblue", linewidth=1.5)
            axes[1, 2].set_title("Total loss", fontsize=10)
            axes[1, 2].set_xlabel("Step", fontsize=8)
            axes[1, 2].grid(True, alpha=0.25)

            gram_ = np.array([r.get("loss_gram", np.nan) for r in rows])
            valid_g = ~np.isnan(gram_) & (gram_ > 0)
            if valid_g.any():
                axes[1, 3].plot(steps_[valid_g], gram_[valid_g],
                                color="purple", alpha=0.4, linewidth=0.7)
                axes[1, 3].set_title("Gram anchoring loss", fontsize=10)
                axes[1, 3].set_xlabel("Step", fontsize=8)
                axes[1, 3].grid(True, alpha=0.25)
            else:
                _na_panel(axes[1, 3], "Gram loss\nnot yet active")
        except Exception:
            _na_panel(axes[1, 2], "Loss data\nread error")
            _na_panel(axes[1, 3], "Loss data\nread error")
    else:
        _na_panel(axes[1, 2], "Loss curves\n(no jsonl)")
        _na_panel(axes[1, 3], "Gram loss\n(no jsonl)")

    # ── Row 2: Feature space ──────────────────────────────────────────────────
    if features_array is not None and anatomy_labels is not None:
        from sklearn.manifold import TSNE
        n_pts = min(len(features_array), 3_000)
        idx_  = np.random.choice(len(features_array), n_pts, replace=False)
        feats_sub = features_array[idx_]
        anat_sub  = [anatomy_labels[i] for i in idx_]
        try:
            coords = TSNE(n_components=2, random_state=42,
                          perplexity=min(30, n_pts - 1)).fit_transform(feats_sub)
            fams   = sorted(set(anat_sub))
            cmap_  = plt.cm.tab20
            for ci, fam in enumerate(fams):
                mask_ = np.array(anat_sub) == fam
                axes[2, 0].scatter(coords[mask_, 0], coords[mask_, 1],
                                   s=5, alpha=0.5, color=cmap_(ci / max(len(fams), 1)),
                                   label=fam, linewidths=0)
            axes[2, 0].set_title("Feature t-SNE (anatomy)", fontsize=10)
            axes[2, 0].axis("off")
            if len(fams) <= 15:
                axes[2, 0].legend(markerscale=2, fontsize=6, loc="best",
                                  ncol=max(1, len(fams) // 8))
        except Exception:
            _na_panel(axes[2, 0], "t-SNE\nfailed")
    else:
        _na_panel(axes[2, 0], "Feature UMAP\n(no features)")

    # Remaining row 2 panels: placeholder for dataset UMAP and alignment
    _na_panel(axes[2, 1], "Dataset UMAP\n(run feat_viz.plot_feature_umap)")
    _na_panel(axes[2, 2], "Cross-branch\nalignment")
    _na_panel(axes[2, 3], "Similarity\nmatrix")

    # ── Row 3: Regression and SAM ─────────────────────────────────────────────
    if ef_pred is not None and ef_true is not None:
        axes[3, 0].scatter(ef_true, ef_pred, s=8, alpha=0.4,
                           color="steelblue", linewidths=0)
        lo_ = min(ef_true.min(), ef_pred.min()) - 2
        hi_ = max(ef_true.max(), ef_pred.max()) + 2
        axes[3, 0].plot([lo_, hi_], [lo_, hi_], "k--", linewidth=1, alpha=0.5)
        from eval.metrics import mae as mae_fn
        axes[3, 0].set_title(f"EF prediction  MAE={mae_fn(ef_pred, ef_true):.2f}%",
                              fontsize=10)
        axes[3, 0].set_xlabel("True EF (%)"); axes[3, 0].set_ylabel("Predicted EF (%)")
        axes[3, 0].grid(True, alpha=0.25)

        # Bland-Altman mini
        means_ = (ef_pred + ef_true) / 2
        diffs_ = ef_pred - ef_true
        axes[3, 1].scatter(means_, diffs_, s=6, alpha=0.4, linewidths=0)
        axes[3, 1].axhline(diffs_.mean(), color="red", linewidth=1.2)
        axes[3, 1].axhline(diffs_.mean() + 1.96 * diffs_.std(), color="tomato",
                           linestyle="--", linewidth=0.9)
        axes[3, 1].axhline(diffs_.mean() - 1.96 * diffs_.std(), color="tomato",
                           linestyle="--", linewidth=0.9)
        axes[3, 1].set_title("Bland-Altman (EF)", fontsize=10)
        axes[3, 1].grid(True, alpha=0.25)
    else:
        _na_panel(axes[3, 0], "EF scatter\n(no EchoNet data)")
        _na_panel(axes[3, 1], "Bland-Altman\n(no EchoNet data)")

    _na_panel(axes[3, 2], "SAM prompts\n(run sam_prompts.plot_prompt_placement)")
    _na_panel(axes[3, 3], "SAM refinement\n(run sam_prompts.plot_iterative_refinement)")

    fig.tight_layout(rect=[0, 0, 1, 1.0])

    if save_path:
        save_figure(fig, save_path, dpi=dpi)
    return fig
