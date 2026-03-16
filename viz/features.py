"""
viz/features.py  ·  Feature space visualisation
=====================================================

CLS token UMAP / t-SNE to verify the backbone is learning transferable,
anatomy-discriminative representations.

Plots:
  1. UMAP by anatomy family  — primary evidence representations are structured
  2. UMAP by dataset          — checks for dataset bias / batch effects
  3. Cross-branch alignment   — image teacher vs video student in shared space
  4. CLS cosine similarity matrix across anatomy families
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from viz.core import save_figure, _get_plt


def _reduce_to_2d(features: np.ndarray, method: str = "umap") -> np.ndarray:
    """Reduce (N, D) features to (N, 2) for plotting."""
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, random_state=42,
                             n_neighbors=30, min_dist=0.1).fit_transform(features)
        except ImportError:
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=42,
                    perplexity=min(30, len(features) - 1)).fit_transform(features)
    raise ValueError(f"Unknown method: {method}")


def plot_feature_umap(
    features:        np.ndarray,          # (N, D)
    colour_labels:   np.ndarray,          # (N,) int
    label_names:     Optional[list[str]] = None,
    title:           str   = "CLS token UMAP",
    method:          str   = "umap",
    max_points:      int   = 10_000,
    save_path:       Optional[str] = None,
):
    """
    2-D embedding of CLS tokens coloured by a categorical label.

    colour_labels: integer class indices
    label_names:   string names for the legend
    """
    plt = _get_plt()

    # Subsample if needed
    if len(features) > max_points:
        idx      = np.random.choice(len(features), max_points, replace=False)
        features = features[idx]
        colour_labels = colour_labels[idx]

    coords = _reduce_to_2d(features, method)

    unique = sorted(set(colour_labels.tolist()))
    cmap   = plt.cm.get_cmap("tab20", max(len(unique), 2))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls_id in enumerate(unique):
        mask = colour_labels == cls_id
        name = label_names[i] if label_names and i < len(label_names) else str(cls_id)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=cmap(i), s=12, alpha=0.6, label=name, linewidths=0)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    if len(unique) <= 20:
        ax.legend(markerscale=2, fontsize=8, loc="best",
                  ncol=max(1, len(unique) // 10))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_anatomy_umap(
    features:        np.ndarray,
    anatomy_families: list[str],
    method:          str = "umap",
    save_path:       Optional[str] = None,
):
    """Convenience wrapper: UMAP coloured by anatomy family."""
    families = sorted(set(anatomy_families))
    fam_to_idx = {f: i for i, f in enumerate(families)}
    labels = np.array([fam_to_idx[f] for f in anatomy_families])
    return plot_feature_umap(
        features, labels, label_names=families,
        title=f"CLS tokens by anatomy family ({method.upper()})",
        method=method, save_path=save_path,
    )


def plot_cross_branch_alignment(
    img_features: np.ndarray,   # (N_img, align_dim)  L2-normalised
    vid_features: np.ndarray,   # (N_vid, align_dim)
    method:       str = "umap",
    save_path:    Optional[str] = None,
):
    """
    Joint UMAP of image teacher and video student features in the shared
    256-D alignment space.  Tight interleaving = good alignment.
    """
    plt = _get_plt()

    all_feats  = np.concatenate([img_features, vid_features], axis=0)
    coords     = _reduce_to_2d(all_feats, method)
    n_img      = len(img_features)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords[:n_img, 0], coords[:n_img, 1],
               color="steelblue", s=10, alpha=0.55, label="Image (teacher)", linewidths=0)
    ax.scatter(coords[n_img:, 0], coords[n_img:, 1],
               color="tomato",    s=10, alpha=0.55, label="Video (student)",  linewidths=0)

    ax.set_title(f"Cross-branch alignment ({method.upper()})\n"
                 f"Tight interleaving = strong alignment", fontsize=11)
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_similarity_matrix(
    features:        np.ndarray,   # (N, D)
    anatomy_families: list[str],
    save_path:       Optional[str] = None,
):
    """
    Mean cosine similarity between anatomy family centroids.
    Shows which anatomy families have similar representations.
    """
    plt = _get_plt()

    import torch.nn.functional as F_
    feat_t  = torch.from_numpy(features).float()
    feat_t  = F_.normalize(feat_t, dim=-1)

    families = sorted(set(anatomy_families))
    K        = len(families)
    centroids = torch.zeros(K, feat_t.shape[-1])

    for i, fam in enumerate(families):
        mask = np.array(anatomy_families) == fam
        centroids[i] = feat_t[mask].mean(0)

    centroids = F_.normalize(centroids, dim=-1)
    sim_matrix = (centroids @ centroids.T).numpy()

    fig, ax = plt.subplots(figsize=(K * 0.7 + 1, K * 0.7 + 1))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(families, fontsize=8)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title("Inter-anatomy cosine similarity\n(diagonal = 1.0)", fontsize=10)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig
