"""
viz/probes.py  ·  Linear probe and SSL quality visualisation
=================================================================

The primary fitness signal for SSL training quality.

Plots:
  1. AUC over training     — macro AUC at multiple checkpoints (learning curve)
  2. Per-anatomy AUC bar   — which anatomies the model handles well vs poorly
  3. Anatomy AUC heatmap   — AUC at each checkpoint for each anatomy
  4. Confusion matrix       — classification errors on the linear probe val set
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from viz.core import save_figure, _get_plt


def plot_probe_learning_curve(
    steps:        list[int],
    macro_aucs:   list[float],
    per_anatomy:  Optional[dict] = None,    # {family: list[auc per step]}
    save_path:    Optional[str] = None,
):
    """
    AUC over training steps — how does SSL quality improve during training?

    steps:       list of global_step values where evaluation was run
    macro_aucs:  macro-average AUC at each step
    per_anatomy: optional dict of per-anatomy AUC lists (same length as steps)
    """
    plt = _get_plt()

    has_anatomy = per_anatomy and len(per_anatomy) > 0

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, macro_aucs, "ko-", linewidth=2, markersize=5,
            label="Macro AUC", zorder=10)

    if has_anatomy:
        cmap = plt.cm.tab20
        for i, (fam, aucs) in enumerate(per_anatomy.items()):
            if len(aucs) == len(steps):
                ax.plot(steps, aucs, alpha=0.5, linewidth=1.0,
                        color=cmap(i / max(len(per_anatomy), 1)), label=fam)

    # Phase shading (rough estimates)
    if len(steps) > 1:
        total = steps[-1]
        ax.axvspan(steps[0], total * 0.1, alpha=0.07, color="blue",   label="Phase 1")
        ax.axvspan(total * 0.1, total * 0.2, alpha=0.07, color="green", label="Phase 2")
        ax.axvspan(total * 0.2, total * 0.9, alpha=0.07, color="orange")

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Training step")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Linear probe AUC over training", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right",
              ncol=max(1, (1 + (len(per_anatomy) if has_anatomy else 0)) // 10))
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_per_anatomy_auc(
    per_anatomy_auc: dict,             # {family: auc_float}
    title:           str = "Per-anatomy AUC (linear probe)",
    save_path:       Optional[str] = None,
):
    """
    Horizontal bar chart of per-anatomy AUC, sorted descending.
    Highlights anatomies below 0.7 (poor) in red.
    """
    plt = _get_plt()

    items  = sorted(per_anatomy_auc.items(), key=lambda x: x[1], reverse=True)
    labels = [it[0] for it in items]
    aucs   = [it[1] for it in items]
    colours= ["tomato" if a < 0.70 else "steelblue" for a in aucs]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.45)))
    y = np.arange(len(labels))
    ax.barh(y, aucs, color=colours, alpha=0.85, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0.5,  color="gray",  linestyle=":",  alpha=0.6, label="Random (0.5)")
    ax.axvline(0.7,  color="orange",linestyle="--", alpha=0.6, label="Threshold (0.7)")
    ax.axvline(0.9,  color="green", linestyle="--", alpha=0.4, label="Good (0.9)")
    macro = float(np.mean(aucs))
    ax.axvline(macro, color="black", linestyle="-", linewidth=1.5,
               label=f"Macro={macro:.3f}")
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("AUC-ROC")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_anatomy_auc_heatmap(
    steps:          list[int],
    anatomy_aucs:   dict,       # {family: list[auc per step]}
    save_path:      Optional[str] = None,
):
    """
    Heatmap: anatomies × training steps, cell value = AUC.
    Reveals which anatomies are learned early vs late.
    """
    plt = _get_plt()

    families = sorted(anatomy_aucs.keys())
    matrix   = np.array([[anatomy_aucs[f][i] if i < len(anatomy_aucs[f]) else np.nan
                          for i in range(len(steps))]
                         for f in families])

    fig, ax = plt.subplots(figsize=(max(8, len(steps) * 1.2), max(4, len(families) * 0.5)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="AUC-ROC")

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=9)
    ax.set_title("Per-anatomy AUC across training checkpoints", fontsize=11)
    ax.set_xlabel("Training step")
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    class_names: Optional[list[str]] = None,
    title:       str = "Linear probe confusion matrix",
    save_path:   Optional[str] = None,
):
    """Normalised confusion matrix from linear probe predictions."""
    plt = _get_plt()

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    K  = cm.shape[0]
    names = class_names or [str(i) for i in range(K)]

    fig, ax = plt.subplots(figsize=(max(5, K * 0.6), max(5, K * 0.6)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall")

    ax.set_xticks(range(K)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)

    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if cm[i, j] > 0.6 else "black")

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig
