"""
viz/regression.py  ·  Regression and measurement visualisation
===================================================================

For EchoNet-Dynamic (EF prediction) and HC18 (head circumference).

Plots:
  1. Scatter plot         — predicted vs true with identity line
  2. Bland-Altman         — agreement plot (mean vs difference)
  3. Error distribution   — histogram of absolute errors
  4. Clinical thresholds  — confusion at EF=50% cut-off
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from viz.core import save_figure, _get_plt
from eval.metrics import mae, rmse, pearson_r, r2_score


def plot_regression_scatter(
    pred:       np.ndarray,          # (N,)
    true:       np.ndarray,          # (N,)
    xlabel:     str   = "True EF (%)",
    ylabel:     str   = "Predicted EF (%)",
    title:      str   = "EF Regression",
    units:      str   = "%",
    threshold:  Optional[float] = 50.0,
    save_path:  Optional[str] = None,
):
    """
    Scatter of predicted vs true values with:
      - identity line (perfect prediction)
      - ±5 unit error bands
      - regression line
      - metric annotations (MAE, R², Pearson r)
    """
    plt = _get_plt()

    mae_val    = mae(pred, true)
    r2_val     = r2_score(pred, true)
    pearson_val= pearson_r(pred, true)
    rmse_val   = rmse(pred, true)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Identity line
    lo = min(true.min(), pred.min()) - 2
    hi = max(true.max(), pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.5, label="Identity")

    # ±5% error bands
    band = 5.0
    ax.fill_between([lo, hi], [lo - band, hi - band], [lo + band, hi + band],
                     alpha=0.08, color="steelblue", label=f"±{band}{units}")

    # Clinical threshold lines
    if threshold is not None:
        ax.axvline(threshold, color="tomato", linestyle=":", alpha=0.6,
                   label=f"Clinical threshold ({threshold}{units})")
        ax.axhline(threshold, color="tomato", linestyle=":", alpha=0.6)

    # Scatter
    ax.scatter(true, pred, s=15, alpha=0.4, color="steelblue", linewidths=0)

    # Regression line
    m, b = np.polyfit(true, pred, 1)
    x_fit = np.linspace(lo, hi, 200)
    ax.plot(x_fit, m * x_fit + b, "tomato", linewidth=1.5,
            label=f"Fit: y={m:.2f}x+{b:.1f}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    metrics_text = (f"MAE   = {mae_val:.2f}{units}\n"
                    f"RMSE  = {rmse_val:.2f}{units}\n"
                    f"R²    = {r2_val:.4f}\n"
                    f"r     = {pearson_val:.4f}")
    ax.text(0.03, 0.97, metrics_text, transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_bland_altman(
    pred:       np.ndarray,
    true:       np.ndarray,
    ylabel:     str = "Difference (pred − true) [%]",
    title:      str = "Bland-Altman Agreement",
    save_path:  Optional[str] = None,
):
    """
    Bland-Altman plot: x = (pred+true)/2, y = pred−true.
    Shows limits of agreement (mean±1.96 SD).
    """
    plt = _get_plt()

    means = (pred + true) / 2.0
    diffs = pred - true

    mean_diff = diffs.mean()
    std_diff  = diffs.std()
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, diffs, s=12, alpha=0.4, color="steelblue", linewidths=0)

    ax.axhline(mean_diff,  color="black",   linewidth=1.5,
               label=f"Mean diff = {mean_diff:.2f}")
    ax.axhline(loa_upper,  color="tomato",  linewidth=1.0, linestyle="--",
               label=f"+1.96 SD = {loa_upper:.2f}")
    ax.axhline(loa_lower,  color="tomato",  linewidth=1.0, linestyle="--",
               label=f"−1.96 SD = {loa_lower:.2f}")
    ax.axhline(0,          color="grey",    linewidth=0.7, linestyle=":")

    ax.set_xlabel("Mean of predicted and true", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_error_distribution(
    pred:      np.ndarray,
    true:      np.ndarray,
    title:     str = "Absolute error distribution",
    units:     str = "%",
    save_path: Optional[str] = None,
):
    """Histogram of |pred − true|."""
    plt = _get_plt()

    abs_err = np.abs(pred - true)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(abs_err, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(abs_err.mean(), color="red", linestyle="--",
               label=f"MAE = {abs_err.mean():.2f}{units}")
    ax.axvline(np.percentile(abs_err, 90), color="orange", linestyle="--",
               label=f"90th pct = {np.percentile(abs_err, 90):.2f}{units}")
    ax.set_xlabel(f"Absolute error ({units})")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=11)
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig
