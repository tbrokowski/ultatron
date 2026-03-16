"""
viz/training.py  ·  Training dynamics visualisation
=========================================================

Reads metrics.jsonl written by MetricLogger and produces diagnostic plots.

Plots:
  1. Loss curves    — all loss components over training steps
  2. ALP hardness   — average per-anatomy hardness over time
  3. Gram coherence — Gram loss value over steps (should stay low)
  4. Resolution     — max_global_crop_px schedule (step function)
  5. LR schedule    — learning rate over steps
  6. EMA drift      — cosine distance between student and EMA teacher over time
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from viz.core import save_figure, _get_plt


def load_metrics(jsonl_path: str) -> list[dict]:
    """Load all rows from a metrics.jsonl file."""
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_loss_curves(
    jsonl_path: str,
    components: Optional[list[str]] = None,
    smooth:     int   = 100,
    save_path:  Optional[str] = None,
):
    """
    Plot all loss components from metrics.jsonl.

    components: list of keys to plot (default: all keys starting with 'loss_')
    """
    plt = _get_plt()

    rows  = load_metrics(jsonl_path)
    steps = np.array([r["step"] for r in rows])

    if components is None:
        components = sorted(
            k for k in rows[0].keys() if k.startswith("loss_") and k != "loss"
        )

    # Also include total loss
    plot_keys = ["loss"] + [c for c in components if c != "loss"]
    plot_keys = [k for k in plot_keys if any(k in r for r in rows)]

    n    = len(plot_keys)
    cols = min(3, n)
    rows_fig = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_fig, cols,
                              figsize=(cols * 5, rows_fig * 3.5))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    colours = plt.cm.tab10(np.linspace(0, 1, max(n, 2)))

    for i, key in enumerate(plot_keys):
        ax  = axes_flat[i]
        vals = np.array([r.get(key, np.nan) for r in rows])
        valid = ~np.isnan(vals)

        ax.plot(steps[valid], vals[valid], alpha=0.25, color=colours[i], linewidth=0.8)
        if smooth > 1 and valid.sum() > smooth:
            s_vals = _smooth(vals[valid], smooth)
            s_steps = steps[valid][smooth - 1:]
            ax.plot(s_steps, s_vals, color=colours[i], linewidth=1.5)

        # Phase boundaries (shaded regions)
        all_phases = [r.get("phase", 0) for r in rows]
        for ph_start, ph_end, ph_num in _phase_spans(steps, all_phases):
            ax.axvspan(ph_start, ph_end, alpha=0.06,
                       color=f"C{ph_num - 1}", label=f"Phase {ph_num}" if i == 0 else "")

        ax.set_title(key, fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.2)
        if i == 0 and any(p > 0 for p in all_phases):
            ax.legend(fontsize=7)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Training loss curves", fontsize=12)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def _phase_spans(steps, phases):
    """Extract (start_step, end_step, phase_number) spans."""
    spans = []
    if not phases:
        return spans
    cur_ph = phases[0]
    start  = steps[0]
    for i, ph in enumerate(phases[1:], 1):
        if ph != cur_ph:
            spans.append((start, steps[i - 1], cur_ph))
            cur_ph = ph
            start  = steps[i]
    spans.append((start, steps[-1], cur_ph))
    return spans


def plot_lr_and_resolution(
    jsonl_path: str,
    save_path:  Optional[str] = None,
):
    """Learning rate and resolution curriculum on the same time axis."""
    plt = _get_plt()

    rows  = load_metrics(jsonl_path)
    steps = np.array([r["step"] for r in rows])

    lr_vals  = np.array([r.get("lr", np.nan) for r in rows])
    res_vals = np.array([r.get("max_global_crop_px", np.nan) for r in rows])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    valid_lr = ~np.isnan(lr_vals)
    if valid_lr.any():
        ax1.plot(steps[valid_lr], lr_vals[valid_lr], color="steelblue", linewidth=1.2)
        ax1.set_ylabel("Learning rate")
        ax1.set_title("LR schedule")
        ax1.grid(True, alpha=0.25)

    valid_res = ~np.isnan(res_vals)
    if valid_res.any():
        ax2.step(steps[valid_res], res_vals[valid_res], color="darkorange", linewidth=1.5)
        ax2.set_ylabel("max_global_crop_px")
        ax2.set_xlabel("Step")
        ax2.set_title("Resolution curriculum")
        ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_gram_loss(
    jsonl_path:  str,
    gram_start:  Optional[int] = None,
    save_path:   Optional[str] = None,
):
    """Gram anchoring loss over training. Should activate at gram_start and stay low."""
    plt = _get_plt()

    rows  = load_metrics(jsonl_path)
    steps = np.array([r["step"] for r in rows])
    gram  = np.array([r.get("loss_gram", np.nan) for r in rows])
    valid = ~np.isnan(gram) & (gram > 0)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(steps[valid], gram[valid], color="purple", alpha=0.4, linewidth=0.7)
    if valid.sum() > 50:
        s = _smooth(gram[valid], 50)
        ax.plot(steps[valid][49:], s, color="purple", linewidth=1.5, label="Gram loss (smoothed)")

    if gram_start:
        ax.axvline(gram_start, color="red", linestyle="--", alpha=0.6,
                   label=f"Activation (step {gram_start})")

    ax.set_title("Gram anchoring loss over training", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("L_gram")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig
