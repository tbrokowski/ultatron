"""
finetune/report.py  ·  Multi-backbone comparison report generator
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finetune.experiment_registry import (
    DASHBOARD_DIR,
    EXPERIMENTS,
    REFERENCE_COLOR,
    REFERENCE_SCORES,
    STUDENT_COLOR,
    chart_metrics_for,
    default_head_for,
    is_student_backbone,
    primary_metric_for,
    title_for,
)
from finetune.results_collector import (
    collect_from_dir,
    collect_sweep,
    merge_trees,
    write_coverage_report,
)

log = logging.getLogger(__name__)


def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        log.warning("matplotlib not available — skipping charts (%s)", exc)
        return None


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _load_results(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Skipping unreadable results: %s (%s)", path, exc)
        return None


def collect_results(eval_results_dir: Path) -> dict[str, Any]:
    """Walk *eval_results_dir* and collect all results.json files."""
    return collect_from_dir(eval_results_dir)


def _inject_reference_scores(tree: dict[str, Any]) -> dict[str, Any]:
    for experiment, refs in REFERENCE_SCORES.items():
        head_type = default_head_for(experiment)
        for backbone, metrics in refs.items():
            tree.setdefault(backbone, {}).setdefault(experiment, {})[head_type] = {
                **metrics,
                "_path": "reference",
            }
    return tree


def _format_cell(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    if val is None:
        return "—"
    return str(val)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown_report(tree: dict[str, Any], out_path: Path) -> None:
    experiments: set[str] = set()
    for backbone_data in tree.values():
        experiments.update(backbone_data.keys())

    sections: list[str] = ["# Finetune Backbone Comparison Report\n"]

    for experiment in sorted(experiments):
        sections.append(f"## {title_for(experiment)}\n")

        head_types: set[str] = set()
        backbones: set[str] = set()
        for backbone, exp_data in tree.items():
            if experiment not in exp_data:
                continue
            backbones.add(backbone)
            head_types.update(exp_data[experiment].keys())

        for head_type in sorted(head_types):
            sections.append(f"### head_type: `{head_type}`\n")

            chart_specs = chart_metrics_for(experiment)
            metric_keys = [key for key, _, _ in chart_specs] if chart_specs else []
            if not metric_keys:
                for backbone in sorted(backbones):
                    entry = tree.get(backbone, {}).get(experiment, {}).get(head_type)
                    if entry:
                        metric_keys = sorted(
                            k for k, v in entry.items()
                            if not k.startswith("_")
                            and isinstance(v, (int, float))
                            and (k.startswith("val_") or k.endswith("_mean")
                                 or k in {"mae", "rmse", "r2", "pearson_r", "cls_accuracy"})
                        )
                        break

            if not metric_keys:
                sections.append("_No results found._\n")
                continue

            headers = ["backbone"] + metric_keys
            rows: list[list[str]] = []
            for backbone in sorted(backbones):
                entry = tree.get(backbone, {}).get(experiment, {}).get(head_type)
                if not entry:
                    continue
                rows.append([backbone] + [_format_cell(entry.get(k)) for k in metric_keys])

            sections.append(_markdown_table(headers, rows) + "\n")

    out_path.write_text("\n".join(sections))
    log.info("Markdown report → %s", out_path)


def _metric_value(entry: dict[str, Any], key: str) -> float | None:
    val = entry.get(key)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if math.isnan(float(val)):
            return None
        return float(val)
    return None


def _collect_metric_scores(
    tree: dict[str, Any],
    experiment: str,
    metric: str,
    head_type: str | None = None,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for backbone, exp_data in tree.items():
        if experiment not in exp_data:
            continue
        heads = exp_data[experiment]
        ht = head_type or default_head_for(experiment)
        if ht not in heads:
            ht = next(iter(heads), None)
        if ht is None:
            continue
        val = _metric_value(heads[ht], metric)
        if val is not None:
            scores[backbone] = val
    return scores


def _bar_colors(names: list[str], experiment: str, plt) -> list:
    ref_set = set(REFERENCE_SCORES.get(experiment, {}).keys())
    student_idx = 0
    baseline_idx = 0
    n_baselines = sum(1 for n in names if n not in ref_set and not is_student_backbone(n))
    colors = []
    for name in names:
        if name in ref_set:
            colors.append(REFERENCE_COLOR)
        elif is_student_backbone(name):
            colors.append(STUDENT_COLOR)
        else:
            colors.append(plt.cm.viridis(baseline_idx / max(n_baselines - 1, 1)))
            baseline_idx += 1
    return colors


def write_comparison_charts(tree: dict[str, Any], out_dir: Path) -> list[Path]:
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    plt = _get_plt()
    if plt is None:
        return written

    experiments = sorted(
        exp for exp in EXPERIMENTS
        if any(exp in bd for bd in tree.values())
    )

    for experiment in experiments:
        metric_specs = chart_metrics_for(experiment)
        if not metric_specs:
            continue

        n_metrics = len(metric_specs)
        ncols = min(2, n_metrics)
        nrows = math.ceil(n_metrics / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, max(4, 0.45 * len(tree) + 2) * nrows),
            squeeze=False,
        )
        fig.suptitle(f"Backbone comparison — {title_for(experiment)}", fontsize=14, fontweight="bold")

        for idx, (metric_key, label, lower_is_better) in enumerate(metric_specs):
            ax = axes[idx // ncols][idx % ncols]
            scores = _collect_metric_scores(tree, experiment, metric_key)
            if not scores:
                ax.set_visible(False)
                continue

            ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=not lower_is_better)
            names = [k for k, _ in ordered]
            values = [v for _, v in ordered]
            colors = _bar_colors(names, experiment, plt)

            y_pos = range(len(names))
            ax.barh(list(y_pos), values, color=colors, edgecolor="white", height=0.7)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(names, fontsize=9)
            ax.set_xlabel(label)
            ax.set_title(label)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.25)
            for y, v in zip(y_pos, values):
                ax.text(v, y, f" {v:.4f}", va="center", fontsize=8)

        for idx in range(n_metrics, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = charts_dir / f"{experiment}.png"
        _save_figure(fig, out_path)
        plt.close(fig)
        written.append(out_path)
        log.info("Chart → %s", out_path)

    return written


def write_summary_chart(tree: dict[str, Any], out_path: Path) -> None:
    """Cross-dataset heatmap: backbones × experiments (primary metric)."""
    import numpy as np

    experiments = [e for e in EXPERIMENTS if any(e in bd for bd in tree.values())]
    if not experiments:
        log.warning("No experiments with data — skipping summary chart")
        return

    plt = _get_plt()
    if plt is None:
        return

    backbones = sorted(
        b for b in tree
        if b != "reference" and any(exp in tree[b] for exp in experiments)
    )
    if not backbones:
        return

    matrix = []
    for backbone in backbones:
        row = []
        for exp in experiments:
            metric = primary_metric_for(exp)
            scores = _collect_metric_scores(tree, exp, metric)
            row.append(scores.get(backbone))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(max(8, len(experiments) * 1.2), max(4, len(backbones) * 0.4 + 2)))

    data = np.array([[v if v is not None else float("nan") for v in row] for row in matrix])
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=np.nanmin(data), vmax=np.nanmax(data))

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(backbones)))
    ax.set_yticklabels(backbones, fontsize=9)
    ax.set_title("Primary metric across all finetune benchmarks", fontweight="bold")

    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            if val is not None:
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)
    log.info("Summary chart → %s", out_path)


def generate_comparison_report(
    eval_results_dir: Path,
    *,
    from_logs: bool = False,
) -> dict[str, Any]:
    eval_results_dir = Path(eval_results_dir)
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    tree = collect_sweep(eval_results_dir, from_logs=from_logs)
    tree = _inject_reference_scores(tree)

    json_path = eval_results_dir / "comparison_report.json"
    json_path.write_text(json.dumps(tree, indent=2))
    log.info("JSON report → %s", json_path)

    write_markdown_report(tree, eval_results_dir / "comparison_report.md")
    write_comparison_charts(tree, eval_results_dir)
    return tree


def generate_dashboard(tree: dict[str, Any], out_dir: Path | None = None) -> dict[str, Any]:
    """Write unified cross-sweep dashboard."""
    out_dir = Path(out_dir or DASHBOARD_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = _inject_reference_scores(tree)
    (out_dir / "comparison_report.json").write_text(json.dumps(tree, indent=2))
    write_markdown_report(tree, out_dir / "comparison_report.md")
    write_coverage_report(tree, out_dir / "coverage.md")
    write_comparison_charts(tree, out_dir)
    write_summary_chart(tree, out_dir / "summary_primary_metric.png")

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments_with_data": sorted(
            e for e in EXPERIMENTS if any(e in bd for bd in tree.values())
        ),
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    return tree
