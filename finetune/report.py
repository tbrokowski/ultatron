"""
finetune/report.py  ·  Multi-backbone comparison report generator
====================================================================

Reads per-run ``results.json`` files under an eval_results tree and writes:
  - comparison_report.json  — nested backbone → experiment → head_type → metrics
  - comparison_report.md    — markdown tables per experiment
  - charts/<experiment>.png — bar charts comparing backbones per dataset
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Primary metric to highlight per experiment (first match wins)
_PRIMARY_METRICS = (
    "dice_lv_endo_mean", "dice_lv_epi_mean", "dice_la_mean", "dice_macro_fg_mean",
    "dice_tumor_mean", "cls_accuracy", "val_dice", "val_auc", "val_auc_tb", "val_mae", "val_acc", "val_loss",
)

# Metrics plotted per dataset (test-set results.json keys).
# Tuple: (metric_key, axis_label, lower_is_better)
_EXPERIMENT_CHART_METRICS: dict[str, list[tuple[str, str, bool]]] = {
    "busi": [
        ("dice_tumor_mean", "Tumour Dice", False),
        ("dice_mean",       "Dice (all)",  False),
        ("iou_tumor_mean",  "Tumour IoU",  False),
        ("precision_mean",  "Precision",   False),
        ("recall_mean",     "Recall",      False),
    ],
    "busi_multitask": [
        ("dice_tumor_mean", "Tumour Dice", False),
        ("cls_accuracy",    "Cls accuracy", False),
        ("iou_tumor_mean",  "Tumour IoU",  False),
        ("dice_mean",       "Dice (all)",  False),
    ],
    "camus": [
        ("dice_lv_epi_mean",  "LVEpi Dice",  False),
        ("dice_lv_epi_ed",    "LVEpi ED",    False),
        ("dice_lv_epi_es",    "LVEpi ES",    False),
        ("hd95_mean",         "HD95",        True),
    ],
    "camus_lv_endo": [
        ("dice_lv_endo_mean", "LVEndo Dice", False),
        ("dice_lv_endo_ed",   "LVEndo ED",   False),
        ("dice_lv_endo_es",   "LVEndo ES",   False),
        ("hd95_mean",         "HD95",        True),
    ],
    "camus_lv_epi": [
        ("dice_lv_epi_mean",  "LVEpi Dice",  False),
        ("dice_lv_epi_ed",    "LVEpi ED",    False),
        ("dice_lv_epi_es",    "LVEpi ES",    False),
        ("hd95_mean",         "HD95",        True),
    ],
    "camus_la": [
        ("dice_la_mean",      "LA Dice",     False),
        ("dice_la_ed",        "LA ED",       False),
        ("dice_la_es",        "LA ES",       False),
        ("hd95_mean",         "HD95",        True),
    ],
    "camus_multiclass": [
        ("dice_macro_fg_mean", "Macro FG Dice", False),
        ("dice_class_1_mean",  "Cavity Dice",   False),
        ("dice_class_2_mean",  "Myo Dice",      False),
        ("dice_class_3_mean",  "LA Dice",       False),
    ],
    "echonet": [
        ("mae",       "MAE",   True),
        ("rmse",      "RMSE",  True),
        ("r2",        "R²",    False),
        ("pearson_r", "Pearson r", False),
    ],
    "lus": [
        ("val_auc_tb",        "AUC TB",        False),
        ("val_auc_pneumonia", "AUC Pneumonia", False),
        ("val_auc_covid",     "AUC COVID",     False),
        ("val_auc_macro",     "AUC Macro",     False),
    ],
    "lus_video": [
        ("val_auc_a_line",              "AUC A-line",              False),
        ("val_auc_b_line",              "AUC B-line",              False),
        ("val_auc_confluent_b_line",    "AUC Confluent B-line",    False),
        ("val_auc_pleural_effusion",    "AUC Pleural effusion",    False),
        ("val_auc_large_consolidation", "AUC Large consolidation", False),
        ("val_auc_small_consolidation", "AUC Small consolidation", False),
        ("val_auc_pneumothorax",        "AUC Pneumothorax",        False),
        ("val_auc_macro",               "AUC Macro",               False),
    ],
}

_EXPERIMENT_TITLES: dict[str, str] = {
    "busi":           "BUSI — breast tumour segmentation",
    "busi_multitask": "BUSI — tumour seg + 3-class classification",
    "camus":          "CAMUS — cardiac LV segmentation",
    "camus_lv_endo":  "CAMUS — LVEndo binary",
    "camus_lv_epi":   "CAMUS — LVEpi binary",
    "camus_la":       "CAMUS — LA binary",
    "camus_multiclass": "CAMUS — 4-class multiclass",
    "echonet":   "EchoNet-Dynamic — EF regression",
    "lus":       "LUS — patient-level multilabel (MIL)",
    "lus_video": "LUS — video/image multilabel findings",
}

# Default head_type per experiment (matches comparison_representative.yaml)
_DEFAULT_HEAD_TYPES: dict[str, str] = {
    "lus":       "mil",
    "lus_video": "mlp",
}

# Fixed reference scores (percentages converted to [0, 1]) shown alongside finetune runs.
_EXPERIMENT_REFERENCE_SCORES: dict[str, dict[str, dict[str, float]]] = {
    "busi": {
        "reference": {
            "dice_tumor_mean": 0.7709,
            "dice_mean":       0.7709,
            "iou_tumor_mean":  0.6546,
            "precision_mean":  0.8029,
            "recall_mean":     0.7438,
        },
    },
}


def _load_results(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(f"Skipping unreadable results: {path} ({exc})")
        return None


def _infer_primary_metric(metrics: dict[str, Any]) -> str | None:
    for key in _PRIMARY_METRICS:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return key
    for key, val in metrics.items():
        if key.startswith("val_") and isinstance(val, (int, float)):
            return key
    return None


def collect_results(eval_results_dir: Path) -> dict[str, Any]:
    """
    Walk *eval_results_dir* and collect all ``results.json`` files.

    Expected layout:
        {backbone}/{experiment}/{head_type}/results.json
    """
    tree: dict[str, Any] = {}
    if not eval_results_dir.exists():
        return tree

    for results_path in sorted(eval_results_dir.rglob("results.json")):
        rel = results_path.relative_to(eval_results_dir)
        parts = rel.parts
        if len(parts) < 4:
            log.debug(f"Skipping non-standard path: {rel}")
            continue

        if len(parts) >= 5 and parts[1] == "camus":
            backbone, experiment, head_type = parts[0], f"camus_{parts[2]}", parts[3]
        else:
            backbone, experiment, head_type = parts[0], parts[1], parts[2]
        metrics = _load_results(results_path)
        if metrics is None:
            continue

        tree.setdefault(backbone, {}).setdefault(experiment, {})[head_type] = {
            **metrics,
            "_path": str(results_path.parent),
        }

    return tree


def _inject_reference_scores(tree: dict[str, Any]) -> dict[str, Any]:
    """Merge literature / baseline scores into the results tree for reporting."""
    for experiment, refs in _EXPERIMENT_REFERENCE_SCORES.items():
        head_type = _DEFAULT_HEAD_TYPES.get(experiment, "linear")
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
    """Write human-readable markdown tables grouped by experiment."""
    experiments: set[str] = set()
    for backbone_data in tree.values():
        experiments.update(backbone_data.keys())

    sections: list[str] = ["# Finetune Backbone Comparison Report\n"]

    for experiment in sorted(experiments):
        sections.append(f"## {experiment}\n")

        # Collect head types and backbones for this experiment
        head_types: set[str] = set()
        backbones: set[str] = set()
        for backbone, exp_data in tree.items():
            if experiment not in exp_data:
                continue
            backbones.add(backbone)
            head_types.update(exp_data[experiment].keys())

        for head_type in sorted(head_types):
            sections.append(f"### head_type: `{head_type}`\n")

            # Determine metric columns from chart config or first available result
            metric_keys: list[str] = []
            chart_specs = _EXPERIMENT_CHART_METRICS.get(experiment)
            if chart_specs:
                metric_keys = [key for key, _, _ in chart_specs]
            else:
                for backbone in sorted(backbones):
                    entry = tree.get(backbone, {}).get(experiment, {}).get(head_type)
                    if entry:
                        metric_keys = sorted(
                            k for k, v in entry.items()
                            if not k.startswith("_")
                            and isinstance(v, (int, float))
                            and (k.startswith("val_") or k.endswith("_mean")
                                 or k in {"mae", "rmse", "r2", "pearson_r"})
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
    log.info(f"Markdown report → {out_path}")


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
        ht = head_type or _DEFAULT_HEAD_TYPES.get(experiment) or next(iter(heads), None)
        if ht is None or ht not in heads:
            continue
        val = _metric_value(heads[ht], metric)
        if val is not None:
            scores[backbone] = val
    return scores


def write_comparison_charts(tree: dict[str, Any], out_dir: Path) -> list[Path]:
    """
    Write one bar-chart PNG per dataset/experiment comparing all backbones.

    Saved to *out_dir*/charts/<experiment>.png
    """
    from viz.core import save_figure, _get_plt

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    experiments = sorted(
        exp for exp in _EXPERIMENT_CHART_METRICS
        if any(exp in bd for bd in tree.values())
    )

    for experiment in experiments:
        metric_specs = _EXPERIMENT_CHART_METRICS[experiment]
        n_metrics = len(metric_specs)
        if n_metrics == 0:
            continue

        ncols = min(2, n_metrics)
        nrows = math.ceil(n_metrics / ncols)
        plt = _get_plt()
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(7 * ncols, max(4, 0.45 * len(tree) + 2) * nrows),
            squeeze=False,
        )
        title = _EXPERIMENT_TITLES.get(experiment, experiment)
        fig.suptitle(f"Backbone comparison — {title}", fontsize=14, fontweight="bold")

        for idx, (metric_key, label, lower_is_better) in enumerate(metric_specs):
            ax = axes[idx // ncols][idx % ncols]
            scores = _collect_metric_scores(tree, experiment, metric_key)
            if not scores:
                ax.set_visible(False)
                continue

            ordered = sorted(
                scores.items(),
                key=lambda kv: kv[1],
                reverse=not lower_is_better,
            )
            names  = [k for k, _ in ordered]
            values = [v for _, v in ordered]

            colors = []
            for name in names:
                if name in _EXPERIMENT_REFERENCE_SCORES.get(experiment, {}):
                    colors.append("#e67e22")
                else:
                    colors.append(None)
            if all(c is None for c in colors):
                colors = plt.cm.viridis(
                    [i / max(len(values) - 1, 1) for i in range(len(values))]
                )
            else:
                backbone_idx = 0
                n_backbones = sum(
                    1 for n in names
                    if n not in _EXPERIMENT_REFERENCE_SCORES.get(experiment, {})
                )
                filled = []
                for c in colors:
                    if c is None:
                        filled.append(plt.cm.viridis(
                            backbone_idx / max(n_backbones - 1, 1)
                        ))
                        backbone_idx += 1
                    else:
                        filled.append(c)
                colors = filled
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

        # Hide unused subplot cells
        for idx in range(n_metrics, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = charts_dir / f"{experiment}.png"
        save_figure(fig, out_path)
        plt.close(fig)
        written.append(out_path)
        log.info(f"Chart → {out_path}")

    return written


def generate_comparison_report(eval_results_dir: Path) -> dict[str, Any]:
    """
    Aggregate all results under *eval_results_dir* and write JSON + markdown reports.

    Returns the nested results dict.
    """
    eval_results_dir = Path(eval_results_dir)
    tree = collect_results(eval_results_dir)
    tree = _inject_reference_scores(tree)

    json_path = eval_results_dir / "comparison_report.json"
    json_path.write_text(json.dumps(tree, indent=2))
    log.info(f"JSON report → {json_path}")

    md_path = eval_results_dir / "comparison_report.md"
    write_markdown_report(tree, md_path)

    write_comparison_charts(tree, eval_results_dir)

    return tree
