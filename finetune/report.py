"""
finetune/report.py  ·  Multi-backbone comparison report generator
====================================================================

Reads per-run ``results.json`` files under an eval_results tree and writes:
  - comparison_report.json  — nested backbone → experiment → head_type → metrics
  - comparison_report.md    — markdown tables per experiment
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Primary metric to highlight per experiment (first match wins)
_PRIMARY_METRICS = (
    "val_dice", "val_auc", "val_auc_tb", "val_mae", "val_acc", "val_loss",
)


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

        backbone, experiment, head_type = parts[0], parts[1], parts[2]
        metrics = _load_results(results_path)
        if metrics is None:
            continue

        tree.setdefault(backbone, {}).setdefault(experiment, {})[head_type] = {
            **metrics,
            "_path": str(results_path.parent),
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

            # Determine metric columns from first available result
            metric_keys: list[str] = []
            for backbone in sorted(backbones):
                entry = tree.get(backbone, {}).get(experiment, {}).get(head_type)
                if entry:
                    metric_keys = sorted(
                        k for k, v in entry.items()
                        if k.startswith("val_") and isinstance(v, (int, float))
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


def generate_comparison_report(eval_results_dir: Path) -> dict[str, Any]:
    """
    Aggregate all results under *eval_results_dir* and write JSON + markdown reports.

    Returns the nested results dict.
    """
    eval_results_dir = Path(eval_results_dir)
    tree = collect_results(eval_results_dir)

    json_path = eval_results_dir / "comparison_report.json"
    json_path.write_text(json.dumps(tree, indent=2))
    log.info(f"JSON report → {json_path}")

    md_path = eval_results_dir / "comparison_report.md"
    write_markdown_report(tree, md_path)

    return tree
