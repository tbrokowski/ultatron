"""Collect finetune results from results.json files and Slurm logs."""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any

from finetune.experiment_registry import (
    EXPERIMENTS,
    LOG_ROOT,
    RESULTS_ROOT,
    SWEEPS,
    all_sweep_dirs,
    normalize_experiment,
)

log = logging.getLogger(__name__)

_BACKBONE_RE = re.compile(r"INFO Backbone: (\S+)")
_BRACKET_DICT_RE = re.compile(r"INFO \[([\w_]+)/(\w+)\]\s+(\{.+?\})(?:\s*$|\s)")
_DONE_METRICS_RE = re.compile(r"Done\.\s+((?:\w+=\S+\s*)+)")
_KV_RE = re.compile(r"(\w+)=([\d.eE+-]+)")

# Match longest log prefix first (busi_multitask before busi).
_SPECS_BY_LOG_PREFIX = sorted(EXPERIMENTS.values(), key=lambda s: len(s.log_prefix), reverse=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Skipping unreadable results: %s (%s)", path, exc)
        return None


def _parse_results_path(results_path: Path, sweep_dir: Path) -> tuple[str, str, str] | None:
    rel = results_path.relative_to(sweep_dir)
    parts = rel.parts
    if len(parts) < 4:
        return None
    if len(parts) >= 5 and parts[1] == "camus":
        return parts[0], f"camus_{parts[2]}", parts[3]
    return parts[0], parts[1], parts[2]


def collect_from_dir(sweep_dir: Path) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    if not sweep_dir.exists():
        return tree

    for results_path in sorted(sweep_dir.rglob("results.json")):
        parsed = _parse_results_path(results_path, sweep_dir)
        if parsed is None:
            continue
        backbone, experiment, head_type = parsed
        exp_key = normalize_experiment(experiment) or experiment
        metrics = _load_json(results_path)
        if metrics is None:
            continue
        tree.setdefault(backbone, {}).setdefault(exp_key, {})[head_type] = {
            **metrics,
            "_path": str(results_path.parent),
            "_source": "results.json",
        }
    return tree


def _parse_done_metrics(line: str) -> dict[str, float]:
    m = _DONE_METRICS_RE.search(line)
    if not m:
        return {}
    out: dict[str, float] = {}
    for key, val in _KV_RE.findall(m.group(1)):
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def _infer_experiment(metrics: dict[str, Any], bracket_exp: str) -> str:
    for field in ("experiment", "benchmark"):
        val = metrics.get(field)
        if isinstance(val, str):
            key = normalize_experiment(val)
            if key:
                return key
    key = normalize_experiment(bracket_exp)
    return key or bracket_exp


def parse_log_file(log_path: Path) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    current_backbone: str | None = None

    try:
        text = log_path.read_text(errors="replace")
    except OSError as exc:
        log.warning("Cannot read log %s: %s", log_path, exc)
        return tree

    pending_done: dict[str, float] = {}

    for line in text.splitlines():
        bm = _BACKBONE_RE.search(line)
        if bm:
            current_backbone = bm.group(1)
            pending_done = {}
            continue

        if current_backbone is None:
            continue

        if _DONE_METRICS_RE.search(line):
            pending_done = _parse_done_metrics(line)

        m = _BRACKET_DICT_RE.search(line)
        if not m:
            continue

        bracket_exp, head_type, dict_str = m.group(1), m.group(2), m.group(3)
        try:
            metrics = ast.literal_eval(dict_str)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(metrics, dict):
            continue

        metrics = dict(metrics)
        if pending_done:
            for k, v in pending_done.items():
                metrics.setdefault(k, v)
            pending_done = {}

        exp_key = normalize_experiment(_infer_experiment(metrics, bracket_exp)) or bracket_exp
        metrics["_path"] = str(log_path)
        metrics["_source"] = "log"
        metrics["_log_file"] = log_path.name

        tree.setdefault(current_backbone, {}).setdefault(exp_key, {})[head_type] = metrics

    return tree


def _log_job_id(path: Path) -> int:
    m = re.search(r"_(\d+)\.(?:err|out)$", path.name)
    return int(m.group(1)) if m else 0


def _best_log_files() -> dict[str, Path]:
    if not LOG_ROOT.exists():
        return {}

    by_exp: dict[str, Path] = {}
    for log_path in sorted(LOG_ROOT.glob("ultatron_ft_*.err")):
        for spec in _SPECS_BY_LOG_PREFIX:
            if not log_path.name.startswith(spec.log_prefix):
                continue
            prev = by_exp.get(spec.key)
            if prev is None:
                by_exp[spec.key] = log_path
            elif _log_job_id(log_path) > _log_job_id(prev):
                by_exp[spec.key] = log_path
            elif (
                _log_job_id(log_path) == _log_job_id(prev)
                and log_path.stat().st_mtime > prev.stat().st_mtime
            ):
                by_exp[spec.key] = log_path
            break
    return by_exp


def collect_from_logs() -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for exp_key, log_path in _best_log_files().items():
        log.info("Parsing log for %s: %s", exp_key, log_path.name)
        partial = parse_log_file(log_path)
        for backbone, exp_data in partial.items():
            for experiment, heads in exp_data.items():
                for head_type, metrics in heads.items():
                    merged.setdefault(backbone, {}).setdefault(experiment, {})[head_type] = metrics
    return merged


def merge_trees(*trees: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for tree in trees:
        for backbone, exp_data in tree.items():
            for experiment, heads in exp_data.items():
                for head_type, metrics in heads.items():
                    merged.setdefault(backbone, {}).setdefault(experiment, {})[head_type] = metrics
    return merged


def collect_sweep(
    sweep_dir: Path,
    *,
    from_logs: bool = False,
    logs_only: bool = False,
) -> dict[str, Any]:
    json_tree = {} if logs_only else collect_from_dir(sweep_dir)
    if not from_logs and not logs_only:
        return json_tree

    sweep_name = next((k for k, v in SWEEPS.items() if v == sweep_dir), None)
    log_tree: dict[str, Any] = {}
    for spec in EXPERIMENTS.values():
        if sweep_name and spec.sweep != sweep_name:
            continue
        logs = _best_log_files()
        if spec.key not in logs:
            continue
        log_tree = merge_trees(log_tree, parse_log_file(logs[spec.key]))

    if logs_only:
        return log_tree
    return merge_trees(log_tree, json_tree)


def collect_all_sweeps(*, from_logs: bool = False, logs_only: bool = False) -> dict[str, Any]:
    trees = [collect_sweep(sd, from_logs=from_logs, logs_only=logs_only) for sd in all_sweep_dirs()]
    if from_logs or logs_only:
        trees.append(collect_from_logs())
    return merge_trees(*trees)


def write_backfilled_results(tree: dict[str, Any]) -> int:
    written = 0
    for backbone, exp_data in tree.items():
        if backbone == "reference":
            continue
        for experiment, heads in exp_data.items():
            spec = EXPERIMENTS.get(experiment)
            out_sweep = SWEEPS[spec.sweep] if spec else RESULTS_ROOT / "representative"
            for head_type, metrics in heads.items():
                out_metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}
                out_dir = out_sweep / backbone / experiment / head_type
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "results.json"
                out_path.write_text(json.dumps(out_metrics, indent=2))
                written += 1
                log.info("Backfilled → %s", out_path)
    return written


def write_coverage_report(tree: dict[str, Any], out_path: Path) -> None:
    lines = ["# Finetune Results Coverage\n"]
    for exp_key, spec in sorted(EXPERIMENTS.items()):
        backbones = sorted(b for b, ed in tree.items() if exp_key in ed and b != "reference")
        status = "OK" if backbones else "MISSING"
        lines.append(f"- **{exp_key}** ({spec.sweep}): {status}")
        if backbones:
            lines.append(f"  - backbones: {', '.join(backbones)}")
    out_path.write_text("\n".join(lines) + "\n")
