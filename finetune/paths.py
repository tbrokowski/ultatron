"""
finetune/paths.py  ·  Finetune results vs checkpoint path layout
==================================================================

Results (metrics, logs, visualizations) live under the repo finetune tree:
  results/finetune/{sweep}/{backbone}/{experiment}/{head_type}/

Task-head weights (.pt) live on Capstor store:
  /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/Finetune/...
"""
from __future__ import annotations

import os
from pathlib import Path

from data.infra.cscs_paths import CSCS_STORE_ROOT

FINETUNE_RESULTS_SUBDIR = "results/finetune/representative"
FINETUNE_CHECKPOINTS_SUBDIR = "checkpoints/Finetune"


def finetune_results_root(repo: Path | None = None) -> Path:
    """Repo-local root for metrics, logs, and visualizations."""
    if repo is None:
        repo = Path(__file__).resolve().parents[1]
    return repo / FINETUNE_RESULTS_SUBDIR


def finetune_checkpoints_root() -> Path:
    """Capstor store root for finetuned task-head weights."""
    if override := os.environ.get("US_FINETUNE_CKPT_DIR"):
        return Path(override)
    return Path(CSCS_STORE_ROOT) / FINETUNE_CHECKPOINTS_SUBDIR


def finetune_run_dirs(
    *,
    repo: Path | None = None,
    backbone: str = "",
    experiment: str = "",
    head_type: str = "dpt",
    results_parent: Path | None = None,
) -> tuple[Path, Path]:
    """
    Return (results_dir, checkpoint_dir) for one finetune run.

    *results_parent* overrides the default finetune results root (comparison YAML).
    """
    rel = Path(backbone) / experiment / head_type if backbone else Path(experiment)
    results_root = results_parent or finetune_results_root(repo)
    return results_root / rel, finetune_checkpoints_root() / rel


def mirror_checkpoint_dir(results_dir: Path, repo: Path | None = None) -> Path:
    """Map a results directory to the matching Capstor checkpoint path.

    When the results directory sits under ``results/finetune/`` (or legacy
    ``dataset_exploration_outputs/finetune``), the relative path is used
    directly.
    """
    repo = repo or Path(__file__).resolve().parents[1]
    results_dir = Path(results_dir)
    if not results_dir.is_absolute():
        results_dir = (repo / results_dir).resolve()
    else:
        results_dir = results_dir.resolve()

    results_root = finetune_results_root(repo).resolve()
    finetune_root = (repo / "results" / "finetune").resolve()
    deo = (repo / "dataset_exploration_outputs").resolve()

    for base in (results_root, finetune_root, deo):
        try:
            rel = results_dir.relative_to(base)
            return finetune_checkpoints_root() / rel
        except ValueError:
            continue

    # Last resort: keep the three innermost components (backbone/exp/head).
    parts = results_dir.parts
    rel = Path(*parts[-3:]) if len(parts) >= 3 else Path(parts[-1])
    return finetune_checkpoints_root() / rel
