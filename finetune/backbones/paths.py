"""
finetune/backbones/paths.py  ·  External FM checkpoint path resolution
======================================================================

Ablation / comparison weights live on Capstor store at:
  /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/Ablations/

Falls back to repo-local AblationModelWeights/ for local dev.
"""
from __future__ import annotations

import os
from pathlib import Path

from data.infra.cscs_paths import ABLATION_WEIGHTS_STORE, STUDENT_SMOKE_CHECKPOINTS_STORE

_REPO_ABLATION = Path(__file__).resolve().parents[2] / "AblationModelWeights"
_STORE_ABLATION = Path(ABLATION_WEIGHTS_STORE)


def ablation_weights_dir() -> Path:
    if _STORE_ABLATION.exists():
        return _STORE_ABLATION
    if _REPO_ABLATION.exists():
        return _REPO_ABLATION
    return _STORE_ABLATION


def ablation_weight_path(filename: str, env_var: str | None = None) -> str:
    if env_var and (override := os.environ.get(env_var)):
        return override
    return str(ablation_weights_dir() / filename)


def student_smoke_checkpoints_dir() -> Path:
    """
    Directory for student-pipeline smoke checkpoints on Capstor store:
      /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/StudentSmoke/
    Sibling of checkpoints/Ablations/.  Override with US_STUDENT_SMOKE_CKPT_DIR.
    """
    if override := os.environ.get("US_STUDENT_SMOKE_CKPT_DIR"):
        return Path(override)
    return Path(STUDENT_SMOKE_CHECKPOINTS_STORE)
