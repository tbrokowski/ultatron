"""
scripts/pocus/paths.py  ·  WP5 path constants
=============================================

Defaults match the feasibility spec (11 Sep 2026).  Override with env:

  POCUS_ACCOUNT, POCUS_EVIDENCE_ROOT, POCUS_RAW_ROOT, POCUS_SHARD_ROOT,
  POCUS_SCRATCH_ROOT, ULTATRON_EDF_ENV, ULTATRON_REPO
"""
from __future__ import annotations

import os
from pathlib import Path


ACCOUNT_DEFAULT = os.environ.get("POCUS_ACCOUNT", os.environ.get("ULTATRON_ACCOUNT", "a127"))
# Spec §1: evidence tree is shared with WP1 under infra01.
EVIDENCE_ROOT = Path(os.environ.get(
    "POCUS_EVIDENCE_ROOT",
    "/capstor/store/cscs/swissai/infra01/meditron-feasibility-review/pocus",
))
STORE_ACCT = Path(os.environ.get(
    "POCUS_STORE_ACCT",
    f"/capstor/store/cscs/swissai/{ACCOUNT_DEFAULT}",
))
RAW_ROOT = Path(os.environ.get(
    "POCUS_RAW_ROOT",
    str(STORE_ACCT / "pocus-bench" / "raw"),
))
SHARD_ROOT = Path(os.environ.get(
    "POCUS_SHARD_ROOT",
    str(STORE_ACCT / "pocus-bench" / "shards"),
))
MANIFEST_ROOT = Path(os.environ.get(
    "POCUS_MANIFEST_ROOT",
    str(STORE_ACCT / "pocus-bench" / "manifests"),
))
_USER = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
SCRATCH_ROOT = Path(os.environ.get(
    "POCUS_SCRATCH_ROOT",
    f"/iopsstor/scratch/{_USER}/pocus-bench",
))
# Production US-365K already on Alps.
US365K_A127 = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/multi_organ/US-365K"
)
A127_US_ROOT = Path("/capstor/store/cscs/swissai/a127/ultrasound/raw")

DATASET_DEFAULTS = {
    "US-365K": US365K_A127,
    "COVID-BLUES": A127_US_ROOT / "lung" / "COVID-BLUES",
    "CardiacUDC": A127_US_ROOT / "cardiac" / "CardiacUDC",
    "IUGC2024": A127_US_ROOT / "fetal" / "IUGC-2024",
}

STUDENT_CKPT_DEFAULT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/checkpoints/StudentPretrain/step_30000.pt"
)

SEED = 1234
N_ENC_IMAGES = 100_000
N_RL_PROMPTS = 20_000
N_RL_HELDOUT = 1_000
N_RL_VIDEO = 200

LICENCES = {
    "US-365K": {
        "licence": "Academic research use per the paper; no licence on the HF card [TBC]",
        "source": "https://huggingface.co/datasets/JJY-0823/US-365K",
        "sensitivity": "research-only; no clinical identifiers expected",
    },
    "CardiacUDC": {
        "licence": "Apache-2.0 per the Ultrasound Open Access directory [TBC on the Kaggle page]",
        "source": "https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset",
        "sensitivity": "de-identified echo videos",
    },
    "COVID-BLUES": {
        "licence": "CC BY-NC-ND 4.0 (no derivatives: do not redistribute shards or derived data)",
        "source": "https://huggingface.co/datasets/jannisborn/COVID-BLUES",
        "sensitivity": "clinical LUS; do not republish derived shards",
    },
    "IUGC2024": {
        "licence": "CC BY 4.0 (cite Zenodo 10.5281/zenodo.17655183)",
        "source": "https://doi.org/10.5281/zenodo.17655183",
        "sensitivity": "de-identified obstetric videos",
    },
}


def evidence_job_dir(workload: str, jobid: str) -> Path:
    return EVIDENCE_ROOT / workload / str(jobid)
