#!/usr/bin/env bash
# capture_env.sh  ·  Spec §3.4 environment capture for every WP5 job
# Writes env/<jobid>.txt (or $1) with git, image, pip, nvidia-smi, NCCL/FI, Slurm, config.
set -euo pipefail

OUT="${1:?usage: capture_env.sh OUTFILE [config.yaml]}"
CFG="${2:-}"
REPO="${ULTATRON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

mkdir -p "$(dirname "${OUT}")"
{
    echo "===== git ====="
    git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo "HEAD: unavailable"
    git -C "${REPO}" status --porcelain=v1 2>/dev/null || true
    echo
    echo "===== container ====="
    echo "EDF=${ULTATRON_EDF_ENV:-${HOME}/.edf/ultatron.toml}"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
    fi
    # Best-effort image path/digest from the EDF.
    if [[ -f "${ULTATRON_EDF_ENV:-${HOME}/.edf/ultatron.toml}" ]]; then
        grep -E '^(image|sqsh)' "${ULTATRON_EDF_ENV:-${HOME}/.edf/ultatron.toml}" || true
    fi
    echo
    echo "===== pip freeze ====="
    python3 -m pip freeze 2>/dev/null || pip freeze 2>/dev/null || echo "pip freeze unavailable"
    echo
    echo "===== nvidia-smi (header) ====="
    nvidia-smi -q 2>/dev/null | head -n 80 || echo "nvidia-smi unavailable"
    echo
    echo "===== NCCL_* / FI_* ====="
    env | grep -E '^(NCCL_|FI_|LD_LIBRARY_PATH=)' | sort || true
    echo
    echo "===== Slurm ====="
    env | grep -E '^SLURM_' | sort || true
    echo
    echo "===== config ====="
    if [[ -n "${CFG}" && -f "${CFG}" ]]; then
        echo "# ${CFG}"
        cat "${CFG}"
    elif [[ -n "${US_STUDENT_CONFIG:-}" && -f "${US_STUDENT_CONFIG}" ]]; then
        echo "# ${US_STUDENT_CONFIG}"
        cat "${US_STUDENT_CONFIG}"
    else
        echo "(no yaml path provided)"
    fi
} > "${OUT}"

echo "env capture → ${OUT}"
