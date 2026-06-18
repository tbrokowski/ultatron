#!/usr/bin/env bash
# Install comparison/ablation backbone deps for finetune sweeps.
#
# Called from scripts/finetune/run_all.py before scripts/finetune.py --comparison-config.
# Installs only when imports are missing; OpenUS mamba_ssm is best-effort (aarch64
# may need a long source build — failure is non-fatal and OpenUS is skipped).
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
USFM_VENDOR="${REPO_DIR}/finetune/backbones/vendor/usfm"
OPENUS_VENDOR="${REPO_DIR}/finetune/backbones/vendor/openus"
OPENUS_REPO="https://github.com/XZheng0427/OpenUS"

pip_install_if_missing() {
    local import_name="$1"
    local pip_spec="$2"
    if python3 -c "import ${import_name}" 2>/dev/null; then
        echo "[ensure_ablation_deps] ${import_name} already installed"
        return 0
    fi
    echo "[ensure_ablation_deps] Installing ${pip_spec} ..."
    python3 -m pip install -q --no-warn-conflicts "${pip_spec}"
}

echo "[ensure_ablation_deps] Checking ablation backbone dependencies ..."

pip_install_if_missing open_clip open_clip_torch
pip_install_if_missing monai monai

if python3 -c "import usdsgen" 2>/dev/null; then
    echo "[ensure_ablation_deps] usdsgen already installed"
elif [[ -d "${USFM_VENDOR}/usdsgen" ]]; then
    echo "[ensure_ablation_deps] Installing usdsgen (editable) from ${USFM_VENDOR} ..."
    python3 -m pip install -q -e "${USFM_VENDOR}"
else
    echo "[ensure_ablation_deps] WARN: USFM vendor not found at ${USFM_VENDOR} — usfm backbone will fail"
fi

if [[ -d "${OPENUS_VENDOR}/.git" ]] || [[ -f "${OPENUS_VENDOR}/README.md" ]]; then
    echo "[ensure_ablation_deps] OpenUS vendor present at ${OPENUS_VENDOR}"
else
    echo "[ensure_ablation_deps] Cloning OpenUS vendor → ${OPENUS_VENDOR} ..."
    mkdir -p "$(dirname "${OPENUS_VENDOR}")"
    git clone --depth=1 "${OPENUS_REPO}" "${OPENUS_VENDOR}"
fi

if python3 -c "import mamba_ssm" 2>/dev/null; then
    echo "[ensure_ablation_deps] mamba_ssm already installed"
else
    echo "[ensure_ablation_deps] mamba_ssm missing — attempting source install (optional, OpenUS only) ..."
    python3 -m pip install -q --no-warn-conflicts packaging ninja einops || true
    set +e
    python3 -m pip install -q causal-conv1d --no-build-isolation 2>/dev/null
    MAMBA_RC=$?
    if [[ ${MAMBA_RC} -eq 0 ]]; then
        python3 -m pip install -q mamba-ssm --no-build-isolation 2>/dev/null
        MAMBA_RC=$?
    fi
    set -e
    if [[ ${MAMBA_RC} -eq 0 ]] && python3 -c "import mamba_ssm" 2>/dev/null; then
        echo "[ensure_ablation_deps] mamba_ssm installed"
    else
        echo "[ensure_ablation_deps] WARN: mamba_ssm install failed — OpenUS backbone will be skipped"
    fi
fi

echo "[ensure_ablation_deps] Done."
