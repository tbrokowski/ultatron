#!/usr/bin/env bash
# Install USFM segmentation stack (usdsgen + mmcv + mmsegmentation) for BUSI replication.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
USFM_VENDOR="${REPO_DIR}/finetune/backbones/vendor/usfm"
MMSEG_VENDOR="${REPO_DIR}/finetune/backbones/vendor/mmsegmentation"
MMSEG_REPO="https://github.com/George-Jiao/mmsegmentation.git"
MMSEG_BRANCH="gj_mmcv2_2_0"

echo "[ensure_usfm_seg_deps] USFM vendor: ${USFM_VENDOR}"

if [[ ! -d "${USFM_VENDOR}/usdsgen" ]]; then
    echo "[ensure_usfm_seg_deps] Cloning USFM → ${USFM_VENDOR} ..."
    mkdir -p "$(dirname "${USFM_VENDOR}")"
    git clone --depth=1 https://github.com/openmedlab/USFM.git "${USFM_VENDOR}"
fi

if ! python3 -c "import usdsgen" 2>/dev/null; then
    echo "[ensure_usfm_seg_deps] Installing usdsgen (editable) ..."
    python3 -m pip install -q -r "${USFM_VENDOR}/requirements.txt" || true
    python3 -m pip install -q -e "${USFM_VENDOR}"
fi

if ! python3 -c "import mmcv" 2>/dev/null; then
    echo "[ensure_usfm_seg_deps] Installing mmcv (may take a few minutes) ..."
    TORCH_VER=$(python3 -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))" 2>/dev/null || echo "2.4")
    CUDA_TAG="cu118"
    if python3 -c "import torch; v=torch.version.cuda; exit(0 if v and v.startswith('12') else 1)" 2>/dev/null; then
        CUDA_TAG="cu121"
    fi
    python3 -m pip install -q "mmcv==2.2.0" \
        -f "https://download.openmmlab.com/mmcv/dist/${CUDA_TAG}/torch${TORCH_VER}/index.html" \
        || python3 -m pip install -q "mmcv==2.2.0"
fi

if ! python3 -c "import mmseg" 2>/dev/null; then
    if [[ ! -d "${MMSEG_VENDOR}/.git" ]]; then
        echo "[ensure_usfm_seg_deps] Cloning modified mmsegmentation → ${MMSEG_VENDOR} ..."
        mkdir -p "$(dirname "${MMSEG_VENDOR}")"
        git clone --depth=1 --branch "${MMSEG_BRANCH}" "${MMSEG_REPO}" "${MMSEG_VENDOR}" \
            || git clone --depth=1 "${MMSEG_REPO}" "${MMSEG_VENDOR}"
    fi
    echo "[ensure_usfm_seg_deps] Installing mmsegmentation (editable) ..."
    python3 -m pip install -q -e "${MMSEG_VENDOR}"
fi

python3 - <<'PY'
import usdsgen  # noqa: F401
import mmcv  # noqa: F401
import mmseg  # noqa: F401
print("[ensure_usfm_seg_deps] usdsgen + mmcv + mmseg OK")
PY

echo "[ensure_usfm_seg_deps] Done."
