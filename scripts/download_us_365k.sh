#!/usr/bin/env bash
# =============================================================================
# download_us_365k.sh  ·  US-365K image-caption dataset (Hugging Face)
# =============================================================================
#
# Downloads JJY-0823/US-365K from Hugging Face into:
#   /capstor/store/cscs/swissai/a127/ultrasound/raw/multi_organ/US-365K
#
# Layout after completion:
#   US-365K/
#     hf_snapshot/     raw HF repo (images.zip + JSONL)
#     images/          materialized JPEGs
#     metadata/        train.jsonl, val.jsonl, test.jsonl
#
# Prerequisites:
#   Optional HF_TOKEN if the repo becomes gated:
#     export HF_TOKEN="hf_..."
#
# Usage:
#   sbatch scripts/download_us_365k.sh
#   bash   scripts/download_us_365k.sh
#
# Monitor:
#   tail -f logs/download/us_365k_<jobid>.out
#   squeue -u $USER
# =============================================================================
#SBATCH --job-name=us_365k_download
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/users/tbrokowski/Ultatron/logs/download/us_365k_%j.out
#SBATCH --error=/users/tbrokowski/Ultatron/logs/download/us_365k_%j.err

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
TARGET_DIR="/capstor/store/cscs/swissai/a127/ultrasound/raw/multi_organ/US-365K"
VENV="${REPO_DIR}/.venv"
SNAPSHOT_DIR="${TARGET_DIR}/hf_snapshot"

mkdir -p "${TARGET_DIR}"
mkdir -p "${REPO_DIR}/logs/download"

echo "================================================================"
echo " US-365K download (Hugging Face)"
echo " Job    : ${SLURM_JOB_ID:-local}"
echo " Node   : $(hostname)"
echo " Target : ${TARGET_DIR}"
echo " Start  : $(date -Is)"
echo "================================================================"

source "${VENV}/bin/activate"

python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub

if [[ -f "${SNAPSHOT_DIR}/images.zip" && -f "${SNAPSHOT_DIR}/train.jsonl" ]]; then
  echo ""
  echo "[1/2] HF snapshot already present — skipping download"
else
  echo ""
  echo "[1/2] Downloading HF snapshot → ${SNAPSHOT_DIR}"
  python3 - <<PYEOF
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="JJY-0823/US-365K",
    repo_type="dataset",
    local_dir="${SNAPSHOT_DIR}",
    local_dir_use_symlinks=False,
)
print(f"Snapshot ready: {path}")
PYEOF
fi

echo ""
echo "[2/2] Materializing images + metadata JSONL"
python3 "${REPO_DIR}/scripts/materialize_us365k.py" --target "${TARGET_DIR}"

echo ""
echo "Verification:"
echo "  images:   $(find "${TARGET_DIR}/images" -type f 2>/dev/null | wc -l) files"
echo "  metadata: $(wc -l "${TARGET_DIR}"/metadata/*.jsonl 2>/dev/null || echo 'none')"
head -1 "${TARGET_DIR}/metadata/train.jsonl" 2>/dev/null || true

echo ""
echo "================================================================"
echo " Finished : $(date -Is)"
echo "================================================================"
