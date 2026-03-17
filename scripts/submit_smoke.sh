#!/usr/bin/env bash
# =============================================================================
# submit_smoke.sh  ·  Submit the multi-dataset training smoke test to CSCS Alps
# =============================================================================
#
# Runs tests/dataset_adapters/training_smoke.py on a single GH200 node.
# Uses 1 GPU (smoke runs single-process, not distributed).
# Expected runtime: ~5–10 minutes.
#
# Submit from the login node:
#   bash scripts/submit_smoke.sh
#
# Override dataset roots via environment if data is elsewhere:
#   US_BUSI_ROOT=/other/path bash scripts/submit_smoke.sh
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
TIME_LIMIT="00:20:00"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
LOG_DIR="${REPO_DIR}/logs/smoke"
JOB_NAME="ultatron_smoke"

mkdir -p "${LOG_DIR}"

echo "Submitting smoke test job..."

JOB_ID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --cpus-per-task=8 \
    --time="${TIME_LIMIT}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --environment="${EDF_ENV}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --parsable \
    --wrap="
set -euo pipefail
cd ${REPO_DIR}
source .venv/bin/activate

echo '================================================================'
echo ' Ultatron Smoke Test'
echo \" Job ID   : \${SLURM_JOB_ID}\"
echo \" Node     : \$(hostname)\"
echo \" Start    : \$(date)\"
echo \" Python   : \$(python --version)\"
echo \" PyTorch  : \$(python -c 'import torch; print(torch.__version__)')\"
echo \" CUDA     : \$(python -c 'import torch; print(torch.version.cuda)')\"
echo \" GPUs     : \$(python -c 'import torch; print(torch.cuda.device_count())')\"
echo '================================================================'

# Set dataset roots (default: capstor store paths)
export US_BUSI_ROOT=\${US_BUSI_ROOT:-/capstor/store/cscs/swissai/a127/ultrasound/raw/breast/BUSI}
export US_ECHONET_ROOT=\${US_ECHONET_ROOT:-/capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/EchoNet-Dynamic}
export US_BENIN_ROOT=\${US_BENIN_ROOT:-/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/Benin_Videos}

echo \"BUSI root     : \${US_BUSI_ROOT}\"
echo \"EchoNet root  : \${US_ECHONET_ROOT}\"
echo \"Benin root    : \${US_BENIN_ROOT}\"
echo ''

python -m tests.dataset_adapters.training_smoke

echo ''
echo '================================================================'
echo \" Smoke test PASSED\"
echo \" End : \$(date)\"
echo '================================================================'
")

echo "Submitted job ${JOB_ID}"
echo "Logs: ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo ""
echo "Watch: tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "Queue: squeue -u \$USER -j ${JOB_ID}"
