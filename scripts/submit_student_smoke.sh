#!/usr/bin/env bash
# =============================================================================
# submit_student_smoke.sh  ·  Submit the single-student pipeline smoke test
# =============================================================================
#
# Full 800-step smoke run: loads real models (Hiera + DINOv3-L + V-JEPA2-L),
# samples from every configured dataset, and trains 200 steps per curriculum
# stage on 4× GH200 GPUs (DDP).
#
# Usage:
#   bash scripts/submit_student_smoke.sh              # 4-GPU DDP (default)
#   bash scripts/submit_student_smoke.sh --single   # single-GPU debug
#
# Override steps, checkpoint dir, model variant, or resume:
#   US_SMOKE_STEPS=500 bash scripts/submit_student_smoke.sh
#   bash scripts/submit_student_smoke.sh --resume
#   US_STUDENT_HIERA_VARIANT=hiera_large_video_mae_k400 bash scripts/submit_student_smoke.sh
#   US_SMOKE_FORCE_REBUILD=1 bash scripts/submit_student_smoke.sh
#
# Checkpoints saved to (alongside checkpoints/Ablations/):
#   /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/StudentSmoke/
#     stage1_end.pt  stage2_end.pt  stage3_end.pt  latest.pt
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
LOG_DIR="${REPO_DIR}/logs/smoke"
JOB_NAME="ultatron_student_smoke"

DDP=1
TIME_LIMIT="06:00:00"
GPUS=4
CPUS=32
RESUME=0
RESUME_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --single|-single) DDP=0; GPUS=1; CPUS=8; TIME_LIMIT="03:00:00"; shift ;;
        --ddp|-ddp)       DDP=1; GPUS=4; CPUS=32; TIME_LIMIT="06:00:00"; shift ;;
        --resume|-resume) RESUME=1; RESUME_ARGS="--resume"; shift ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${DDP}" -eq 0 ]]; then
    JOB_NAME="ultatron_student_smoke_1gpu"
fi

mkdir -p "${LOG_DIR}"

INNERSCRIPT="${LOG_DIR}/.inner_student_smoke.sh"
OUTERSCRIPT="$(mktemp /tmp/ultatron_outer_student_smoke_XXXXX.sh)"
trap "rm -f ${OUTERSCRIPT}" EXIT

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
ulimit -c 0
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESUME_ARGS="${RESUME_ARGS}"

echo "================================================================"
echo " Ultatron — Student Pipeline Smoke (800 steps, 200/stage)"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
echo " Mode   : $([ "${DDP}" -eq 1 ] && echo "4-GPU DDP" || echo "single-GPU")"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""

bash "${REPO_DIR}/scripts/ensure_deps.sh"
source "${REPO_DIR}/scripts/setup_hf_cache.sh"

if [[ "\${US_STUDENT_HIERA_VARIANT:-}" == "hiera_large_video_mae_k400" ]]; then
    pip install --quiet hiera-transformer
fi

INNER_EOF

if [[ "${DDP}" -eq 1 ]]; then
cat >> "${INNERSCRIPT}" << 'INNER_DDP_EOF'

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

echo "Running 800-step student smoke (4-GPU DDP, 200 steps/stage)..."
python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    -m tests.dataset_adapters.student_training_smoke ${RESUME_ARGS}

INNER_DDP_EOF
else
cat >> "${INNERSCRIPT}" << 'INNER_SINGLE_EOF'

echo "Running 800-step student smoke (single-GPU, 200 steps/stage)..."
python3 -m tests.dataset_adapters.student_training_smoke ${RESUME_ARGS}

INNER_SINGLE_EOF
fi

cat >> "${INNERSCRIPT}" << 'INNER_EOF'

echo ""
echo "================================================================"
echo " STUDENT SMOKE PASSED  -- $(date)"
echo "================================================================"
INNER_EOF

chmod +x "${INNERSCRIPT}"

cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
set -euo pipefail
srun --ntasks-per-node=1 \
     --environment=${EDF_ENV} \
     bash ${INNERSCRIPT}
OUTER_EOF

chmod +x "${OUTERSCRIPT}"

echo "Submitting student smoke (${JOB_NAME}, ${GPUS} GPU, 800 steps)..."
JOB_ID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS}" \
    --cpus-per-task="${CPUS}" \
    --time="${TIME_LIMIT}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --parsable \
    "${OUTERSCRIPT}")

echo ""
echo "  Job ID : ${JOB_ID}"
echo "  Log    : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Watch  : tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out   # milestones + checkpoints"
echo "  Errors : tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.err   # full Python logs"
