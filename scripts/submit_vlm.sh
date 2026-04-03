#!/usr/bin/env bash
# =============================================================================
# submit_vlm.sh  ·  Ultatron VLM GRPO job submitter (CSCS Alps)
# =============================================================================
#
# Usage:
#   bash scripts/submit_vlm.sh                       # full run1_vlm GRPO
#   bash scripts/submit_vlm.sh --stage 2             # force-start at Stage 2
#   bash scripts/submit_vlm.sh --no-medgemini        # rule-based rewards only
#   bash scripts/submit_vlm.sh --resume /path/ckpt   # resume from checkpoint
#
# Requires:
#   - run1 SSL training complete (phase3_end.pt must exist)
#   - run1 manifest built (run1_train.jsonl must exist)
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"

CKPT_ROOT="/capstor/scratch/cscs/tbrokowski/ultrasound/checkpoints"
LOG_ROOT="${REPO_DIR}/logs"
LOG_DIR="${LOG_ROOT}/vlm_run1"

JOB_NAME="ultatron_vlm_run1"
TIME="12:00:00"
NODES=1
GPUS=4
CPUS=64

# ── Parse args ────────────────────────────────────────────────────────────────
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)        EXTRA_ARGS="$EXTRA_ARGS --stage $2"; shift 2 ;;
        --resume)       EXTRA_ARGS="$EXTRA_ARGS --resume $2"; shift 2 ;;
        --no-sam2)      EXTRA_ARGS="$EXTRA_ARGS --no-sam2"; shift ;;
        --no-medgemini) EXTRA_ARGS="$EXTRA_ARGS --no-medgemini"; shift ;;
        --eval-only)    EXTRA_ARGS="$EXTRA_ARGS --eval-only"; shift ;;
        *) echo "[ERROR] Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"
INNERSCRIPT="${LOG_DIR}/.inner_vlm.sh"
OUTERSCRIPT=$(mktemp /tmp/ultatron_outer_vlm_XXXXX.sh)
trap "rm -f ${OUTERSCRIPT}" EXIT

# ── Inner compute script ──────────────────────────────────────────────────────
cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"

# Single-node 4×GH200: NVLink for P2P, TCP for NCCL coordination
export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export OMP_NUM_THREADS=8

echo "================================================================"
echo " Ultatron VLM GRPO — ${JOB_NAME}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""

# Verify run1 backbone checkpoint exists
BACKBONE_CKPT="${CKPT_ROOT}/run1/phase3_end.pt"
if [[ ! -f "\${BACKBONE_CKPT}" ]]; then
    echo "[ERROR] run1 backbone checkpoint not found: \${BACKBONE_CKPT}"
    echo "        Run SSL training first: bash scripts/submit_job.sh -run1"
    exit 1
fi

# Verify manifest exists
MANIFEST="${LOG_ROOT}/run1/run1_train.jsonl"
if [[ ! -f "\${MANIFEST}" ]]; then
    echo "Building run1 manifest → \${MANIFEST}"
    python3 scripts/build_manifest.py \
        --config configs/run1/data_run1.yaml \
        --out "\${MANIFEST}" \
        || { echo "ERROR: manifest build failed"; exit 1; }
fi

echo "Backbone  : \${BACKBONE_CKPT}"
echo "Manifest  : \${MANIFEST} (\$(wc -l < \${MANIFEST}) entries)"
echo ""

torchrun \
    --nproc_per_node=4 \
    scripts/train_vlm.py \
    --config configs/vlm/run1_vlm.yaml \
    ${EXTRA_ARGS}

TRAIN_EXIT=\$?
echo ""
echo "================================================================"
echo " VLM GRPO COMPLETE  exit=\${TRAIN_EXIT}  \$(date)"
[[ \${TRAIN_EXIT} -ne 0 ]] && echo " Resume: bash scripts/submit_vlm.sh --resume ${CKPT_ROOT}/vlm_run1/latest"
echo "================================================================"
exit \${TRAIN_EXIT}
INNER_EOF

chmod +x "${INNERSCRIPT}"

# ── Outer wrapper ─────────────────────────────────────────────────────────────
cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
set -euo pipefail
srun --ntasks-per-node=1 \
     --environment=${EDF_ENV} \
     bash ${INNERSCRIPT}
OUTER_EOF

chmod +x "${OUTERSCRIPT}"

# ── Submit ────────────────────────────────────────────────────────────────────
echo "[INFO] Submitting: ${JOB_NAME}  (nodes=${NODES}, gpus=${GPUS}, time=${TIME})"
echo "[INFO] Extra args: ${EXTRA_ARGS:-none}"

JOB_ID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --nodes="${NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS}" \
    --cpus-per-task="${CPUS}" \
    --time="${TIME}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --signal=SIGUSR1@120 \
    --parsable \
    "${OUTERSCRIPT}")

echo ""
echo "  Job ID   : ${JOB_ID}"
echo "  Log      : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Watch    : tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Queue    : squeue -u \$USER -j ${JOB_ID}"
echo ""
echo "  Resume if preempted:"
echo "    bash scripts/submit_vlm.sh --resume ${CKPT_ROOT}/vlm_run1/latest"
