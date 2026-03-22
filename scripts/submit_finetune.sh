#!/usr/bin/env bash
# =============================================================================
# submit_finetune.sh  ·  Ultatron Phase 4 — downstream head fine-tuning
# =============================================================================
#
# Usage:
#   # Run immediately (uses latest.pt from run1 by default):
#   bash scripts/submit_finetune.sh
#
#   # Explicit checkpoint:
#   bash scripts/submit_finetune.sh --checkpoint /path/to/phase3_end.pt
#
#   # Chain after a running training job (SLURM dependency):
#   bash scripts/submit_finetune.sh --after-job 1669838
#
#   # Run only specific experiments:
#   bash scripts/submit_finetune.sh --experiments busi echonet
#
#   # Evaluate only (skip training, load best_head.pt):
#   bash scripts/submit_finetune.sh --eval-only
#
# All three experiments run sequentially on a single GPU.
# Expected wall time: ~3–4 h total (BUSI <5 min, EchoNet ~2 h, LUS ~1 h).
# =============================================================================

set -euo pipefail

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
CKPT_ROOT="/capstor/scratch/cscs/tbrokowski/ultrasound/checkpoints"
LOG_ROOT="${REPO_DIR}/logs"

STORE="/capstor/store/cscs/swissai/a127/ultrasound/raw"
BUSI_ROOT="${STORE}/breast/BUSI"
ECHONET_ROOT="${STORE}/cardiac/EchoNet-Dynamic"
BENIN_ROOT="${STORE}/lung/Benin_Videos"
RSA_ROOT="${STORE}/lung/RSA_Videos"

# ── Helpers ───────────────────────────────────────────────────────────────────
die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO]  $*"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
CHECKPOINT=""
AFTER_JOB=""
EVAL_ONLY_ARG=""
EXPERIMENTS_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$2";             shift 2 ;;
        --after-job)    AFTER_JOB="$2";              shift 2 ;;
        --eval-only)    EVAL_ONLY_ARG="--eval-only"; shift   ;;
        --experiments)
            # Consume all following non-flag arguments as experiment names
            EXPERIMENTS_ARG="--experiments"
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                EXPERIMENTS_ARG="${EXPERIMENTS_ARG} $1"
                shift
            done
            ;;
        -h|--help)
            sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# Default checkpoint: phase3_end.pt or latest.pt from run1
if [[ -z "$CHECKPOINT" ]]; then
    if [[ -f "${CKPT_ROOT}/run1/phase3_end.pt" ]]; then
        CHECKPOINT="${CKPT_ROOT}/run1/phase3_end.pt"
        info "Auto-detected checkpoint: phase3_end.pt"
    elif [[ -f "${CKPT_ROOT}/run1/latest.pt" ]]; then
        CHECKPOINT="${CKPT_ROOT}/run1/latest.pt"
        info "Auto-detected checkpoint: latest.pt (phase3_end.pt not yet saved)"
    else
        die "No checkpoint found in ${CKPT_ROOT}/run1/.  Use --checkpoint to specify one."
    fi
fi

[[ -f "$CHECKPOINT" ]] || die "Checkpoint does not exist: $CHECKPOINT"

# Output dir derived from checkpoint location
CKPT_DIR=$(dirname "${CHECKPOINT}")
OUTPUT_DIR="${CKPT_DIR}/finetune"
LOG_DIR="${LOG_ROOT}/finetune"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

JOB_NAME="ultatron_finetune"
TIME="05:00:00"   # 5 h budget — 3 experiments including EchoNet
NODES=1
GPUS=1            # Phase 4 runs on a single GPU (no DDP)
CPUS=16

# ── Inner compute script ──────────────────────────────────────────────────────
INNERSCRIPT="${LOG_DIR}/.inner_finetune.sh"

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"

# Strip AWS OFI NCCL plugin path (not needed for single-GPU finetune)
export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

echo "================================================================"
echo " Ultatron — ${JOB_NAME}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""
echo "Checkpoint : ${CHECKPOINT}"
echo "Output dir : ${OUTPUT_DIR}"
echo ""

python3 scripts/finetune.py \
    --checkpoint ${CHECKPOINT} \
    --train-config configs/experiments/run1.yaml \
    --output-dir ${OUTPUT_DIR} \
    --busi-root    ${BUSI_ROOT} \
    --echonet-root ${ECHONET_ROOT} \
    --benin-root   ${BENIN_ROOT} \
    --rsa-root     ${RSA_ROOT} \
    ${EVAL_ONLY_ARG} \
    ${EXPERIMENTS_ARG}

FT_EXIT=\$?

echo ""
echo "================================================================"
echo " FINETUNE COMPLETE  exit=\${FT_EXIT}  \$(date)"
[[ \${FT_EXIT} -eq 0 ]] && echo " Results: ${OUTPUT_DIR}/results_summary.json"
echo "================================================================"
exit \${FT_EXIT}
INNER_EOF

chmod +x "${INNERSCRIPT}"

# ── Outer SLURM wrapper ───────────────────────────────────────────────────────
OUTERSCRIPT=$(mktemp /tmp/ultatron_outer_finetune_XXXXX.sh)
trap "rm -f ${OUTERSCRIPT}" EXIT

cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
# Auto-generated by submit_finetune.sh
set -euo pipefail

srun --ntasks-per-node=1 \
     --environment=${EDF_ENV} \
     bash ${INNERSCRIPT}
OUTER_EOF

chmod +x "${OUTERSCRIPT}"

# ── Build dependency flag if requested ────────────────────────────────────────
DEPENDENCY_FLAG=""
if [[ -n "$AFTER_JOB" ]]; then
    DEPENDENCY_FLAG="--dependency=afterok:${AFTER_JOB}"
    info "Will start after job ${AFTER_JOB} completes successfully."
fi

# ── Submit ────────────────────────────────────────────────────────────────────
info "Submitting: ${JOB_NAME}  (gpus=${GPUS}, time=${TIME})"

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
    --parsable \
    ${DEPENDENCY_FLAG} \
    "${OUTERSCRIPT}")

echo ""
echo "  Job ID     : ${JOB_ID}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Output     : ${OUTPUT_DIR}/results_summary.json"
echo "  Log        : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Watch      : tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Queue      : squeue -u \$USER -j ${JOB_ID}"
