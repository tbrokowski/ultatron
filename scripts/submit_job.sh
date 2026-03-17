#!/usr/bin/env bash
# =============================================================================
# submit_job.sh  ·  Ultatron CSCS Alps job submitter
# =============================================================================
#
# Usage:
#   bash scripts/submit_job.sh -smoke
#   bash scripts/submit_job.sh -minimalrun
#   bash scripts/submit_job.sh -run1
#   bash scripts/submit_job.sh -run1 --resume /path/to/latest.pt
#
# Run modes
#   -smoke        Single-GPU smoke test (~10 min). Validates the full pipeline
#                 (data adapters, model builds, all 4 phase steps) without DDP.
#   -minimalrun   4-GPU DDP run, 50 training steps. Verifies the distributed
#                 training loop end-to-end before committing to a full run.
#   -run1         Full 20k-step run1 training on 4×GH200.
#
# Optional flags (all modes except -smoke):
#   --resume <path>   Resume from checkpoint (auto-detects latest.pt if absent)
#   --no-7b           Skip frozen 7B teacher (already default for run1)
# =============================================================================

set -euo pipefail

# ── Repo / cluster constants ─────────────────────────────────────────────────
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

# ── Helpers ──────────────────────────────────────────────────────────────────
die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO]  $*"; }

# Try venv torchrun, fall back to python -m torch.distributed.run
_launcher() {
    if [[ -x "${REPO_DIR}/.venv/bin/torchrun" ]]; then
        echo "${REPO_DIR}/.venv/bin/torchrun"
    else
        echo "python -m torch.distributed.run"
    fi
}

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE=""
RESUME_ARG=""
NO_7B_ARG="--no-7b"   # default for all runs in this iteration

while [[ $# -gt 0 ]]; do
    case "$1" in
        -smoke)       MODE="smoke";      shift ;;
        -minimalrun)  MODE="minimalrun"; shift ;;
        -run1)        MODE="run1";       shift ;;
        --resume)     RESUME_ARG="--resume $2"; shift 2 ;;
        --no-7b)      NO_7B_ARG="--no-7b"; shift ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "Unknown argument: $1. Use -smoke | -minimalrun | -run1" ;;
    esac
done

[[ -z "$MODE" ]] && die "No mode specified. Use: bash $0 -smoke | -minimalrun | -run1"

# ── Per-mode settings ─────────────────────────────────────────────────────────
case "$MODE" in
smoke)
    JOB_NAME="ultatron_smoke"
    TIME="00:20:00"
    NODES=1
    GPUS=1
    CPUS=8
    LOG_DIR="${LOG_ROOT}/smoke"
    ;;
minimalrun)
    JOB_NAME="ultatron_minimalrun"
    TIME="00:30:00"
    NODES=1
    GPUS=4
    CPUS=32
    LOG_DIR="${LOG_ROOT}/minimalrun"
    ;;
run1)
    JOB_NAME="ultatron_run1"
    TIME="12:00:00"
    NODES=1
    GPUS=4
    CPUS=32
    LOG_DIR="${LOG_ROOT}/run1"
    CKPT_DIR="${CKPT_ROOT}/run1"
    # Auto-resume from latest.pt if no explicit --resume given
    if [[ -z "$RESUME_ARG" && -f "${CKPT_DIR}/latest.pt" ]]; then
        RESUME_ARG="--resume ${CKPT_DIR}/latest.pt"
        info "Auto-resume: ${CKPT_DIR}/latest.pt"
    fi
    ;;
esac

mkdir -p "${LOG_DIR}"

# ── Build the job body ────────────────────────────────────────────────────────
COMMON_HEADER='
set -euo pipefail
cd '"${REPO_DIR}"'
source .venv/bin/activate

echo "================================================================"
echo " Ultatron — '"${JOB_NAME}"'"
echo " Job    : ${SLURM_JOB_ID:-local}"
echo " Node   : $(hostname)"
echo " Start  : $(date)"
python -c "import torch; print(f\" PyTorch : {torch.__version__}\")"
python -c "import torch; print(f\" CUDA    : {torch.version.cuda}\")"
python -c "import torch; print(f\" GPUs    : {torch.cuda.device_count()} visible\")"
echo "================================================================"
echo ""
'

case "$MODE" in
# ── Smoke ─────────────────────────────────────────────────────────────────────
smoke)
JOB_BODY="${COMMON_HEADER}
export US_BUSI_ROOT=${BUSI_ROOT}
export US_ECHONET_ROOT=${ECHONET_ROOT}
export US_BENIN_ROOT=${BENIN_ROOT}

echo \"Running: python -m tests.dataset_adapters.training_smoke\"
echo \"Dataset roots:\"
echo \"  BUSI    : \${US_BUSI_ROOT}\"
echo \"  EchoNet : \${US_ECHONET_ROOT}\"
echo \"  Benin   : \${US_BENIN_ROOT}\"
echo \"\"

python -m tests.dataset_adapters.training_smoke

echo \"\"
echo \"================================================================\"
echo \" SMOKE PASSED  — $(date)\"
echo \"================================================================\"
"
;;

# ── Minimal run ───────────────────────────────────────────────────────────────
minimalrun)
LAUNCHER=$(_launcher)
JOB_BODY="${COMMON_HEADER}
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export OMP_NUM_THREADS=4

echo \"Running 50-step DDP smoke run (4 GPUs)...\"
echo \"\"

${LAUNCHER} \\
    --nproc_per_node=4 \\
    scripts/train.py \\
    --config configs/run1/minimal_run1.yaml \\
    ${NO_7B_ARG}

echo \"\"
echo \"================================================================\"
echo \" MINIMAL RUN PASSED  — \$(date)\"
echo \"================================================================\"
"
;;

# ── Full run1 ─────────────────────────────────────────────────────────────────
run1)
LAUNCHER=$(_launcher)
mkdir -p "${CKPT_DIR}"
JOB_BODY="${COMMON_HEADER}
export NCCL_SOCKET_IFNAME=hsn0,hsn1
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export OMP_NUM_THREADS=4

# SIGUSR1 → checkpoint + requeue (SLURM pre-emption)
_checkpoint_and_requeue() {
    echo \"[\$(date)] SIGUSR1: pre-emption signal — checkpointing...\"
    touch ${CKPT_DIR}/.checkpoint_now
    sleep 60
    scontrol requeue \"\${SLURM_JOB_ID}\"
    echo \"[\$(date)] Job requeued.\"
}
trap _checkpoint_and_requeue SIGUSR1

echo \"Checkpoint dir : ${CKPT_DIR}\"
echo \"Resume         : ${RESUME_ARG:-none}\"
echo \"\"

${LAUNCHER} \\
    --nproc_per_node=4 \\
    scripts/train.py \\
    --config configs/experiments/run1.yaml \\
    ${NO_7B_ARG} \\
    ${RESUME_ARG}

TRAIN_EXIT=\$?

if [[ \${TRAIN_EXIT} -eq 0 ]]; then
    echo \"\"
    echo \"--- Post-training validation ---\"
    python train/validate.py \\
        --config configs/experiments/run1.yaml \\
        --checkpoint ${CKPT_DIR}/latest.pt \\
        --mode full \\
        --no-7b \\
        --output ${LOG_DIR}/validation_\${SLURM_JOB_ID}.json \\
    && echo \"Validation → ${LOG_DIR}/validation_\${SLURM_JOB_ID}.json\" \\
    || echo \"Validation failed (non-fatal)\"
fi

echo \"\"
echo \"================================================================\"
echo \" RUN1 COMPLETE  exit=\${TRAIN_EXIT}  \$(date)\"
if [[ \${TRAIN_EXIT} -ne 0 ]]; then
    echo \" Resume with: bash scripts/submit_job.sh -run1 --resume ${CKPT_DIR}/latest.pt\"
fi
echo \"================================================================\"
exit \${TRAIN_EXIT}
"
;;
esac

# ── Submit ────────────────────────────────────────────────────────────────────
info "Submitting: ${JOB_NAME}  (mode=${MODE}, nodes=${NODES}, gpus=${GPUS}, time=${TIME})"

EXTRA_FLAGS=""
[[ "$MODE" == "run1" ]] && EXTRA_FLAGS="--signal=SIGUSR1@120"

JOB_ID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --nodes="${NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS}" \
    --cpus-per-task="${CPUS}" \
    --time="${TIME}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --environment="${EDF_ENV}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --parsable \
    ${EXTRA_FLAGS} \
    --wrap="${JOB_BODY}")

echo ""
echo "  Job ID   : ${JOB_ID}"
echo "  Log      : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Watch    : tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Queue    : squeue -u \$USER -j ${JOB_ID}"
