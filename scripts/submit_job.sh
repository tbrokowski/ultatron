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
#   -smoke        Single-GPU smoke test (~20 min). Validates the full pipeline
#                 (data adapters, model builds, all 4 phase steps) without DDP.
#   -minimalrun   4-GPU DDP run, 50 training steps. Verifies the distributed
#                 training loop end-to-end before committing to a full run.
#   -run1         Full 20k-step run1 training on 4×GH200.
#
# Optional flags (all modes except -smoke):
#   --resume <path>   Resume from checkpoint (auto-detects latest.pt if absent)
#   --no-7b           Skip frozen 7B teacher (already default for run1)
#
# Architecture:
#   sbatch submits a thin "outer" script (no --environment on sbatch itself,
#   as CSCS marks --environment on sbatch as experimental and unreliable for
#   GPU access). The outer script writes a compute "inner" script to a shared
#   path, then runs ONE srun --environment call to execute it inside the
#   container. A single srun = a single container instance = no GPU contention.
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
    GPUS=4        # request all 4 GH200 GPUs; NVLink fabric requires all GPUs
    CPUS=32
    LOG_DIR="${LOG_ROOT}/smoke"
    ;;
minimalrun)
    JOB_NAME="ultatron_minimalrun"
    TIME="00:30:00"
    NODES=1
    GPUS=4
    CPUS=32
    LOG_DIR="${LOG_ROOT}/minimalrun"
    CKPT_DIR="${CKPT_ROOT}/minimalrun"
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

# ── Inner compute script (runs inside the container via srun) ──────────────────
# Written to a shared path accessible on the compute node through the mounts
# defined in ultatron.toml (/users is mounted).
INNERSCRIPT="${LOG_DIR}/.inner_${MODE}.sh"

# ── Outer wrapper script (submitted to sbatch, no --environment) ───────────────
OUTERSCRIPT=$(mktemp /tmp/ultatron_outer_${MODE}_XXXXX.sh)
trap "rm -f ${OUTERSCRIPT}" EXIT

# ── Generate the compute inner script ─────────────────────────────────────────
case "$MODE" in

# ─────────────────────────────────────────────────────── Smoke ────────────────
smoke)
cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"

echo "================================================================"
echo " Ultatron — ${JOB_NAME}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""

export US_BUSI_ROOT=${BUSI_ROOT}
export US_ECHONET_ROOT=${ECHONET_ROOT}
export US_BENIN_ROOT=${BENIN_ROOT}

echo "Running: python3 -m tests.dataset_adapters.training_smoke"
echo "Dataset roots:"
echo "  BUSI    : ${BUSI_ROOT}"
echo "  EchoNet : ${ECHONET_ROOT}"
echo "  Benin   : ${BENIN_ROOT}"
echo ""

python3 -m tests.dataset_adapters.training_smoke

echo ""
echo "================================================================"
echo " SMOKE PASSED  -- \$(date)"
echo "================================================================"
INNER_EOF
;;

# ─────────────────────────────────────────────────── Minimal run ──────────────
minimalrun)
mkdir -p "${CKPT_DIR}"
cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"

# Single-node 4×GH200: use NVLink for P2P + TCP socket for coordination.
# The container toml forces NCCL_NET=AWS Libfabric via LD_LIBRARY_PATH, but
# the CXI/EFA device is not accessible for sbatch-launched containers, so the
# plugin fails. Strip it from LD_LIBRARY_PATH and force Socket transport so
# NCCL uses its built-in TCP rendezvous while still using NVLink for P2P data.
export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export OMP_NUM_THREADS=4

echo "================================================================"
echo " Ultatron — ${JOB_NAME}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""

echo "Running 50-step DDP minimal run (4 GPUs)..."
echo ""

# For initial testing, force a clean start (no auto-resume).
# scripts/train.py will load --ckpt-dir/latest.pt if it exists.
rm -f "${CKPT_DIR}/latest.pt" "${CKPT_DIR}/phase1_end.pt" "${CKPT_DIR}/phase2_end.pt" "${CKPT_DIR}/phase3_end.pt" "${CKPT_DIR}/best.pt" || true

python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/run1/minimal_run1.yaml \
    --ckpt-dir ${CKPT_DIR} \
    ${NO_7B_ARG}

echo ""
echo "================================================================"
echo " MINIMAL RUN PASSED  -- \$(date)"
echo "================================================================"
INNER_EOF
;;

# ──────────────────────────────────────────────────────── Full run1 ───────────
run1)
mkdir -p "${CKPT_DIR}"
cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"

# Single-node 4×GH200: use NVLink for P2P + TCP socket for coordination.
# The container toml forces NCCL_NET=AWS Libfabric via LD_LIBRARY_PATH, but
# the CXI/EFA device is not accessible for sbatch-launched containers, so the
# plugin fails. Strip it from LD_LIBRARY_PATH and force Socket transport so
# NCCL uses its built-in TCP rendezvous while still using NVLink for P2P data.
export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True
export OMP_NUM_THREADS=4

echo "================================================================"
echo " Ultatron — ${JOB_NAME}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""

# SIGUSR1 -> checkpoint + requeue on SLURM pre-emption
_checkpoint_and_requeue() {
    echo "[\$(date)] SIGUSR1: pre-emption — checkpointing..."
    touch ${CKPT_DIR}/.checkpoint_now
    sleep 60
    scontrol requeue "\${SLURM_JOB_ID}"
    echo "[\$(date)] Job requeued."
}
trap _checkpoint_and_requeue SIGUSR1

echo "Checkpoint dir : ${CKPT_DIR}"
echo "Resume         : ${RESUME_ARG:-none}"
echo ""

python3 -m torch.distributed.run \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/experiments/run1.yaml \
    --ckpt-dir ${CKPT_DIR} \
    ${NO_7B_ARG} \
    ${RESUME_ARG}

TRAIN_EXIT=\$?

if [[ \${TRAIN_EXIT} -eq 0 ]]; then
    echo ""
    echo "--- Post-training validation ---"
    python3 train/validate.py \
        --config configs/experiments/run1.yaml \
        --checkpoint ${CKPT_DIR}/latest.pt \
        --mode full \
        --no-7b \
        --output ${LOG_DIR}/validation_\${SLURM_JOB_ID}.json \
    && echo "Validation -> ${LOG_DIR}/validation_\${SLURM_JOB_ID}.json" \
    || echo "Validation failed (non-fatal)"
fi

echo ""
echo "================================================================"
echo " RUN1 COMPLETE  exit=\${TRAIN_EXIT}  \$(date)"
[[ \${TRAIN_EXIT} -ne 0 ]] && echo " Resume: bash scripts/submit_job.sh -run1 --resume ${CKPT_DIR}/latest.pt"
echo "================================================================"
exit \${TRAIN_EXIT}
INNER_EOF
;;
esac

chmod +x "${INNERSCRIPT}"

# ── Generate the outer wrapper (no --environment on sbatch) ───────────────────
cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
# Auto-generated by submit_job.sh — mode=${MODE}
# Thin wrapper: just calls srun once with --environment to enter the container
set -euo pipefail

srun --ntasks-per-node=1 \
     --environment=${EDF_ENV} \
     bash ${INNERSCRIPT}
OUTER_EOF

chmod +x "${OUTERSCRIPT}"

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
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --parsable \
    ${EXTRA_FLAGS} \
    "${OUTERSCRIPT}")

echo ""
echo "  Job ID   : ${JOB_ID}"
echo "  Log      : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Watch    : tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Queue    : squeue -u \$USER -j ${JOB_ID}"

# For run1: print one-liner to chain finetune job after training completes
if [[ "$MODE" == "run1" ]]; then
    echo ""
    echo "  ── Phase 4 finetune ──────────────────────────────────────────────────"
    echo "  Chain finetune after this job:"
    echo "    bash scripts/submit_finetune.sh --after-job ${JOB_ID}"
    echo "  Or run immediately on a saved checkpoint:"
    echo "    bash scripts/submit_finetune.sh --checkpoint ${CKPT_DIR}/phase3_end.pt"
fi
