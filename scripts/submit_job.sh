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
#   -run1         Full run1 training on multi-node GH200 (see NODES in script).
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

# ── Cardiac dataset roots ─────────────────────────────────────────────────────
CAMUS_ROOT="${STORE}/cardiac/CAMUS"
ECHONET_ROOT="${STORE}/cardiac/EchoNet-Dynamic"
ECHONET_PED_ROOT="${STORE}/cardiac/EchoNet-Pediatric"
ECHONET_LVH_ROOT="${STORE}/cardiac/EchoNet-LVH"
MIMIC_ECHO_ROOT="${STORE}/cardiac/MIMIC-IV-Echo"
MIMIC_LVVOL_ROOT="${STORE}/cardiac/MIMIC-IV-Echo-LVVol-A4C"
TED_ROOT="${STORE}/cardiac/TED"
UNITY_ROOT="${STORE}/cardiac/Unity"
CARDIACUDC_ROOT="${STORE}/cardiac/CardiacUDC"
ECHOCP_ROOT="${STORE}/cardiac/EchoCP"

# ── Non-cardiac dataset roots ─────────────────────────────────────────────────
BUSI_ROOT="${STORE}/breast/BUSI"
BENIN_ROOT="${STORE}/lung/Benin_Videos"
RSA_ROOT="${STORE}/lung/RSA_Videos"

# ── Manifest / scratch paths ──────────────────────────────────────────────────
MANIFEST_DIR="${CKPT_ROOT%/*}/manifests"

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
    NODES=8
    GPUS=4        # per node (4 GH200 per node × NODES = 32 GPUs total)
    CPUS=64       # 16 per GPU × 4 GPUs per node
    LOG_DIR="${LOG_ROOT}/run1"
    CKPT_DIR="${CKPT_ROOT}/run1"
    MANIFEST_DIR="${LOG_DIR}"
    MANIFEST_PATH="${MANIFEST_DIR}/run1_train.jsonl"
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

export US_CAMUS_ROOT="${CAMUS_ROOT}"
export US_ECHONET_ROOT="${ECHONET_ROOT}"
export US_ECHONET_PED_ROOT=${ECHONET_PED_ROOT}
export US_ECHONET_LVH_ROOT=${ECHONET_LVH_ROOT}
export US_MIMIC_ECHO_ROOT=${MIMIC_ECHO_ROOT}
export US_MIMIC_LVVOL_ROOT=${MIMIC_LVVOL_ROOT}
export US_TED_ROOT=${TED_ROOT}
export US_UNITY_ROOT=${UNITY_ROOT}
export US_CARDIACUDC_ROOT=${CARDIACUDC_ROOT}
export US_ECHOCP_ROOT=${ECHOCP_ROOT}
export US_BUSI_ROOT=${BUSI_ROOT}
export US_BENIN_ROOT=${BENIN_ROOT}
export US_RSA_ROOT=${RSA_ROOT}

pip install --quiet pydicom

echo "Running: python3 -m tests.dataset_adapters.training_smoke"
echo "Dataset roots:"
echo "  CAMUS              : ${CAMUS_ROOT}"
echo "  EchoNet-Dynamic    : ${ECHONET_ROOT}"
echo "  EchoNet-Pediatric  : ${ECHONET_PED_ROOT}"
echo "  EchoNet-LVH        : ${ECHONET_LVH_ROOT}"
echo "  MIMIC-IV-ECHO      : ${MIMIC_ECHO_ROOT}"
echo "  MIMIC-IV-LVVol-A4C : ${MIMIC_LVVOL_ROOT}"
echo "  TED                : ${TED_ROOT}"
echo "  Unity-Echo         : ${UNITY_ROOT}"
echo "  CardiacUDC         : ${CARDIACUDC_ROOT}"
echo "  EchoCP             : ${ECHOCP_ROOT}"
echo "  BUSI               : ${BUSI_ROOT}"
echo "  Benin-LUS          : ${BENIN_ROOT}"
echo "  RSA-LUS            : ${RSA_ROOT}"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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

pip install --quiet pydicom

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

# Multi-node GH200 (4 GPUs/node; NODES set above) on CSCS Alps / Slingshot-11.
# - Strip aws-ofi-nccl: the AWS OFI plugin requires CXI device access that is
#   not available inside the sbatch-launched container, causing ncclInvalidUsage.
# - Force NCCL_NET=Socket so NCCL uses its built-in TCP transport for cross-node
#   gradient sync (no external plugin needed).  Intra-node traffic still goes
#   over NVLink (NCCL_P2P_LEVEL=NVL), so per-node compute efficiency is preserved.
# - Revisit proper Slingshot OFI transport (FI_CXI_ATS=0 + keep aws-ofi-nccl)
#   once the CXI device is confirmed accessible in the container image.
export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Rendezvous: rank-0 node is the master
export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500
echo "MASTER_ADDR : \${MASTER_ADDR}"
echo "SLURM nodes : \${SLURM_NNODES}  (node \${SLURM_NODEID:-?} of \${SLURM_NNODES})"

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

pip install --quiet pydicom

echo "Checkpoint dir : ${CKPT_DIR}"
echo "Resume         : ${RESUME_ARG:-none}"
echo ""

# ── Build manifest (rank-0 only; idempotent — skip if already exists) ─────────
mkdir -p "${MANIFEST_DIR}"
if [[ ! -f "${MANIFEST_PATH}" || -n "\${FORCE_REBUILD_MANIFEST:-}" ]]; then
    echo "Building run1 manifest → ${MANIFEST_PATH}"
    python3 scripts/build_manifest.py \
        --config configs/run1/data_run1.yaml \
        --out "${MANIFEST_PATH}" \
        || { echo "ERROR: manifest build failed"; exit 1; }
    echo "Manifest built: \$(wc -l < ${MANIFEST_PATH}) entries"
else
    echo "Reusing existing manifest: ${MANIFEST_PATH} (\$(wc -l < ${MANIFEST_PATH}) entries)"
fi
echo ""

python3 -m torch.distributed.run \
    --nnodes=\${SLURM_NNODES} \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \
    --rdzv_id=\${SLURM_JOB_ID} \
    scripts/train.py \
    --config configs/experiments/run1.yaml \
    --manifest "${MANIFEST_PATH}" \
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
