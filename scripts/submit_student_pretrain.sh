#!/usr/bin/env bash
# =============================================================================
# submit_student_pretrain.sh  ·  Student Hiera pretrain (128 GPU default)
# =============================================================================
#
# Full run (100k steps, stages 25k/20k/20k/35k — chain with --resume):
#   bash scripts/submit_student_pretrain.sh
#   bash scripts/submit_student_pretrain.sh --resume
#   bash scripts/submit_student_pretrain.sh --resume --after-job 2575865
#
# 15k pilot (all 4 stages, fixed 512px, separate ckpt dir):
#   bash scripts/submit_student_pretrain.sh --pilot
#
# 16 GPUs:
#   bash scripts/submit_student_pretrain.sh --pilot --nodes 4
#
# Resume:
#   bash scripts/submit_student_pretrain.sh --pilot --resume
#
# Overrides:
#   US_STUDENT_STEPS=50000 bash scripts/submit_student_pretrain.sh
#   US_STUDENT_LR=5e-5 bash scripts/submit_student_pretrain.sh --nodes 4
#   US_STUDENT_RESUME_CKPT=/path/to/step_01500.pt bash scripts/submit_student_pretrain.sh --resume
#   US_STUDENT_RESUME_STAGE_FRACS=checkpoint  # keep checkpoint schedule (legacy runs)
#   US_STUDENT_RESUME_STAGE_FRACS=config       # default: use yaml stage_fracs (EMA stage 4)
#   bash scripts/submit_student_pretrain.sh --resume --after-job JOBID  # chain after Slurm job
#
# Checkpoints:
#   Full  : .../checkpoints/StudentPretrain/
#   Pilot : .../checkpoints/StudentPretrainPilot/
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
LOG_DIR="${REPO_DIR}/logs/pretrain"

PILOT=0
RESUME=0
RESUME_ARGS=""
AFTER_JOB=""
NODES=32
GPUS_PER_NODE=4
CPUS="${POCUS_CPUS_PER_TASK:-288}"
# GH200 nodes expose ~480 GB unified memory; stage-3 resume peaked ~589 GB (job 2544897).
NODE_MEM="475G"
# Slurm partition `normal` MaxTime=12:00:00 — use --resume to chain longer runs.
TIME_LIMIT="12:00:00"
CONFIG="${REPO_DIR}/configs/student/student_pretrain.yaml"
JOB_NAME="ultatron_student_pretrain"
STEPS_LABEL="100k steps"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot)     PILOT=1; shift ;;
        --resume)    RESUME=1; RESUME_ARGS="--resume"; shift ;;
        --after-job) AFTER_JOB="$2"; shift 2 ;;
        --nodes)     NODES="$2"; shift 2 ;;
        --gpus)      GPUS_PER_NODE="$2"; shift 2 ;;
        --single)    NODES=1; GPUS_PER_NODE=4; CPUS="${POCUS_CPUS_PER_TASK:-288}"; shift ;;
        -h|--help)
            sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "${PILOT}" -eq 1 ]]; then
    CONFIG="${REPO_DIR}/configs/student/student_pretrain_pilot.yaml"
    JOB_NAME="ultatron_student_pretrain_pilot"
    STEPS_LABEL="15k pilot"
fi

# Stage-3/4 resume: skip loader warmup + cap workers (OOM / shm, jobs 2543314/2544897/2547540).
STAGE34_RESUME_ENV=""
if [[ "${RESUME}" -eq 1 ]]; then
    STAGE34_RESUME_ENV=$(
        cat <<'STAGE34_EOF'
export US_STUDENT_RESUME=1
export US_STUDENT_LOADER_WARMUP=0
export US_STUDENT_NUM_WORKERS=${US_STUDENT_NUM_WORKERS:-8}
STAGE34_EOF
    )
fi

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
mkdir -p "${LOG_DIR}"

# Per-submit inner script — never reuse a fixed path; a later submit must not
# overwrite the launcher for a job still waiting in the queue.
INNERSCRIPT="$(mktemp "${LOG_DIR}/inner_student_pretrain.XXXXXX.sh")"
OUTERSCRIPT="$(mktemp /tmp/ultatron_outer_student_pretrain_XXXXX.sh)"
trap "rm -f ${OUTERSCRIPT}" EXIT

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
ulimit -c 0
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CSCS srun has no --container-shm-size; avoid DataLoader /dev/shm exhaustion.
export TORCH_SHARED_MEMORY_STRATEGY=file_system

export US_STUDENT_CONFIG="${CONFIG}"
export US_STUDENT_MODE=pretrain
RESUME_ARGS="${RESUME_ARGS}"
${STAGE34_RESUME_ENV}

# Slingshot: keep the aws-ofi-nccl plugin injected by the EDF hook.
# Do NOT set NCCL_NET=Socket and do NOT strip aws-ofi-nccl from LD_LIBRARY_PATH.
unset NCCL_NET || true
export NCCL_NET="AWS Libfabric"
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INIT
export FI_CXI_ATS=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export OMP_NUM_THREADS=8

export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

echo "================================================================"
echo " Ultatron — Student Pretrain (${STEPS_LABEL}, ${TOTAL_GPUS} GPU)"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Nodes  : \${SLURM_NNODES:-1}  GPUs/node: ${GPUS_PER_NODE}"
echo " Config : ${CONFIG}"
echo " Resume : $([ "${RESUME}" -eq 1 ] && echo yes || echo no)"
echo " Mem    : ${NODE_MEM}/node"
if [[ -n "${STAGE34_RESUME_ENV}" ]]; then
echo " Resume : loader_warmup=0 num_workers=${US_STUDENT_NUM_WORKERS:-8} US_STUDENT_RESUME=1"
fi
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""

# Dependencies (pydicom, SimpleITK) are baked into the image — no ensure_deps.sh.
source "${REPO_DIR}/scripts/setup_hf_cache.sh"

python3 -m torch.distributed.run \\
    --nnodes=\${SLURM_NNODES} \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \\
    --rdzv_id=\${SLURM_JOB_ID} \\
    -m train.student_pretrain \${RESUME_ARGS}

echo ""
echo "================================================================"
echo " STUDENT PRETRAIN COMPLETE — \$(date)"
echo "================================================================"
INNER_EOF

chmod +x "${INNERSCRIPT}"

cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
set -euo pipefail
srun --ntasks-per-node=1 \\
     --environment=${EDF_ENV} \\
     bash ${INNERSCRIPT}
OUTER_EOF

chmod +x "${OUTERSCRIPT}"

echo "Submitting ${JOB_NAME} (${NODES} nodes × ${GPUS_PER_NODE} GPU = ${TOTAL_GPUS} total)..."
SBATCH_CMD=(
    sbatch
    --job-name="${JOB_NAME}"
    --nodes="${NODES}"
    --ntasks-per-node=1
    --gpus-per-node="${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS}"
    --mem="${NODE_MEM}"
    --time="${TIME_LIMIT}"
    --partition="${PARTITION}"
    --account="${ACCOUNT}"
    --output="${LOG_DIR}/${JOB_NAME}_%j.out"
    --error="${LOG_DIR}/${JOB_NAME}_%j.err"
    --parsable
)
if [[ -n "${AFTER_JOB}" ]]; then
    SBATCH_CMD+=(--dependency="afterok:${AFTER_JOB}")
fi
JOB_ID=$("${SBATCH_CMD[@]}" "${OUTERSCRIPT}")

echo ""
echo "  Job ID : ${JOB_ID}"
if [[ -n "${AFTER_JOB}" ]]; then
echo "  After  : ${AFTER_JOB} (afterok)"
fi
echo "  GPUs   : ${TOTAL_GPUS} (${NODES} nodes × ${GPUS_PER_NODE})"
echo "  Config : ${CONFIG}"
echo "  Launch : ${INNERSCRIPT}"
echo "  Log    : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Watch  : tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
