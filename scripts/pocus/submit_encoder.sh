#!/usr/bin/env bash
# =============================================================================
# submit_encoder.sh  ·  WP5 Workload A (Ultatron encoder) on Clariden GH200
# =============================================================================
#
# Strong scaling, Slingshot via the aws-ofi-nccl EDF hook.  Do NOT set
# NCCL_NET=Socket and do NOT strip aws-ofi-nccl from LD_LIBRARY_PATH.
#
# Usage:
#   bash scripts/pocus/submit_encoder.sh E0
#   bash scripts/pocus/submit_encoder.sh E1 --nodes 1
#   bash scripts/pocus/submit_encoder.sh E1 --nodes 2 --repeat
#   bash scripts/pocus/submit_encoder.sh E2 --nodes 8
#   bash scripts/pocus/submit_encoder.sh E3 --stage 3 --nodes 4
#   bash scripts/pocus/submit_encoder.sh E4 --nodes 4
#   bash scripts/pocus/submit_encoder.sh E5 --nodes 1
#   bash scripts/pocus/submit_encoder.sh E6            # 2-node NCCL
#
# Env:
#   POCUS_ACCOUNT          [TBC: a127 or infra01]
#   ULTATRON_EDF_ENV       default ~/.edf/ultatron.toml
#   POCUS_EVIDENCE_ROOT    evidence tree (csstaff-readable)
# =============================================================================
set -euo pipefail

REPO_DIR="${ULTATRON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ACCOUNT="${POCUS_ACCOUNT:-${ULTATRON_ACCOUNT:-a127}}"
PARTITION="${ULTATRON_PARTITION:-normal}"
EDF_ENV="${ULTATRON_EDF_ENV:-${HOME}/.edf/ultatron.toml}"
EVIDENCE="${POCUS_EVIDENCE_ROOT:-/capstor/store/cscs/swissai/infra01/meditron-feasibility-review/pocus}"
CONFIG="${REPO_DIR}/configs/pocus/encoder_bench.yaml"
MANIFESTS="${POCUS_MANIFEST_ROOT:-/capstor/store/cscs/swissai/${ACCOUNT}/pocus-bench/manifests}"
GPUS_PER_NODE=4
CPUS="${POCUS_CPUS_PER_TASK:-288}"   # [TBC per CSCS guidance] all GH200 node cores
NODE_MEM="475G"
TIME_LIMIT="01:00:00"
NUM_WORKERS="${US_STUDENT_NUM_WORKERS:-8}"
SEED=1234

EXP="${1:?experiment id: E0|E1|E2|E3|E4|E5|E6}"
shift || true

NODES=1
REPEAT=0
STAGE=""
AFTER_JOB=""
GSSR=0
USER_TIME=0
GPUS_PER_NODE=4
FULL_NODE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes) NODES="$2"; shift 2 ;;
        --gpus|--gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
        --full-node) FULL_NODE=1; GPUS_PER_NODE=4; shift ;;
        --repeat) REPEAT=1; shift ;;
        --stage) STAGE="$2"; shift 2 ;;
        --after-job) AFTER_JOB="$2"; shift 2 ;;
        --gssr) GSSR=1; shift ;;
        --time) TIME_LIMIT="$2"; USER_TIME=1; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${EXP}" in
    E0)
        NODES=1
        if [[ "${FULL_NODE}" -eq 0 && "${GPUS_PER_NODE}" -eq 4 ]]; then
            GPUS_PER_NODE=1
        fi
        [[ "${USER_TIME}" -eq 0 ]] && TIME_LIMIT="00:20:00"
        STAGE="${STAGE:-2}"
        ;;
    E1) STAGE="${STAGE:-1}"; [[ "${USER_TIME}" -eq 0 ]] && TIME_LIMIT="00:45:00" ;;
    E2) STAGE="${STAGE:-2}"; [[ "${USER_TIME}" -eq 0 ]] && TIME_LIMIT="01:00:00" ;;
    E3) STAGE="${STAGE:-3}"; [[ "${USER_TIME}" -eq 0 ]] && TIME_LIMIT="00:30:00" ;;
    E4) [[ "${USER_TIME}" -eq 0 ]] && TIME_LIMIT="00:15:00" ;;
    E5) NODES=1; [[ "${USER_TIME}" -eq 0 ]] && TIME_LIMIT="00:30:00" ;;
    E6)
        exec bash "${REPO_DIR}/scripts/pocus/submit_nccl.sh" --nodes 2 "$@"
        ;;
    *) echo "[ERROR] unknown experiment ${EXP}" >&2; exit 1 ;;
esac

# GSSR on the largest scale of every series, plus every run where cheap.
if [[ "${NODES}" -ge 8 || "${GSSR}" -eq 1 || "${EXP}" == "E0" ]]; then
    GSSR=1
fi

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
JOB_NAME="pocus_${EXP}_${NODES}n"
LOG_DIR="${EVIDENCE}/encoder"
mkdir -p "${LOG_DIR}" "${REPO_DIR}/logs/pocus"

INNERSCRIPT="$(mktemp "${REPO_DIR}/logs/pocus/inner_${EXP}.XXXXXX.sh")"
OUTERSCRIPT="$(mktemp /tmp/pocus_outer_XXXXX.sh)"
trap "rm -f ${OUTERSCRIPT}" EXIT

WORKLOAD="encoder"
case "${EXP}" in
    E1) WORKLOAD="A-stage1" ;;
    E2) WORKLOAD="A-stage2" ;;
    E3) WORKLOAD="A-stage${STAGE:-3}" ;;
    E0) WORKLOAD="E0" ;;
    E4) WORKLOAD="E4" ;;
    E5) WORKLOAD="E5" ;;
esac

EXTRA_FLAGS=""
case "${EXP}" in
    E0)
        # MBS probe with accum=1 (GBS = MBS × n_gpus). Spec: largest MBS that fits.
        EXTRA_FLAGS="--bench-stage ${STAGE} --per-step-timing --no-ckpt --num-workers ${NUM_WORKERS}"
        ;;
    E1)
        EXTRA_FLAGS="--bench-stage ${STAGE} --bench-window --per-step-timing --no-ckpt --forced-type image --num-workers ${NUM_WORKERS} --seed ${SEED}"
        ;;
    E2|E3)
        EXTRA_FLAGS="--bench-stage ${STAGE} --bench-window --per-step-timing --no-ckpt --num-workers ${NUM_WORKERS} --seed ${SEED}"
        ;;
    E4)
        EXTRA_FLAGS="--ckpt-probe --per-step-timing --max-steps 2 --num-workers ${NUM_WORKERS}"
        ;;
    E5)
        EXTRA_FLAGS="--loader-only --max-steps 200 --num-workers ${NUM_WORKERS} --seed ${SEED}"
        ;;
esac

# Image / video manifests.
IMG_MAN="${MANIFESTS}/enc_images.jsonl"
VID_MAN="${MANIFESTS}/enc_videos.jsonl"
if [[ "${EXP}" == "E1" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --manifest ${IMG_MAN}"
elif [[ "${EXP}" != "E5" ]]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --manifest ${IMG_MAN} --video-manifest ${VID_MAN}"
fi

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
ulimit -c 0
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_SHARED_MEMORY_STRATEGY=file_system
export HF_HUB_OFFLINE=1
export US_STUDENT_CONFIG="${CONFIG}"
export US_STUDENT_MODE=pretrain
export US_STUDENT_NUM_WORKERS=${NUM_WORKERS}
export POCUS_EVIDENCE_ROOT="${EVIDENCE}"

# Slingshot: keep aws-ofi-nccl (injected by the EDF hook). Never Socket.
unset NCCL_NET || true
export NCCL_NET="AWS Libfabric"
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=INIT
export NCCL_DEBUG_SUBSYS=INIT,NET
export FI_CXI_ATS=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export OMP_NUM_THREADS=8

export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

JOBID="\${SLURM_JOB_ID:-local}"
EVID="${LOG_DIR}/\${JOBID}"
mkdir -p "\${EVID}/gssr"
export US_STUDENT_LOG_DIR="\${EVID}"
export US_STUDENT_STEPS_JSONL="\${EVID}/steps.jsonl"
export POCUS_EXPERIMENT="${EXP}"
export EVID
python3 - << PY
import json, os, pathlib
p = pathlib.Path(os.environ["US_STUDENT_LOG_DIR"]) / "run.json"
p.write_text(json.dumps({
    "experiment": "${EXP}",
    "workload": "${WORKLOAD}",
    "nodes": ${NODES},
    "n_gpus": ${TOTAL_GPUS},
    "gpus_per_node": ${GPUS_PER_NODE},
    "jobid": os.environ.get("SLURM_JOB_ID", "local"),
    "config": "${CONFIG}",
}, indent=2) + "\n")
PY

source "${REPO_DIR}/scripts/gssr_sidecar.sh"
source "${REPO_DIR}/scripts/setup_hf_cache.sh"
bash "${REPO_DIR}/scripts/pocus/capture_env.sh" "\${EVID}/env.txt" "${CONFIG}"

if [[ "${GSSR}" -eq 1 ]]; then
    gssr_sidecar_start "\${EVID}"
    trap gssr_sidecar_stop EXIT
fi

echo "================================================================"
echo " POCUS ${EXP}  nodes=${NODES} gpus=${TOTAL_GPUS}  Slingshot/OFI"
echo " Job    : \${JOBID}"
echo " Node   : \$(hostname)"
echo " Config : ${CONFIG}"
echo " Flags  : ${EXTRA_FLAGS}"
echo " NCCL_NET=\${NCCL_NET}"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "================================================================"

run_pretrain() {
    python3 -m torch.distributed.run \\
        --nnodes=\${SLURM_NNODES} \\
        --nproc_per_node=${GPUS_PER_NODE} \\
        --rdzv_backend=c10d \\
        --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \\
        --rdzv_id=\${JOBID} \\
        -m train.student_pretrain "\$@"
}

if [[ "${EXP}" == "E0" ]]; then
    set +e
    for MBS in 16 32 40 48; do
        GBS=\$((MBS * ${GPUS_PER_NODE}))
        echo "=== E0 image MBS=\${MBS} GBS=\${GBS} (accum=1) ==="
        run_pretrain ${EXTRA_FLAGS} --image-mbs \${MBS} --gbs-img \${GBS} --max-steps 4 --forced-type image
        rc=\$?
        echo "MBS=\${MBS} gpus=${GPUS_PER_NODE} rc=\${rc}" | tee -a "\${EVID}/mbs_probe.txt"
        if [[ \${rc} -ne 0 ]]; then
            echo "E0 stopped at MBS=\${MBS} (rc=\${rc}) — treat as OOM/fail"
            break
        fi
    done
    set -e
elif [[ "${EXP}" == "E5" ]]; then
    export POCUS_LOADER_TAG=loader_only_images
    run_pretrain ${EXTRA_FLAGS} --forced-type image --manifest ${IMG_MAN}
    export POCUS_LOADER_TAG=loader_only_videos
    run_pretrain ${EXTRA_FLAGS} --forced-type video --manifest ${IMG_MAN} --video-manifest ${VID_MAN}
    for ds in CardiacUDC COVID-BLUES IUGC2024; do
        man="${MANIFESTS}/enc_videos_\${ds}.jsonl"
        if [[ -f "\${man}" ]]; then
            export POCUS_LOADER_TAG=loader_only_\${ds}
            run_pretrain ${EXTRA_FLAGS} --forced-type video --manifest "\${man}" --video-manifest "\${man}"
        fi
    done
else
    run_pretrain ${EXTRA_FLAGS}
fi

# Copy Slurm logs next to evidence when they exist.
if [[ -f "${LOG_DIR}/${JOB_NAME}_\${JOBID}.out" ]]; then
    cp -f "${LOG_DIR}/${JOB_NAME}_\${JOBID}.out" "\${EVID}/" || true
    cp -f "${LOG_DIR}/${JOB_NAME}_\${JOBID}.err" "\${EVID}/" || true
fi
cp -f "${CONFIG}" "\${EVID}/config.yaml" || true

echo " POCUS ${EXP} COMPLETE — \$(date)"
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

echo "Submitting ${JOB_NAME} (${NODES} nodes × ${GPUS_PER_NODE} GPU = ${TOTAL_GPUS}) via Slingshot..."
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
echo "  Job ID : ${JOB_ID}"
echo "  Log    : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Evidence: ${LOG_DIR}/${JOB_ID}/"

if [[ "${REPEAT}" -eq 1 ]]; then
    echo "Submitting repeat..."
    bash "$0" "${EXP}" --nodes "${NODES}" --time "${TIME_LIMIT}" ${STAGE:+--stage ${STAGE}} --after-job "${JOB_ID}"
fi
