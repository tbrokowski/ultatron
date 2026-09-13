#!/usr/bin/env bash
# submit_nccl.sh  ·  E6 two-node all-reduce in the Ultatron container
#
# Default: Slingshot (NCCL_NET="AWS Libfabric").  Pass --sockets to compare
# against the Socket fallback (WP1: 152.9 GB/s vs 11.3 GB/s).
set -euo pipefail

REPO_DIR="${ULTATRON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ACCOUNT="${POCUS_ACCOUNT:-${ULTATRON_ACCOUNT:-a127}}"
PARTITION="${ULTATRON_PARTITION:-normal}"
EDF_ENV="${ULTATRON_EDF_ENV:-${HOME}/.edf/ultatron.toml}"
EVIDENCE="${POCUS_EVIDENCE_ROOT:-/capstor/store/cscs/swissai/infra01/meditron-feasibility-review/pocus}"
NODES=2
GPUS_PER_NODE=4
CPUS="${POCUS_CPUS_PER_TASK:-288}"
MODE="slingshot"
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes) NODES="$2"; shift 2 ;;
        --sockets) MODE="sockets"; shift ;;
        --slingshot) MODE="slingshot"; shift ;;
        --after-job) AFTER_JOB="$2"; shift 2 ;;
        *) shift ;;
    esac
done

JOB_NAME="pocus_E6_${MODE}"
LOG_DIR="${EVIDENCE}/nccl"
mkdir -p "${LOG_DIR}" "${REPO_DIR}/logs/pocus"
INNERSCRIPT="$(mktemp "${REPO_DIR}/logs/pocus/inner_e6.XXXXXX.sh")"
OUTERSCRIPT="$(mktemp /tmp/pocus_e6_XXXXX.sh)"
trap "rm -f ${OUTERSCRIPT}" EXIT

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
if [[ "${MODE}" == "sockets" ]]; then
    export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
    export NCCL_NET=Socket
else
    unset NCCL_NET || true
    export NCCL_NET="AWS Libfabric"
    export FI_CXI_ATS=0
fi
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_P2P_LEVEL=NVL
export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500
JOBID="\${SLURM_JOB_ID:-local}"
EVID="${LOG_DIR}/\${JOBID}"
mkdir -p "\${EVID}"
bash "${REPO_DIR}/scripts/pocus/capture_env.sh" "\${EVID}/env.txt"
echo "E6 mode=${MODE} NCCL_NET=\${NCCL_NET} nodes=\${SLURM_NNODES}"
python3 -m torch.distributed.run \\
    --nnodes=\${SLURM_NNODES} \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \\
    --rdzv_id=\${JOBID} \\
    "${REPO_DIR}/scripts/pocus/allreduce_bench.py" --out "\${EVID}/allreduce.json"
INNER_EOF
chmod +x "${INNERSCRIPT}"

cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
set -euo pipefail
srun --ntasks-per-node=1 --environment=${EDF_ENV} bash ${INNERSCRIPT}
OUTER_EOF
chmod +x "${OUTERSCRIPT}"

JOB_ID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --nodes="${NODES}" \
    --ntasks-per-node=1 \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --cpus-per-task="${CPUS}" \
    --time="00:10:00" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --parsable \
    ${AFTER_JOB:+--dependency=afterok:${AFTER_JOB//,/:}} \
    "${OUTERSCRIPT}")
echo "  Job ID : ${JOB_ID}  mode=${MODE}"
echo "JOB_ID=${JOB_ID}"
echo "  Log    : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
