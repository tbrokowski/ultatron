#!/usr/bin/env bash
# =============================================================================
# submit_masking_ablation_resume.sh  ·  Resume failed masking ablation runs
# =============================================================================
#
# Resumes freq / spatial / both from step_01000.pt in each variant ckpt dir.
# Applies memory-safe config overrides so ALP tier-2 never triggers (tier-1
# only for the whole ablation — appropriate for stage-1 masking comparison).
#
# Usage:
#   # Preview only (default):
#   bash scripts/submit_masking_ablation_resume.sh
#
#   # Submit all three resume jobs:
#   bash scripts/submit_masking_ablation_resume.sh --submit
#
#   # Point at a specific prior sweep artifact dir:
#   bash scripts/submit_masking_ablation_resume.sh --submit \
#       --run-dir logs/pretrain/masking_ablation_20260616T164317Z_masking_ablation
#
# Checkpoints read/written:
#   /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/MaskingAblation/{freq,spatial,both}/
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
LOG_ROOT="${REPO_DIR}/logs/pretrain"
CKPT_ROOT="/capstor/store/cscs/swissai/a127/ultrasound/checkpoints/MaskingAblation"
DEFAULT_RUN_DIR="${LOG_ROOT}/masking_ablation_20260616T164317Z_masking_ablation"

NODES=8
GPUS_PER_NODE=4
CPUS=32
NODE_MEM="460G"
TIME_LIMIT="08:00:00"
ABLATION_STEPS=5000
IMAGE_BATCH_SIZE=24
NUM_WORKERS=2

SUBMIT=0
DRY_RUN=0
RUN_DIR="${DEFAULT_RUN_DIR}"
VARIANTS=("freq" "spatial" "both")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)    SUBMIT=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --run-dir)   RUN_DIR="$2"; shift 2 ;;
        --nodes)     NODES="$2"; shift 2 ;;
        --time)      TIME_LIMIT="$2"; shift 2 ;;
        --batch)     IMAGE_BATCH_SIZE="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
RESUME_ID="$(date -u +%Y%m%dT%H%M%SZ)_masking_ablation_resume"
RESUME_DIR="${RUN_DIR}/resume_${RESUME_ID}"
CONFIG_DIR="${RESUME_DIR}/configs"

echo "================================================================"
echo " Ultatron — Masking Ablation RESUME (from step 1000 → ${ABLATION_STEPS})"
echo " Resume ID : ${RESUME_ID}"
echo " Prior run : ${RUN_DIR}"
echo " Variants  : ${VARIANTS[*]}"
echo " Resources : ${NODES} nodes × ${GPUS_PER_NODE} GPU = ${TOTAL_GPUS} total"
echo " Time      : ${TIME_LIMIT}"
echo " Batch     : ${IMAGE_BATCH_SIZE}  workers: ${NUM_WORKERS}"
echo " Artifacts : ${RESUME_DIR}"
echo " Submit    : ${SUBMIT}"
echo " Dry-run   : ${DRY_RUN}"
echo "================================================================"
echo ""

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[ERROR] Prior run dir not found: ${RUN_DIR}" >&2
    exit 1
fi

for variant in "${VARIANTS[@]}"; do
    ckpt="${CKPT_ROOT}/${variant}/step_01000.pt"
    if [[ ! -f "${ckpt}" ]]; then
        echo "[ERROR] Missing checkpoint for ${variant}: ${ckpt}" >&2
        exit 1
    fi
    echo "  OK  ${variant}  $(ls -lh "${ckpt}" | awk '{print $5, $6, $7, $8}')"
done
echo ""

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] Would write resume configs + sbatch scripts under ${RESUME_DIR}"
    exit 0
fi

mkdir -p "${CONFIG_DIR}"

generate_resume_config() {
    local variant="$1"
    local base_config="$2"
    local config_out="$3"
    local ckpt_dir="$4"

    python3 - "${base_config}" "${variant}" "${config_out}" "${ckpt_dir}" \
        "${ABLATION_STEPS}" "${IMAGE_BATCH_SIZE}" "${NUM_WORKERS}" <<'PYEOF'
import sys, copy, yaml
from pathlib import Path

(base_path, variant, out_path, ckpt_dir,
 steps_str, batch_str, workers_str) = sys.argv[1:8]
steps = int(steps_str)
batch = int(batch_str)
workers = int(workers_str)

with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def deep_merge(base, override):
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base

# Memory-safe resume overrides:
# - alp_stage_fracs [1.0, 1.0]: never unlock tier-2/3 (OOM trigger at step 1250)
# - smaller batch + fewer workers: headroom on unified-memory GH200 nodes
resume_overrides = {
    "training": {
        "total_steps": steps,
        "stage_fracs": [1.0, 0.0, 0.0, 0.0],
    },
    "curriculum": {
        "total_training_steps": steps,
        # frac >= 1.0 → stay ALP stage 1 (tier-1 pool, mask_ratio 0.40) entire run
        "alp_stage_fracs": [1.0, 1.0],
    },
    "loaders": {
        "image_batch_size": batch,
        "num_workers": workers,
    },
    "student_data": {
        "image_batch_size": batch,
        "num_workers": workers,
    },
    "pretrain": {
        "ckpt_dir": ckpt_dir,
        "save_stage_end_ckpts": False,
    },
}

deep_merge(cfg, resume_overrides)

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Generated by submit_masking_ablation_resume.sh\n")
    f.write(f"# Resume variant: {variant}  from step_01000.pt\n")
    f.write(f"# Source config: {base_path}\n")
    f.write("# ALP tier-2 disabled (alp_stage_fracs [1.0, 1.0]) to avoid OOM\n")
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

print(f"  Wrote resume config: {out_path}")
PYEOF
}

echo "--- Generating resume configs ---"
for variant in "${VARIANTS[@]}"; do
    base_config="${RUN_DIR}/configs/mask_${variant}.yaml"
    if [[ ! -f "${base_config}" ]]; then
        echo "[ERROR] Base config missing: ${base_config}" >&2
        exit 1
    fi
    config_path="${CONFIG_DIR}/mask_${variant}_resume.yaml"
    ckpt_dir="${CKPT_ROOT}/${variant}"
    generate_resume_config "${variant}" "${base_config}" "${config_path}" "${ckpt_dir}"
done
echo ""

echo "--- Generating resume SLURM scripts ---"
job_ids=()
RESUME_TSV="${RESUME_DIR}/resume_jobs.tsv"
printf 'resume_id\tvariant\tjob_id\tconfig_path\tckpt_dir\tsbatch_path\tsubmitted_at_utc\n' \
    > "${RESUME_TSV}"

for variant in "${VARIANTS[@]}"; do
    config_path="${CONFIG_DIR}/mask_${variant}_resume.yaml"
    ckpt_dir="${CKPT_ROOT}/${variant}"
    inner_path="${RESUME_DIR}/inner_mask_${variant}_resume.sh"
    outer_path="${RESUME_DIR}/outer_mask_${variant}_resume.sbatch"
    job_name="ultatron_mask_${variant}_resume"

    cat > "${inner_path}" << INNER_EOF
#!/bin/bash
set -euo pipefail
ulimit -c 0
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export US_STUDENT_CONFIG="${config_path}"
export US_STUDENT_MODE=pretrain
export US_STUDENT_RESUME=1

export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export OMP_NUM_THREADS=8

export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

echo "================================================================"
echo " Ultatron — Masking Ablation RESUME: ${variant}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Nodes  : \${SLURM_NNODES:-1}  GPUs/node: ${GPUS_PER_NODE}"
echo " Config : ${config_path}"
echo " Ckpt   : ${ckpt_dir}  (resume from step_01000.pt)"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""

bash "${REPO_DIR}/scripts/ensure_deps.sh"
source "${REPO_DIR}/scripts/setup_hf_cache.sh"

python3 -m torch.distributed.run \\
    --nnodes=\${SLURM_NNODES} \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \\
    --rdzv_id=\${SLURM_JOB_ID} \\
    -m tests.dataset_adapters.student_training_smoke \\
    --resume

echo ""
echo "================================================================"
echo " MASKING ABLATION RESUME (${variant}) COMPLETE — \$(date)"
echo "================================================================"
INNER_EOF
    chmod +x "${inner_path}"

    cat > "${outer_path}" << OUTER_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${NODE_MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${RESUME_DIR}/${job_name}_%j.out
#SBATCH --error=${RESUME_DIR}/${job_name}_%j.err
set -euo pipefail
srun --ntasks-per-node=1 \\
     --environment=${EDF_ENV} \\
     bash ${inner_path}
OUTER_EOF
    chmod +x "${outer_path}"

    echo "  variant=${variant}"
    echo "    config : ${config_path}"
    echo "    sbatch : ${outer_path}"
    echo "    ckpt   : ${ckpt_dir}/step_01000.pt"

    job_id=""
    if [[ "${SUBMIT}" -eq 1 ]]; then
        job_id="$(sbatch --parsable "${outer_path}")"
        job_ids+=("${job_id}")
        echo "    job_id : ${job_id}"
    else
        echo "    submit : sbatch ${outer_path}"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${RESUME_ID}" "${variant}" "${job_id}" \
        "${config_path}" "${ckpt_dir}" "${outer_path}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "${RESUME_TSV}"
    echo ""
done

echo "================================================================"
echo " Resume setup complete"
echo "  Resume dir : ${RESUME_DIR}"
echo "  Jobs TSV   : ${RESUME_TSV}"
if [[ "${SUBMIT}" -eq 1 && "${#job_ids[@]}" -gt 0 ]]; then
    echo "  Submitted  : ${job_ids[*]}"
    echo ""
    echo "After jobs finish, compare results:"
    echo "  python3 ${RUN_DIR}/compare_masking.py ${RUN_DIR}/jobs.tsv ${RUN_DIR}"
else
    echo ""
    echo "Submit all three:"
    echo "  bash scripts/submit_masking_ablation_resume.sh --submit"
fi
echo "================================================================"
