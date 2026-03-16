#!/bin/bash
# ============================================================================
# run_training_job.sh  ·  Ultatron foundation model — CSCS Alps SLURM submission
# ============================================================================
#
# Submits the full 4-phase Ultatron training job to CSCS Alps (GH200 nodes).
# Handles pre-flight checks, data staging, model weight caching, and
# clean post-training archival to Store.
#
# Usage:
#   sbatch run_training_job.sh
#   sbatch run_training_job.sh --resume /path/to/checkpoint.pt
#   sbatch run_training_job.sh --phase 3   (resume at specific phase)
#   sbatch run_training_job.sh --no-7b     (skip 7B teacher, saves ~7GB/node)
#
# Requirements:
#   - HuggingFace token set in environment ($HF_TOKEN) or
#     `huggingface-cli login` run from login node before submitting.
#   - DINOv3 and V-JEPA2 gated model agreements accepted on HF Hub.
#   - All datasets staged to Scratch (run stage_datasets.sh first).
#   - Conda/mamba environment `oura` installed (see environment.yaml).
#
# CSCS Alps node spec (GH200):
#   4× NVIDIA GH200 SXM (96 GB HBM3) per node
#   InfiniBand NDR 400Gb/s inter-node
#   Lustre scratch: /capstor/scratch/cscs/$USER
#   Permanent store: /capstor/store/cscs/swissai/a127
# ============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=oura_train
#SBATCH --nodes=512
#SBATCH --ntasks-per-node=4          # 1 task per GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8            # 8 CPU workers per dataloader
#SBATCH --mem=460G                   # ~115G per GPU slot (GH200 has 480G total)
#SBATCH --time=6-00:00:00            # 6 days wall-clock
#SBATCH --partition=normal
#SBATCH --account=a127               # Swiss AI project account
#SBATCH --output=%x_%j.out           # oura_train_<jobid>.out
#SBATCH --error=%x_%j.err
#SBATCH --signal=SIGUSR1@120         # 2-min pre-emption warning → checkpoint

# ── Parse optional arguments ──────────────────────────────────────────────────
RESUME_CKPT=""
PHASE_ARG=""
NO_7B_ARG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)    RESUME_CKPT="--resume $2";  shift 2 ;;
        --phase)     PHASE_ARG="--phase $2";     shift 2 ;;
        --no-7b)     NO_7B_ARG="--no-7b";        shift ;;
        *)           echo "Unknown arg: $1"; shift ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_DIR="/capstor/store/cscs/swissai/a127/ultrasound/code/ultatron"
SCRATCH_DIR="/capstor/scratch/cscs/${USER}/ultrasound"
CONFIG="${PROJECT_DIR}/configs/data_config.yaml"
LOG_DIR="${SCRATCH_DIR}/logs/job_${SLURM_JOB_ID}"
CKPT_DIR="${SCRATCH_DIR}/checkpoints/current_run"
HF_CACHE="${SCRATCH_DIR}/hf_cache"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}" "${HF_CACHE}"

echo "================================================================"
echo " Oura Foundation Model Training"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Nodes    : ${SLURM_NNODES}"
echo " Tasks    : ${SLURM_NTASKS}"
echo " Start    : $(date)"
echo " Scratch  : ${SCRATCH_DIR}"
echo " Log dir  : ${LOG_DIR}"
echo "================================================================"

# ── Conda environment activation ──────────────────────────────────────────────
# CSCS uses module system; activate via conda run or source activate
source /capstor/store/cscs/swissai/a127/envs/oura/bin/activate 2>/dev/null || \
    conda activate oura 2>/dev/null || \
    { echo "ERROR: conda environment 'oura' not found. Run setup_env.sh first."; exit 1; }

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    $(python -c 'import torch; print(torch.version.cuda)')"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo ""
echo "--- Pre-flight checks ---"

# Check HuggingFace token
if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_CACHE_TOKEN="${HOME}/.cache/huggingface/token"
    if [[ ! -f "${HF_CACHE_TOKEN}" ]]; then
        echo "ERROR: HF_TOKEN not set and ~/.cache/huggingface/token not found."
        echo "       Run: huggingface-cli login"
        echo "       Or:  export HF_TOKEN=hf_..."
        exit 1
    fi
    echo "  HF token: found at ${HF_CACHE_TOKEN}"
else
    echo "  HF token: set via environment"
fi

# Check manifest exists
MANIFEST="${SCRATCH_DIR}/manifests/us_foundation_train.jsonl"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "ERROR: Training manifest not found: ${MANIFEST}"
    echo "       Run stage_datasets.sh first."
    exit 1
fi
echo "  Manifest: ${MANIFEST} ($(wc -l < ${MANIFEST}) entries)"

# Check scratch quota
echo "  Scratch quota:"
lfs quota -u "${USER}" /capstor/scratch/ 2>/dev/null | head -4 || df -h "${SCRATCH_DIR}"

echo ""

# ── Distributed setup ─────────────────────────────────────────────────────────
# CSCS uses PMIx; torchrun is the preferred launcher.
# MASTER_ADDR and MASTER_PORT are derived from SLURM_NODELIST.
export MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE="${SLURM_NTASKS}"

# NCCL settings tuned for Alps InfiniBand NDR
export NCCL_SOCKET_IFNAME=hsn0,hsn1        # Alps high-speed network interfaces
export NCCL_NET_GDR_LEVEL=PIX              # GPU Direct RDMA
export NCCL_ALGO=Ring                      # Ring all-reduce (optimal for large N)
export NCCL_PROTO=Simple
export NCCL_BUFFSIZE=16777216              # 16 MB
export NCCL_IB_TIMEOUT=22
export NCCL_DEBUG=WARN                     # INFO is verbose; WARN for production

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True"
export TOKENIZERS_PARALLELISM=false        # suppress HF tokenizer warning
export OMP_NUM_THREADS=4

# HuggingFace
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"

echo "--- Distributed setup ---"
echo "  MASTER_ADDR : ${MASTER_ADDR}"
echo "  MASTER_PORT : ${MASTER_PORT}"
echo "  WORLD_SIZE  : ${WORLD_SIZE}"
echo ""

# ── Pre-cache HuggingFace model weights ───────────────────────────────────────
# Download model weights to shared Store before launching training so all
# nodes read from fast local cache, not HF Hub during training.
# Only run on rank 0 (first task on the first node).
if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    echo "--- Pre-caching HuggingFace models ---"
    python - << 'PYEOF'
import os, sys
from pathlib import Path

hf_cache = os.environ.get("HF_HOME", "/tmp/hf_cache")

models = [
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "facebook/vjepa2-vitl-fpc64-256",
]
if os.environ.get("NO_7B_ARG", "") == "":
    models.append("facebook/dinov3-vit7b16-pretrain-lvd1689m")

from transformers import AutoModel
for model_id in models:
    cache_path = Path(hf_cache) / "models--" + model_id.replace("/", "--")
    if cache_path.exists() and any(cache_path.iterdir()):
        print(f"  Already cached: {model_id}")
        continue
    print(f"  Downloading: {model_id} ...", flush=True)
    try:
        AutoModel.from_pretrained(model_id, cache_dir=hf_cache)
        print(f"  Cached: {model_id}")
    except Exception as e:
        print(f"  WARNING: Failed to pre-cache {model_id}: {e}")
        print(f"  Will attempt download during training.")
PYEOF
    echo "Pre-caching complete."
fi

# ── SIGUSR1 handler: checkpoint on pre-emption ────────────────────────────────
# When SLURM sends SIGUSR1 (2 min before timeout), trigger a checkpoint
# and requeue the job automatically.
_checkpoint_and_requeue() {
    echo "[$(date)] SIGUSR1 received — checkpointing and requeueing ..."
    # Signal the Python process to checkpoint (it watches for this file)
    touch "${CKPT_DIR}/.checkpoint_now"
    sleep 30
    scontrol requeue "${SLURM_JOB_ID}"
    echo "[$(date)] Job requeued as ${SLURM_JOB_ID}"
}
trap _checkpoint_and_requeue SIGUSR1

# ── Auto-resume detection ─────────────────────────────────────────────────────
# If no explicit --resume given but a latest.pt exists, auto-resume.
if [[ -z "${RESUME_CKPT}" && -f "${CKPT_DIR}/latest.pt" ]]; then
    RESUME_CKPT="--resume ${CKPT_DIR}/latest.pt"
    echo "Auto-resuming from ${CKPT_DIR}/latest.pt"
fi

# ── Launch training ───────────────────────────────────────────────────────────
echo ""
echo "--- Launching training ---"
echo "  Command: srun torchrun ... train.py ${RESUME_CKPT} ${PHASE_ARG} ${NO_7B_ARG}"
echo ""

srun \
    --output="${LOG_DIR}/rank_%t.log" \
    --error="${LOG_DIR}/rank_%t.err" \
    bash -c "
        torchrun \
            --nnodes=${SLURM_NNODES} \
            --nproc_per_node=4 \
            --node_rank=\${SLURM_NODEID} \
            --master_addr=${MASTER_ADDR} \
            --master_port=${MASTER_PORT} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            ${PROJECT_DIR}/train.py \
                --config ${CONFIG} \
                ${RESUME_CKPT} \
                ${PHASE_ARG} \
                ${NO_7B_ARG} \
        2>&1 | tee ${LOG_DIR}/train_rank_\${SLURM_NODEID}.log
    "

TRAIN_EXIT=$?
echo ""
echo "Training process exited with code: ${TRAIN_EXIT}"

# ── Post-training: archive to Store ──────────────────────────────────────────
if [[ ${TRAIN_EXIT} -eq 0 ]]; then
    echo ""
    echo "--- Archiving checkpoints to Store ---"
    STORE_CKPT="/capstor/store/cscs/swissai/a127/ultrasound/checkpoints/job_${SLURM_JOB_ID}"
    mkdir -p "${STORE_CKPT}"
    rsync -avh --progress "${CKPT_DIR}/" "${STORE_CKPT}/"
    echo "  Archived → ${STORE_CKPT}"

    echo ""
    echo "--- Archiving ALP cache ---"
    rsync -avh "${SCRATCH_DIR}/alp_cache/" \
               "/capstor/store/cscs/swissai/a127/ultrasound/alp_cache/" \
               2>/dev/null || echo "  ALP cache not found; skipping."

    echo ""
    echo "--- Running post-training validation ---"
    python "${PROJECT_DIR}/validate.py" \
        --config "${CONFIG}" \
        --checkpoint "${CKPT_DIR}/final.pt" \
        --mode full \
        --output "${LOG_DIR}/final_validation.json" \
        ${NO_7B_ARG} \
    && echo "  Validation complete → ${LOG_DIR}/final_validation.json" \
    || echo "  Validation failed (non-fatal)"

    echo ""
    echo "================================================================"
    echo " Training complete."
    echo " End time : $(date)"
    echo " Duration : $(( ($(date +%s) - SLURM_JOB_START_TIME) / 3600 ))h"
    echo " Checkpoint: ${STORE_CKPT}"
    echo "================================================================"
else
    echo "WARNING: Training exited with non-zero code ${TRAIN_EXIT}."
    echo "         Partial checkpoint (if any) at ${CKPT_DIR}/latest.pt"
    echo "         Resubmit with: sbatch ${0} --resume ${CKPT_DIR}/latest.pt"
fi

exit ${TRAIN_EXIT}
