#!/usr/bin/env bash
# =============================================================================
# download_mimic_iv_echo_pgroup.sh  ·  Parallel + resumable MIMIC-IV-Echo download
# =============================================================================
#
# SLURM array job — one task per patient group (p11–p19 by default).
# Each task runs PARALLEL_JOBS=4 concurrent wget workers, one patient at a time,
# which is the main throughput improvement over the original single-stream script.
#
# Resumable: wget -c skips fully-downloaded files and continues partial ones.
# Re-submit this job array as many times as needed; it converges to complete.
#
# Usage:
#   sbatch --array=0-8 scripts/download_mimic_iv_echo_pgroup.sh   # all 9 groups
#   sbatch --array=2   scripts/download_mimic_iv_echo_pgroup.sh   # just p13
#
# Task → group mapping (0-indexed):
#   0=p11  1=p12  2=p13  3=p14  4=p15  5=p16  6=p17  7=p18  8=p19
#
# Monitor:
#   tail -f logs/download/mimic_pgroup_p11_<jobid>.out
#   squeue -u $USER
# =============================================================================
#SBATCH --job-name=mimic_pgroup
#SBATCH --account=a127
#SBATCH --partition=xfer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/users/tbrokowski/Ultatron/logs/download/mimic_pgroup_%a_%j.out
#SBATCH --error=/users/tbrokowski/Ultatron/logs/download/mimic_pgroup_%a_%j.err

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PHYSIONET_USER="tbrokowski"
PHYSIONET_PASS="PhysioNet2468!"

TARGET_DIR="/capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/MIMIC-IV-Echo"
BASE_URL="https://physionet.org/files/mimic-iv-echo/1.0/files"

PARALLEL_JOBS=4   # concurrent wget workers within this SLURM task

LOG_DIR="/users/tbrokowski/Ultatron/logs/download"
mkdir -p "${LOG_DIR}"

# ── Resolve patient group from array task ID ──────────────────────────────────
PGROUPS=(p11 p12 p13 p14 p15 p16 p17 p18 p19)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PGROUP="${PGROUPS[$TASK_ID]}"

echo "================================================================"
echo " MIMIC-IV-Echo parallel download"
echo " Job     : ${SLURM_JOB_ID:-local}  task=${TASK_ID}"
echo " Group   : ${PGROUP}"
echo " Node    : $(hostname)"
echo " Start   : $(date -Is)"
echo " Workers : ${PARALLEL_JOBS}"
echo "================================================================"

# ── Build patient list from existing directory structure ──────────────────────
# The original wget crawl already created patient+study directory skeletons.
# We enumerate from disk rather than spidering PhysioNet again.
PGROUP_DIR="${TARGET_DIR}/physionet.org/files/mimic-iv-echo/1.0/files/${PGROUP}"

if [[ ! -d "${PGROUP_DIR}" ]]; then
    echo "ERROR: patient group directory not found: ${PGROUP_DIR}"
    echo "The original crawl may not have reached this group yet."
    echo "Run the original download_mimic_iv_echo.sh first to build the directory tree."
    exit 1
fi

PATIENT_LIST=$(ls "${PGROUP_DIR}" | grep "^p[0-9]" | sort)
TOTAL_PATIENTS=$(echo "${PATIENT_LIST}" | wc -l)

echo "Found ${TOTAL_PATIENTS} patients in ${PGROUP}"
echo ""

# ── Count already-complete patients (have at least one .dcm) ─────────────────
DONE=0
PENDING_LIST=""
while IFS= read -r patient; do
    dcm_count=$(find "${PGROUP_DIR}/${patient}" -name "*.dcm" 2>/dev/null | wc -l)
    if [[ "$dcm_count" -gt 0 ]]; then
        DONE=$((DONE + 1))
    else
        PENDING_LIST="${PENDING_LIST}${patient}"$'\n'
    fi
done <<< "${PATIENT_LIST}"

PENDING=$(echo "${PENDING_LIST}" | grep -c "^p" || true)
echo "Already downloaded : ${DONE} / ${TOTAL_PATIENTS} patients"
echo "Pending            : ${PENDING} patients"
echo ""

if [[ "${PENDING}" -eq 0 ]]; then
    echo "All patients in ${PGROUP} already downloaded. Nothing to do."
    exit 0
fi

# ── Worker function (called by xargs in parallel) ─────────────────────────────
export PHYSIONET_USER PHYSIONET_PASS TARGET_DIR BASE_URL PGROUP LOG_DIR

download_patient() {
    local patient="$1"
    local url="${BASE_URL}/${PGROUP}/${patient}/"
    local patient_log="${LOG_DIR}/mimic_${PGROUP}_${patient}.log"

    wget \
        --recursive \
        --continue \
        --no-parent \
        --tries=5 \
        --timeout=60 \
        --waitretry=10 \
        --directory-prefix="${TARGET_DIR}" \
        --output-file="${patient_log}" \
        --user="${PHYSIONET_USER}" \
        --password="${PHYSIONET_PASS}" \
        "${url}"

    local dcm_count
    dcm_count=$(find "${TARGET_DIR}/physionet.org/files/mimic-iv-echo/1.0/files/${PGROUP}/${patient}" \
                     -name "*.dcm" 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] ${PGROUP}/${patient}: ${dcm_count} dcm files"
}

export -f download_patient

# ── Run parallel downloads ────────────────────────────────────────────────────
echo "Starting ${PARALLEL_JOBS} parallel workers..."
echo ""

echo "${PENDING_LIST}" \
    | grep "^p[0-9]" \
    | xargs -P "${PARALLEL_JOBS}" -I{} bash -c 'download_patient "$@"' _ {}

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
FINAL_DCM=$(find "${PGROUP_DIR}" -name "*.dcm" 2>/dev/null | wc -l)
FINAL_PATIENTS=$(find "${PGROUP_DIR}" -mindepth 1 -maxdepth 1 -type d | wc -l)
COMPLETE_PATIENTS=$(find "${PGROUP_DIR}" -mindepth 1 -maxdepth 1 -type d | while read -r d; do
    [[ $(find "$d" -name "*.dcm" 2>/dev/null | wc -l) -gt 0 ]] && echo "$d"
done | wc -l)

echo " DONE    : $(date -Is)"
echo " Group   : ${PGROUP}"
echo " DCM files downloaded : ${FINAL_DCM}"
echo " Patients with data   : ${COMPLETE_PATIENTS} / ${FINAL_PATIENTS}"
echo "================================================================"

# Exit non-zero if incomplete (triggers SLURM job status = FAILED for easy monitoring)
if [[ "${COMPLETE_PATIENTS}" -lt "${FINAL_PATIENTS}" ]]; then
    echo "NOTE: $(( FINAL_PATIENTS - COMPLETE_PATIENTS )) patients still pending."
    echo "Resubmit this job — wget -c will resume where it left off."
    exit 1
fi
