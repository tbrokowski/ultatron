#!/usr/bin/env bash
# =============================================================================
# download_covid_blues.sh  ·  COVID-BLUES lung ultrasound videos (GitHub)
# =============================================================================
#
# Downloads NinaWie/COVID-BLUES into:
#   /capstor/store/cscs/swissai/a127/ultrasound/raw/lung/COVID-BLUES
#
# Dataset: 362 LUS videos (BLUE protocol), severity.csv, clinical_variables.csv
# Source:   https://github.com/NinaWie/COVID-BLUES
#
# After clone, runs format_covid_blues_labels.py to produce:
#   metadata/video_labels.jsonl
#   metadata/patient_splits.json
#
# Usage:
#   sbatch scripts/download_covid_blues.sh
#   bash   scripts/download_covid_blues.sh
#
# Monitor:
#   tail -f logs/download/covid_blues_<jobid>.out
# =============================================================================
#SBATCH --job-name=covid_blues_dl
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/users/tbrokowski/Ultatron/logs/download/covid_blues_%j.out
#SBATCH --error=/users/tbrokowski/Ultatron/logs/download/covid_blues_%j.err

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
TARGET_DIR="/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/COVID-BLUES"
GITHUB_REPO="https://github.com/NinaWie/COVID-BLUES.git"
VENV="${REPO_DIR}/.venv"

mkdir -p "${TARGET_DIR}"
mkdir -p "${REPO_DIR}/logs/download"

echo "================================================================"
echo " COVID-BLUES download (GitHub)"
echo " Job    : ${SLURM_JOB_ID:-local}"
echo " Node   : $(hostname)"
echo " Target : ${TARGET_DIR}"
echo " Start  : $(date -Is)"
echo "================================================================"

# ── Clone or update ───────────────────────────────────────────────────────────
if [[ -d "${TARGET_DIR}/.git" ]]; then
    echo "Repo exists — pulling latest..."
    git -C "${TARGET_DIR}" pull --ff-only
else
    echo "Cloning ${GITHUB_REPO} ..."
    # Clone into a temp dir then move contents (TARGET may be non-empty)
    TMP_CLONE="$(mktemp -d)"
    git clone --depth 1 "${GITHUB_REPO}" "${TMP_CLONE}/COVID-BLUES"
    shopt -s dotglob
    for item in "${TMP_CLONE}/COVID-BLUES"/*; do
        name="$(basename "${item}")"
        if [[ ! -e "${TARGET_DIR}/${name}" ]]; then
            mv "${item}" "${TARGET_DIR}/"
            echo "  Installed: ${name}"
        else
            echo "  Skip (exists): ${name}"
        fi
    done
    shopt -u dotglob
    rm -rf "${TMP_CLONE}"
fi

# Git LFS for mp4 if configured in repo
if command -v git-lfs >/dev/null 2>&1 && [[ -d "${TARGET_DIR}/.git" ]]; then
    echo "Pulling Git LFS objects (if any)..."
    git -C "${TARGET_DIR}" lfs pull || true
fi

echo ""
echo "Video count: $(find "${TARGET_DIR}/lus_videos" -name '*.mp4' 2>/dev/null | wc -l)"

# ── Format labels ─────────────────────────────────────────────────────────────
echo ""
echo "Formatting labels → metadata/video_labels.jsonl"
if [[ -f "${VENV}/bin/activate" ]]; then
    source "${VENV}/bin/activate"
fi
python3 "${REPO_DIR}/scripts/format_covid_blues_labels.py" --root "${TARGET_DIR}"

echo ""
echo "Sample label record:"
head -1 "${TARGET_DIR}/metadata/video_labels.jsonl" 2>/dev/null || true

echo ""
echo "================================================================"
echo " Finished : $(date -Is)"
echo "================================================================"
