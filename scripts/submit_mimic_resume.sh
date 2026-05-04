#!/usr/bin/env bash
# =============================================================================
# submit_mimic_resume.sh  ·  Submit / re-submit MIMIC-IV-Echo parallel download
# =============================================================================
#
# Submits one SLURM array job covering p11–p19 (9 tasks).
# Safe to run multiple times — wget -c resumes partial downloads and the job
# script skips patients that already have DCM files.
#
# Usage:
#   bash scripts/submit_mimic_resume.sh              # submit all pending groups
#   bash scripts/submit_mimic_resume.sh --status     # show current progress only
#   bash scripts/submit_mimic_resume.sh --group p13  # submit one specific group
#   bash scripts/submit_mimic_resume.sh --dry-run    # print sbatch command only
#
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
SCRIPT="${REPO_DIR}/scripts/download_mimic_iv_echo_pgroup.sh"
LOG_DIR="${REPO_DIR}/logs/download"
TARGET_DIR="/capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/MIMIC-IV-Echo"
FILES_DIR="${TARGET_DIR}/physionet.org/files/mimic-iv-echo/1.0/files"

ACCOUNT="a127"
PARTITION="xfer"   # transfer partition: 24h limit, designed for I/O jobs
TIME="24:00:00"

# task index → group name (matches array in the job script)
PGROUPS=(p11 p12 p13 p14 p15 p16 p17 p18 p19)

# ── Arg parsing ───────────────────────────────────────────────────────────────
STATUS_ONLY=0
DRY_RUN=0
SINGLE_GROUP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)    STATUS_ONLY=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        --group)     SINGLE_GROUP="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"

# ── Progress check ────────────────────────────────────────────────────────────
echo ""
echo "  MIMIC-IV-Echo download progress"
echo "  ────────────────────────────────────────────────────────────"
printf "  %-6s  %8s  %8s  %8s  %s\n" "Group" "Patients" "With DCM" "Pending" "Status"
echo "  ────────────────────────────────────────────────────────────"

PENDING_TASKS=()

for i in "${!PGROUPS[@]}"; do
    grp="${PGROUPS[$i]}"
    grp_dir="${FILES_DIR}/${grp}"

    if [[ ! -d "${grp_dir}" ]]; then
        printf "  %-6s  %8s  %8s  %8s  %s\n" "${grp}" "?" "?" "?" "DIR MISSING"
        PENDING_TASKS+=("$i")
        continue
    fi

    total=$(ls "${grp_dir}" | grep -c "^p[0-9]" || echo 0)
    # Count patients that have at least one DCM using a single find + awk
    with_dcm=$(find "${grp_dir}" -name "*.dcm" 2>/dev/null \
               | awk -F/ '{print $(NF-2)}' | sort -u | wc -l)
    pending=$(( total - with_dcm ))

    if [[ "$pending" -eq 0 && "$total" -gt 0 ]]; then
        status="COMPLETE"
    elif [[ "$with_dcm" -gt 0 ]]; then
        status="IN PROGRESS"
        PENDING_TASKS+=("$i")
    else
        status="NOT STARTED"
        PENDING_TASKS+=("$i")
    fi

    printf "  %-6s  %8d  %8d  %8d  %s\n" "${grp}" "${total}" "${with_dcm}" "${pending}" "${status}"
done

echo "  ────────────────────────────────────────────────────────────"
echo ""

if [[ "$STATUS_ONLY" -eq 1 ]]; then
    exit 0
fi

# ── Determine which tasks to submit ──────────────────────────────────────────
if [[ -n "$SINGLE_GROUP" ]]; then
    # Find task index for the requested group
    TASK_IDS=()
    for i in "${!PGROUPS[@]}"; do
        [[ "${PGROUPS[$i]}" == "${SINGLE_GROUP}" ]] && TASK_IDS+=("$i")
    done
    if [[ ${#TASK_IDS[@]} -eq 0 ]]; then
        echo "[ERROR] Unknown group: ${SINGLE_GROUP}. Valid: ${PGROUPS[*]}" >&2
        exit 1
    fi
    ARRAY_SPEC="${TASK_IDS[0]}"
else
    # Submit all pending tasks
    if [[ ${#PENDING_TASKS[@]} -eq 0 ]]; then
        echo "All groups are complete. Nothing to submit."
        exit 0
    fi
    # Build SLURM array spec from pending task indices, e.g. "0,2,5-8"
    ARRAY_SPEC=$(printf '%s,' "${PENDING_TASKS[@]}" | sed 's/,$//')
fi

echo "  Submitting tasks: ${ARRAY_SPEC}"
echo ""

SBATCH_CMD=(
    sbatch
    --array="${ARRAY_SPEC}"
    --account="${ACCOUNT}"
    --partition="${PARTITION}"
    --time="${TIME}"
    --job-name="mimic_pgroup"
    --nodes=1
    --ntasks=1
    --cpus-per-task=4
    --output="${LOG_DIR}/mimic_pgroup_%a_%j.out"
    --error="${LOG_DIR}/mimic_pgroup_%a_%j.err"
    "${SCRIPT}"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  [DRY-RUN] ${SBATCH_CMD[*]}"
    exit 0
fi

JOB_ID=$("${SBATCH_CMD[@]}" --parsable)
echo "  Submitted job array ID : ${JOB_ID}"
echo "  Tasks                  : ${ARRAY_SPEC}"
echo "  Time limit             : ${TIME} per task"
echo "  Log dir                : ${LOG_DIR}/"
echo ""
echo "  Monitor:"
echo "    squeue -u \$USER"
echo "    tail -f ${LOG_DIR}/mimic_pgroup_0_*.out   # group p11"
echo ""
echo "  Re-run this script at any time to check progress and resubmit incomplete groups."
