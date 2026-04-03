#!/usr/bin/env bash
# =============================================================================
# submit_stage.sh  ·  Submit SLURM job to stage datasets Store -> Scratch
# =============================================================================
#
# Usage:
#   bash scripts/submit_stage.sh -run1
#   bash scripts/submit_stage.sh -run1 --dry-run
#   bash scripts/submit_stage.sh -run1 --config configs/run1/data_run1.yaml
#
# Behavior:
#   - Submits one lightweight SLURM job.
#   - Runs inside the same EDF container environment as training.
#   - Reads dataset roots from the selected YAML config (datasets: section).
#   - Copies each dataset directory from /capstor/store/.../raw/... to
#     /capstor/scratch/cscs/$USER/ultrasound/raw/... via rsync.
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
LOG_DIR="${REPO_DIR}/logs/staging"

JOB_NAME="ultatron_stage"
TIME="04:00:00"
NODES=1
GPUS=0
CPUS=4

MODE=""
CONFIG="configs/run1/data_run1.yaml"
DRY_RUN=0

die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO]  $*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -run1) MODE="run1"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

[[ -n "$MODE" ]] || die "No mode specified. Use: -run1"

ABS_CONFIG="${REPO_DIR}/${CONFIG}"
[[ -f "${ABS_CONFIG}" ]] || die "Config not found: ${ABS_CONFIG}"
mkdir -p "${LOG_DIR}"

INNERSCRIPT="${LOG_DIR}/.inner_stage_${MODE}.sh"
OUTERSCRIPT="$(mktemp /tmp/ultatron_outer_stage_${MODE}_XXXXX.sh)"
trap "rm -f ${OUTERSCRIPT}" EXIT

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export CSCS_USER="\${CSCS_USER:-\${USER:-tbrokowski}}"

echo "================================================================"
echo " Ultatron — ${JOB_NAME}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
echo " Config : ${ABS_CONFIG}"
echo " DryRun : ${DRY_RUN}"
echo "================================================================"
echo ""

python3 - << 'PY_EOF'
import os
import subprocess
from pathlib import Path

import yaml

config_path = Path("${ABS_CONFIG}")
dry_run = bool(${DRY_RUN})
user = os.environ.get("CSCS_USER") or os.environ.get("USER")
if not user:
    raise SystemExit("CSCS_USER/USER is not set; cannot resolve scratch path.")

store_raw = Path("/capstor/store/cscs/swissai/a127/ultrasound/raw")
scratch_raw = Path(f"/capstor/scratch/cscs/{user}/ultrasound/raw")
scratch_raw.mkdir(parents=True, exist_ok=True)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

datasets = (cfg.get("datasets") or {})
if not datasets:
    raise SystemExit(f"No datasets found in config: {config_path}")

print(f"store_raw   : {store_raw}")
print(f"scratch_raw : {scratch_raw}")
print(f"dataset_cnt : {len(datasets)}")
print("")

has_rsync = (subprocess.run(["bash", "-lc", "command -v rsync >/dev/null 2>&1"]).returncode == 0)
copy_tool = "rsync" if has_rsync else "cp"
print(f"copy_tool   : {copy_tool}")
print("")

failures = []
for dataset_id, src_raw in datasets.items():
    src = Path(str(src_raw))
    if not src.exists():
        print(f"[FAIL] {dataset_id}: source missing -> {src}")
        failures.append(dataset_id)
        continue

    try:
        rel = src.relative_to(store_raw)
    except ValueError:
        print(f"[FAIL] {dataset_id}: source not under store raw root -> {src}")
        failures.append(dataset_id)
        continue

    dst = scratch_raw / rel
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[SYNC] {dataset_id}")
    print(f"       {src} -> {dst}")
    if has_rsync:
        cmd = ["rsync", "-ah", "--info=progress2", f"{src}/", f"{dst}/"]
        if dry_run:
            cmd.insert(1, "--dry-run")
        rc = subprocess.run(cmd).returncode
    else:
        if dry_run:
            print("       [DRY-RUN] cp -r <src>/. <dst>/")
            rc = 0
        else:
            # Portable fallback when rsync is unavailable in container.
            # Use non-preserving recursive copy because scratch may reject
            # ownership/permission/timestamp metadata from store.
            cmd = ["cp", "-r", f"{src}/.", f"{dst}/"]
            rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"[FAIL] {dataset_id}: copy command exit code {rc}")
        failures.append(dataset_id)

print("")
if failures:
    print("Staging finished with failures:")
    for d in failures:
        print(f"  - {d}")
    raise SystemExit(1)

print("Staging complete: all datasets synced successfully.")
PY_EOF

echo ""
echo "================================================================"
echo " STAGING COMPLETE  -- \$(date)"
echo "================================================================"
INNER_EOF

chmod +x "${INNERSCRIPT}"

cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
set -euo pipefail

srun --ntasks-per-node=1 \
     --environment=${EDF_ENV} \
     bash ${INNERSCRIPT}
OUTER_EOF

chmod +x "${OUTERSCRIPT}"

info "Submitting: ${JOB_NAME}  (mode=${MODE}, time=${TIME})"
JOB_ID=$(sbatch \
    --job-name="${JOB_NAME}" \
    --nodes="${NODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${CPUS}" \
    --time="${TIME}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
    --parsable \
    "${OUTERSCRIPT}")

echo ""
echo "  Job ID : ${JOB_ID}"
echo "  Log    : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Error  : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.err"
echo "  Watch  : tail -f ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
