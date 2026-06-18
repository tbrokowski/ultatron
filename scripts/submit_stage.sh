#!/usr/bin/env bash
# =============================================================================
# submit_stage.sh  ·  Submit SLURM job to stage datasets Store -> Scratch
# =============================================================================
#
# Usage:
#   bash scripts/submit_stage.sh
#   bash scripts/submit_stage.sh --dry-run
#   bash scripts/submit_stage.sh --config configs/run1/data_run1.yaml   # subset only
#
# Default behavior:
#   - Stages every dataset in DATASET_STORE_MAP that exists on Capstor Store.
#   - Skips datasets whose store directory is missing (not staged yet on archive).
#   - Uses rsync when available, otherwise cp -r (safe for paths with spaces/parens).
#   - Runs inside the same EDF container environment as training.
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
LOG_DIR="${REPO_DIR}/logs/staging"

JOB_NAME="ultatron_stage"
TIME="12:00:00"
NODES=1
GPUS=0
CPUS=4

CONFIG=""
DRY_RUN=0

die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO]  $*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        -run1) shift ;;   # legacy alias; default is now all in-store datasets
        --config) CONFIG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

if [[ -n "${CONFIG}" ]]; then
    ABS_CONFIG="${REPO_DIR}/${CONFIG}"
    [[ -f "${ABS_CONFIG}" ]] || die "Config not found: ${ABS_CONFIG}"
    CONFIG_ARG="${ABS_CONFIG}"
    MODE="config"
else
    CONFIG_ARG=""
    MODE="all"
fi

mkdir -p "${LOG_DIR}"

INNERSCRIPT="${LOG_DIR}/.inner_stage_${MODE}.sh"
OUTERSCRIPT="$(mktemp /tmp/ultatron_outer_stage_${MODE}_XXXXX.sh)"
trap "rm -f ${OUTERSCRIPT}" EXIT

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
ulimit -c 0
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CSCS_USER="\${CSCS_USER:-\${USER:-tbrokowski}}"

echo "================================================================"
echo " Ultatron — ${JOB_NAME}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
echo " Mode   : ${MODE}"
echo " Config : ${CONFIG_ARG:-<all in-store datasets>}"
echo " DryRun : ${DRY_RUN}"
echo "================================================================"
echo ""

python3 -u << 'PY_EOF'
import os
import subprocess
import sys
import types
from pathlib import Path

def log(msg: str) -> None:
    print(msg, flush=True)

REPO = Path("${REPO_DIR}")
dry_run = bool(${DRY_RUN})
config_path = "${CONFIG_ARG}"

# Avoid importing data/__init__.py (pulls torch).
if "data" not in sys.modules:
    data_stub = types.ModuleType("data")
    data_stub.__path__ = [str(REPO / "data")]
    data_stub.__package__ = "data"
    sys.modules["data"] = data_stub

from data.infra.storage import DATASET_STORE_MAP, StorageConfig

cfg = StorageConfig()
if not cfg.scratch_root:
    raise SystemExit("scratch_root not configured. Set CSCS_USER or US_SCRATCH_ROOT.")

store_raw = cfg.store_root / "raw"
scratch_raw = cfg.scratch_root / "raw"
scratch_raw.mkdir(parents=True, exist_ok=True)

def resolve_sources():
    """Return ordered list of (dataset_id, src Path)."""
    if config_path:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            ycfg = yaml.safe_load(f) or {}
        datasets = ycfg.get("datasets") or {}
        if not datasets:
            raise SystemExit(f"No datasets found in config: {config_path}")
        out = []
        for dataset_id, src_raw in datasets.items():
            out.append((dataset_id, Path(str(src_raw))))
        return out

    out = []
    for dataset_id in sorted(DATASET_STORE_MAP.keys()):
        anatomy, subdir = DATASET_STORE_MAP[dataset_id]
        src = store_raw / anatomy / subdir
        out.append((dataset_id, src))
    return out

sources = resolve_sources()

from shutil import which as _which
has_rsync = _which("rsync") is not None
copy_tool = "rsync" if has_rsync else "cp"

log("Scanning store and building dataset list...")
log(f"store_raw    : {store_raw}")
log(f"scratch_raw  : {scratch_raw}")
log(f"dataset_cnt  : {len(sources)}")
log(f"copy_tool    : {copy_tool}")
log("")

skipped = []
failures = []
synced = []

for dataset_id, src in sources:
    if not src.exists() or not any(src.iterdir()):
        log(f"[SKIP] {dataset_id}: not on store -> {src}")
        skipped.append(dataset_id)
        continue

    try:
        rel = src.relative_to(store_raw)
    except ValueError:
        log(f"[FAIL] {dataset_id}: source not under store raw root -> {src}")
        failures.append(dataset_id)
        continue

    dst = scratch_raw / rel
    dst.mkdir(parents=True, exist_ok=True)

    log(f"[SYNC] {dataset_id}")
    log(f"       {src} -> {dst}")

    if has_rsync:
        cmd = ["rsync", "-ah", "--info=progress2", f"{src}/", f"{dst}/"]
        if dry_run:
            cmd.insert(1, "--dry-run")
        rc = subprocess.run(cmd).returncode
    elif dry_run:
        log("       [DRY-RUN] cp -r <src>/. <dst>/")
        rc = 0
    else:
        cmd = ["cp", "-r", f"{src}/.", f"{dst}/"]
        rc = subprocess.run(cmd).returncode

    if rc != 0:
        log(f"[FAIL] {dataset_id}: copy command exit code {rc}")
        failures.append(dataset_id)
    else:
        synced.append(dataset_id)

log("")
log(f"Synced  : {len(synced)}")
log(f"Skipped : {len(skipped)} (not on store)")
log(f"Failed  : {len(failures)}")
if failures:
    log("")
    log("Staging finished with copy failures:")
    for d in failures:
        log(f"  - {d}")
    raise SystemExit(1)

log("Staging complete.")
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
