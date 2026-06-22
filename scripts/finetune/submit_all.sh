#!/usr/bin/env bash
# Submit ALL finetune comparison experiments (representative + openus + fetal planes).
#
# Usage:
#   bash scripts/finetune/submit_all.sh              # submit everything
#   bash scripts/finetune/submit_all.sh --after-job 2566886   # wait for pretrain job
#   bash scripts/finetune/submit_all.sh --local        # run representative sweep locally
#
# Individual experiment scripts are in scripts/finetune/submit_*.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

AFTER_JOB=""
LOCAL=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --after-job) AFTER_JOB="$2"; shift 2 ;;
    --local)     LOCAL=1; shift ;;
    *)           EXTRA+=("$1"); shift ;;
  esac
done

_common_args() {
  local args=()
  [[ -n "${AFTER_JOB}" ]] && args+=(--after-job "${AFTER_JOB}")
  [[ ${LOCAL} -eq 1 ]] && args+=(--local)
  args+=("${EXTRA[@]}")
  echo "${args[@]}"
}

_info() { echo "[submit_all] $*"; }

_info "Student checkpoint: step_22500.pt (StudentPretrain)"
_info "Results root: results/finetune/"

# ── Representative sweep (comparison_representative.yaml) ─────────────────────
# busi, busi_multitask, camus, echonet, lus, lus_video
_info "Submitting representative sweep (6 experiments, 4 GPUs)..."
"${FINETUNE_PYTHON:-python3.11}" scripts/finetune/run_all.py $( _common_args ) --gpus 4

# ── OpenUS segmentation (openus_segmentation.yaml) ───────────────────────────
# busbra + tn3k in parallel (2 GPUs)
_info "Submitting OpenUS segmentation (busbra + tn3k, 2 GPUs)..."
bash scripts/finetune/submit_openus_segmentation.sh $( _common_args )

# ── Fetal planes classification ───────────────────────────────────────────────
_info "Submitting fetal planes DB..."
bash scripts/finetune/submit_fetal_planes_db.sh $( _common_args )

_info "All finetune jobs submitted."
_info "Monitor: ls logs/finetune/ultatron_ft_*.err"
_info "Report:  python3.11 scripts/finetune/report.py --from-logs --all-sweeps"
