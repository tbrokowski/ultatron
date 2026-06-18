#!/usr/bin/env bash
# Submit FETAL_PLANES_DB fetal plane classification finetune (1 GPU).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=submit_common.sh
source "${SCRIPT_DIR}/submit_common.sh"
submit_finetune_experiment fetal_planes_db \
  --comparison-config "${SCRIPT_DIR}/../../configs/finetune/fetal_planes_classification.yaml" \
  "$@"
