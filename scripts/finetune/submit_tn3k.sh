#!/usr/bin/env bash
# Submit TN3K finetune comparison (OpenUS protocol, 1 GPU).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=submit_common.sh
source "${SCRIPT_DIR}/submit_common.sh"
submit_finetune_experiment tn3k \
  --comparison-config "${SCRIPT_DIR}/../../configs/finetune/openus_segmentation.yaml" \
  --all-backbones \
  "$@"
