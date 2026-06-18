#!/usr/bin/env bash
# Submit BUSI finetune comparison (all default backbones, 1 GPU).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=submit_common.sh
source "${SCRIPT_DIR}/submit_common.sh"
submit_finetune_experiment busi "$@"
