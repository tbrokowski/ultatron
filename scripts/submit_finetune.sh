#!/usr/bin/env bash
# Wrapper for scripts/finetune/ launchers (comparison mode).
# Legacy single-checkpoint mode still uses scripts/finetune.py inside the EDF container.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

for arg in "$@"; do
    if [[ "${arg}" == "--checkpoint" ]]; then
        exec bash "${REPO_DIR}/scripts/run_in_edf.sh" scripts/finetune.py "$@"
    fi
done

SCRIPT_DIR="${REPO_DIR}/scripts/finetune"
# shellcheck source=finetune/submit_common.sh
source "${SCRIPT_DIR}/submit_common.sh"
submit_finetune_all "$@"
