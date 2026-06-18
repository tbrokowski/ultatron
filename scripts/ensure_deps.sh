#!/usr/bin/env bash
# Install Ultatron project dependencies (pyproject.toml) when not already present.
set -euo pipefail
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=disable_core_dumps.sh
source "${REPO_DIR}/scripts/disable_core_dumps.sh"
python3 "${REPO_DIR}/scripts/ensure_deps.py"
