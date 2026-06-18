#!/usr/bin/env bash
# Run Python (or any command) inside the CSCS EDF container when the login-node
# interpreter is too old (e.g. Python 3.6 → SyntaxError on `from __future__ import annotations`).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EDF_ENV="${ULTATRON_EDF_ENV:-${HOME}/.edf/ultatron.toml}"
ACCOUNT="${ULTATRON_ACCOUNT:-a127}"
PARTITION="${ULTATRON_PARTITION:-normal}"

_python_ok() {
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - <<'PY' >/dev/null 2>&1
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
PY
}

if [[ "${ULTATRON_FORCE_EDF:-}" != "1" ]] && _python_ok; then
  cd "${REPO_DIR}"
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
  exec python3 "$@"
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "[ERROR] Need Python 3.10+ or Slurm srun with EDF (${EDF_ENV})" >&2
  exit 1
fi

cd "${REPO_DIR}"
quoted=()
for arg in "$@"; do
  quoted+=("$(printf '%q' "$arg")")
done

exec srun \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:10:00 \
  --environment="${EDF_ENV}" \
  bash -lc "cd $(printf '%q' "$REPO_DIR") && export PYTHONPATH=$(printf '%q' "$REPO_DIR"):\${PYTHONPATH:-} && exec python3 ${quoted[*]}"
