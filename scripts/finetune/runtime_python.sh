#!/usr/bin/env bash
# Resolve Python >=3.10 for finetune jobs (login node vs EDF compute node).
resolve_finetune_python() {
  local _py
  for _py in python3 python3.12 python3.11; do
    if command -v "${_py}" >/dev/null 2>&1 \
      && "${_py}" -c 'import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then
      echo "${_py}"
      return 0
    fi
  done
  echo "[ERROR] Need Python >=3.10 (tried python3, python3.12, python3.11)" >&2
  return 1
}
