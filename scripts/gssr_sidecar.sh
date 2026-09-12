#!/usr/bin/env bash
# =============================================================================
# gssr_sidecar.sh  ·  One host-level GSSR recorder per node (WP5 / spec §1, §5)
# =============================================================================
#
# Reuse pattern from the meditron-4 repo: start a sidecar on local rank 0 of
# every node, write under $OUT/gssr/, stop on EXIT.
#
# Usage:
#   source scripts/gssr_sidecar.sh
#   gssr_sidecar_start "$EVIDENCE_DIR"
#   ... job ...
#   gssr_sidecar_stop
#
# If the CSCS `gssr` CLI is not on PATH, falls back to nvidia-smi dmon.
# =============================================================================
set -euo pipefail

_GSSR_PID=""
_GSSR_OUT=""

gssr_sidecar_start() {
    local out="${1:?output dir}"
    _GSSR_OUT="${out}/gssr"
    mkdir -p "${_GSSR_OUT}"

    # One recorder per node: torchrun LOCAL_RANK=0 or srun localid 0.
    local lr="${LOCAL_RANK:-${SLURM_LOCALID:-0}}"
    if [[ "${lr}" != "0" ]]; then
        return 0
    fi

    if command -v gssr >/dev/null 2>&1; then
        echo "[gssr] starting CSCS gssr → ${_GSSR_OUT}"
        gssr --output "${_GSSR_OUT}" >/dev/null 2>&1 &
        _GSSR_PID=$!
        echo "${_GSSR_PID}" > "${_GSSR_OUT}/gssr.pid"
        return 0
    fi

    echo "[gssr] gssr CLI not found; falling back to nvidia-smi dmon"
    nvidia-smi dmon -s pucvmet -d 5 -o DT > "${_GSSR_OUT}/nvidia-smi-dmon.csv" 2>&1 &
    _GSSR_PID=$!
    echo "${_GSSR_PID}" > "${_GSSR_OUT}/gssr.pid"
}

gssr_sidecar_stop() {
    if [[ -z "${_GSSR_PID}" && -n "${_GSSR_OUT:-}" && -f "${_GSSR_OUT}/gssr.pid" ]]; then
        _GSSR_PID="$(cat "${_GSSR_OUT}/gssr.pid" || true)"
    fi
    if [[ -n "${_GSSR_PID}" ]] && kill -0 "${_GSSR_PID}" 2>/dev/null; then
        kill "${_GSSR_PID}" 2>/dev/null || true
        wait "${_GSSR_PID}" 2>/dev/null || true
    fi
    if [[ -n "${_GSSR_OUT:-}" ]]; then
        echo "[gssr] stopped (dir=${_GSSR_OUT})"
    fi
}
