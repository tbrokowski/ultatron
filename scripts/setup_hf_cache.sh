#!/usr/bin/env bash
# =============================================================================
# setup_hf_cache.sh  ·  Point HuggingFace at Capstor scratch/store cache
# =============================================================================
#
# Source before any job that loads gated HF models (DINOv3, V-JEPA2, Hiera, …):
#
#   source scripts/setup_hf_cache.sh
#
# Auth for fresh downloads (if a model is not yet cached):
#   1. One-time on login node:  huggingface-cli login
#      → writes ~/.cache/huggingface/token (recommended)
#   2. Per-session:             export HF_TOKEN=hf_...
#
# Do NOT commit HF tokens to the repo.
# =============================================================================

_STORE_HF="/capstor/store/cscs/swissai/a127/ultrasound/hf_cache"
_USER="${USER:-$(whoami)}"
_SCRATCH_HF="/capstor/scratch/cscs/${_USER}/ultrasound/hf_cache"

_hf_has_models() {
    [[ -d "$1" ]] && compgen -G "$1/models--*" > /dev/null
}

if [[ -n "${US_HF_CACHE_DIR:-}" ]]; then
    _HF_CACHE="${US_HF_CACHE_DIR}"
elif _hf_has_models "${_SCRATCH_HF}"; then
    _HF_CACHE="${_SCRATCH_HF}"
elif _hf_has_models "${_STORE_HF}"; then
    _HF_CACHE="${_STORE_HF}"
elif [[ -d "${_SCRATCH_HF}" ]]; then
    _HF_CACHE="${_SCRATCH_HF}"
else
    _HF_CACHE="${_STORE_HF}"
fi

mkdir -p "${_HF_CACHE}"

export HF_HOME="${_HF_CACHE}"
export HF_HUB_CACHE="${_HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${_HF_CACHE}"

# huggingface_hub reads HF_TOKEN; token file stays in ~/.cache/huggingface/token
if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
    export HF_TOKEN="$(tr -d '[:space:]' < "${HOME}/.cache/huggingface/token")"
fi

echo "HF cache : ${_HF_CACHE}"
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "HF auth  : token set (env or ~/.cache/huggingface/token)"
else
    echo "HF auth  : none — cached weights only; run 'huggingface-cli login' to download new models"
fi
