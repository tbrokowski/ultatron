#!/usr/bin/env bash
# Prepare Run US-365K inside its CSCS container.
# Usage: bash scripts/setup_run_us365k.sh
# Overrides: ULTATRON_ACCOUNT, ULTATRON_PARTITION, ULTATRON_EDF_ENV.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCOUNT="${ULTATRON_ACCOUNT:-a127}"
PARTITION="${ULTATRON_PARTITION:-normal}"
EDF_ENV="${ULTATRON_EDF_ENV:-${REPO_DIR}/configs/run_us365k/ultatron.toml}"

exec srun \
    --account="${ACCOUNT}" --partition="${PARTITION}" --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --mem=64G --time=00:30:00 \
    --mpi=pmix --network=disable_rdzv_get \
    --environment="${EDF_ENV}" \
    bash -s -- "${REPO_DIR}" <<'SETUP'
set -euo pipefail
cd "$1"
bash scripts/ensure_deps.sh
source scripts/setup_hf_cache.sh
python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "facebook/sam2.1-hiera-large",
    "config.json",
    cache_dir=os.environ["US_HF_CACHE_DIR"],
)
print(f"SAM2 architecture configuration cached at {path}")
PY
SETUP
