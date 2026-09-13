#!/usr/bin/env bash
# account.sh  ·  WP5 Slurm account and writable store (source; do not exec)
#
# Billing and writes go to a0238.  Existing US-365K / video copies / student
# checkpoints stay on a127 (read-only).  Override with POCUS_* env vars.
#
#   source scripts/pocus/account.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "source ${BASH_SOURCE[0]} — do not execute" >&2
    exit 2
fi

# Idempotent when several launchers source this in one shell.
if [[ -n "${_POCUS_ACCOUNT_SH:-}" ]]; then
    return 0
fi
_POCUS_ACCOUNT_SH=1

ACCOUNT="${POCUS_ACCOUNT:-${ULTATRON_ACCOUNT:-a0238}}"
export ACCOUNT
export POCUS_ACCOUNT="${ACCOUNT}"

STORE_ACCT="${POCUS_STORE_ACCT:-/capstor/store/cscs/swissai/${ACCOUNT}}"
export STORE_ACCT
export POCUS_STORE_ACCT="${STORE_ACCT}"

# Evidence under the billed account so mkdir / sbatch logs succeed.
# The infra01 tree is not writable from a0238.
EVIDENCE="${POCUS_EVIDENCE_ROOT:-${STORE_ACCT}/meditron-feasibility-review/pocus}"
export EVIDENCE
export POCUS_EVIDENCE_ROOT="${EVIDENCE}"

DATA_ROOT="${POCUS_DATA_ROOT:-${STORE_ACCT}/pocus-bench}"
RAW_ROOT="${POCUS_RAW_ROOT:-${DATA_ROOT}/raw}"
SHARD_ROOT="${POCUS_SHARD_ROOT:-${DATA_ROOT}/shards}"
MANIFESTS="${POCUS_MANIFEST_ROOT:-${DATA_ROOT}/manifests}"
export DATA_ROOT RAW_ROOT SHARD_ROOT MANIFESTS
export POCUS_DATA_ROOT="${DATA_ROOT}"
export POCUS_RAW_ROOT="${RAW_ROOT}"
export POCUS_SHARD_ROOT="${SHARD_ROOT}"
export POCUS_MANIFEST_ROOT="${MANIFESTS}"

# Read-only production copies on a127.
US365K_A127="${POCUS_US365K:-/capstor/store/cscs/swissai/a127/ultrasound/raw/multi_organ/US-365K}"
A127_US_ROOT="${POCUS_A127_US_ROOT:-/capstor/store/cscs/swissai/a127/ultrasound/raw}"
STUDENT_CKPT_DEFAULT="${POCUS_STUDENT_CKPT:-/capstor/store/cscs/swissai/a127/ultrasound/checkpoints/StudentPretrain/step_30000.pt}"
export US365K_A127 A127_US_ROOT STUDENT_CKPT_DEFAULT

# mkdir dest, or fall back so a wrong evidence path does not abort the campaign.
pocus_ensure_dir() {
    local dest="$1"
    local fallback="$2"
    if mkdir -p "${dest}" 2>/dev/null; then
        printf '%s\n' "${dest}"
        return 0
    fi
    echo "[WARN] cannot write ${dest}; using ${fallback}" >&2
    mkdir -p "${fallback}"
    printf '%s\n' "${fallback}"
}
