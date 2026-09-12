#!/usr/bin/env bash
# set_acls.sh  ·  Make the evidence tree readable by group csstaff (spec §7)
set -euo pipefail
ROOT="${1:-${POCUS_EVIDENCE_ROOT:-/capstor/store/cscs/swissai/infra01/meditron-feasibility-review/pocus}}"
GROUP="${2:-csstaff}"

mkdir -p "${ROOT}"
# Default ACL so new files inherit group readability.
setfacl -R -m "g:${GROUP}:rX" "${ROOT}" || {
    echo "[WARN] setfacl failed; falling back to chmod g+rX"
    chmod -R g+rX "${ROOT}" || true
}
setfacl -R -d -m "g:${GROUP}:rX" "${ROOT}" 2>/dev/null || true
echo "ACLs on ${ROOT}:"
getfacl -p "${ROOT}" | head -n 20
