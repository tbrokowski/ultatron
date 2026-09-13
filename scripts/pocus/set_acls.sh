#!/usr/bin/env bash
# set_acls.sh  ·  Make the evidence tree readable by group csstaff (spec §7)
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/account.sh"
ROOT="${1:-${EVIDENCE}}"
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
