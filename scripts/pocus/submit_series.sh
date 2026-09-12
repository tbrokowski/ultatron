#!/usr/bin/env bash
# submit_series.sh  ·  Strong-scaling grids for WP5 (spec §4.4)
#
#   bash scripts/pocus/submit_series.sh encoder   # E1 + E2 at 1/2/4/8 nodes
#   bash scripts/pocus/submit_series.sh rl        # R1 + R2 at 1/2/4/8
#   bash scripts/pocus/submit_series.sh all
#
# Repeats at the smallest and largest scale. GSSR is on at 8 nodes.
set -euo pipefail

REPO_DIR="${ULTATRON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WHICH="${1:-encoder}"
NODES=(1 2 4 8)

launch_enc() {
    local exp="$1" n="$2"
    local extra=()
    if [[ "$n" -eq 1 || "$n" -eq 8 ]]; then extra+=(--repeat); fi
    if [[ "$n" -ge 8 ]]; then extra+=(--gssr); fi
    bash "${REPO_DIR}/scripts/pocus/submit_encoder.sh" "${exp}" --nodes "${n}" "${extra[@]}"
}

launch_rl() {
    local exp="$1" n="$2"
    local extra=()
    if [[ "$n" -eq 1 || "$n" -eq 8 ]]; then extra+=(--repeat); fi
    if [[ "$n" -ge 8 ]]; then extra+=(--gssr); fi
    bash "${REPO_DIR}/scripts/pocus/submit_rl.sh" "${exp}" --nodes "${n}" "${extra[@]}"
}

case "${WHICH}" in
    encoder)
        for n in "${NODES[@]}"; do launch_enc E1 "$n"; launch_enc E2 "$n"; done
        ;;
    rl)
        for n in "${NODES[@]}"; do launch_rl R1 "$n"; launch_rl R2 "$n"; done
        ;;
    all)
        for n in "${NODES[@]}"; do
            launch_enc E1 "$n"; launch_enc E2 "$n"
            launch_rl R1 "$n"; launch_rl R2 "$n"
        done
        ;;
    *)
        echo "usage: $0 encoder|rl|all" >&2
        exit 1
        ;;
esac
