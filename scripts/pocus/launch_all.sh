#!/usr/bin/env bash
# =============================================================================
# launch_all.sh  ·  Submit the full WP5 POCUS campaign on Clariden
# =============================================================================
#
# Spec §4.4 run order:
#   1. data build (optional --prep-data) and E6 NCCL
#   2. E0, R0
#   3. E1/E2 in parallel with R1/R2  (1/2/4/8 nodes; repeats at 1 and 8)
#   4. E3/E4/E5, R3/R4/R5 at n* (default 4 nodes; override --nstar-nodes)
#
# On a Clariden login node:
#   export POCUS_ACCOUNT=a0238         # default; billed account + writable store
#   bash scripts/pocus/launch_all.sh
#   bash scripts/pocus/launch_all.sh --dry-run
#   bash scripts/pocus/launch_all.sh --prep-data --nstar-nodes 4
#   bash scripts/pocus/launch_all.sh --encoder-only
#   bash scripts/pocus/launch_all.sh --rl-only
#
# Writes $EVIDENCE/launch_all_<timestamp>.txt with every Slurm job ID.
# =============================================================================
set -euo pipefail

REPO_DIR="${ULTATRON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/account.sh"
NODES=(1 2 4 8)
NSTAR_NODES="${POCUS_NSTAR_NODES:-4}"
DRY_RUN=0
PREP_DATA=0
DO_ENCODER=1
DO_RL=1
DO_R5=1
FAKE_ID=0
LAST_IDS=""
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --prep-data) PREP_DATA=1; shift ;;
        --encoder-only) DO_RL=0; shift ;;
        --rl-only) DO_ENCODER=0; shift ;;
        --skip-r5) DO_R5=0; shift ;;
        --nstar-nodes) NSTAR_NODES="$2"; shift 2 ;;
        --nodes)
            IFS=',' read -r -a NODES <<< "$2"
            shift 2
            ;;
        -h|--help) usage ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "${REPO_DIR}/logs/pocus"
JOBS_FILE="${REPO_DIR}/logs/pocus/launch_all_${STAMP}.txt"
if [[ "${DRY_RUN}" -eq 0 ]]; then
    if mkdir -p "${EVIDENCE}" 2>/dev/null; then
        JOBS_FILE="${EVIDENCE}/launch_all_${STAMP}.txt"
    else
        echo "[WARN] cannot write ${EVIDENCE}; job list → ${JOBS_FILE}" >&2
    fi
else
    JOBS_FILE="${REPO_DIR}/logs/pocus/launch_all_${STAMP}.dry.txt"
fi

{
    echo "# WP5 POCUS launch_all  ${STAMP}"
    echo "# account=${ACCOUNT}  evidence=${EVIDENCE}  n*=${NSTAR_NODES} nodes"
    echo "# dry_run=${DRY_RUN} encoder=${DO_ENCODER} rl=${DO_RL} r5=${DO_R5}"
    echo
} | tee "${JOBS_FILE}"

parse_ids() {
    # stdin: submit script stdout. Prints colon-separated job IDs.
    awk '
        /^JOB_ID=/ { sub(/^JOB_ID=/, ""); gsub(/[[:space:]]/, ""); if ($0 != "") print $0 }
        /^  Job ID : / { id=$4; gsub(/[^0-9]/, "", id); if (id != "") print id }
    ' | awk 'NF && !seen[$0]++' | paste -sd: -
}

run_submit() {
    local kind="$1"
    shift
    local out ids
    LAST_IDS=""
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        FAKE_ID=$((FAKE_ID + 1))
        echo "[dry-run] ${kind} $*" >&2
        echo "${kind}	$*	${FAKE_ID}" >> "${JOBS_FILE}"
        LAST_IDS="${FAKE_ID}"
        return 0
    fi
    echo ">>> ${kind} $*" >&2
    out="$(bash "${REPO_DIR}/scripts/pocus/${kind}" "$@" | tee /dev/stderr)" || {
        echo "[ERROR] ${kind} $* failed" >&2
        return 1
    }
    ids="$(printf '%s\n' "${out}" | parse_ids)"
    if [[ -z "${ids}" ]]; then
        echo "[ERROR] no JOB_ID from ${kind} $*" >&2
        return 1
    fi
    echo "${kind}	$*	${ids}" | tee -a "${JOBS_FILE}" >&2
    LAST_IDS="${ids}"
}

after_args() {
    # usage: after_args "$ids" → fills AFTER as (--after-job id) or empty
    AFTER=()
    if [[ -n "${1:-}" ]]; then
        AFTER+=(--after-job "$1")
    fi
}

join_dep() {
    local acc="" x
    for x in "$@"; do
        [[ -z "${x}" ]] && continue
        if [[ -z "${acc}" ]]; then acc="${x}"; else acc="${acc}:${x}"; fi
    done
    echo "${acc}"
}

prep_data() {
    echo "=== data build (spec §2.2–2.5) ==="
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[dry-run] bash scripts/pocus/download.sh"
        echo "[dry-run] python3 scripts/pocus/inspect_us365k.py"
        echo "[dry-run] python3 scripts/pocus/build_manifests.py --out ${MANIFESTS}"
        echo "[dry-run] python3 scripts/pocus/build_shards.py"
        echo "[dry-run] python3 scripts/pocus/data_facts.py --out ${EVIDENCE}/data_facts.json"
        return 0
    fi
    bash "${REPO_DIR}/scripts/pocus/download.sh"
    python3 "${REPO_DIR}/scripts/pocus/inspect_us365k.py" \
        --out "${MANIFESTS}/us365k_fields.json" || true
    python3 "${REPO_DIR}/scripts/pocus/build_manifests.py" --out "${MANIFESTS}"
    python3 "${REPO_DIR}/scripts/pocus/build_shards.py"
    python3 "${REPO_DIR}/scripts/pocus/data_facts.py" --out "${EVIDENCE}/data_facts.json"
}

# ---------------------------------------------------------------------------
if [[ "${PREP_DATA}" -eq 1 ]]; then
    prep_data
fi

E6_IDS=""
E0_IDS=""
ENC_SERIES=""
R0_IDS=""
RL_SERIES=""
NSTAR_DEP=""

echo "=== 1. E6 NCCL (Slingshot, then sockets) ==="
run_submit submit_nccl.sh --slingshot
E6_IDS="${LAST_IDS}"
run_submit submit_nccl.sh --sockets
E6_IDS="$(join_dep "${E6_IDS}" "${LAST_IDS}")"

if [[ "${DO_ENCODER}" -eq 1 ]]; then
    echo "=== 2. E0 minimal setup / OOM probe (1 GPU, then 1 node) ==="
    run_submit submit_encoder.sh E0
    E0_1="${LAST_IDS}"
    after_args "${E0_1}"
    run_submit submit_encoder.sh E0 --full-node "${AFTER[@]}"
    E0_IDS="$(join_dep "${E0_1}" "${LAST_IDS}")"
fi

if [[ "${DO_RL}" -eq 1 ]]; then
    echo "=== 2. R0 host-memory / OOM probe ==="
    run_submit submit_rl.sh R0
    R0_IDS="${LAST_IDS}"
fi

if [[ "${DO_ENCODER}" -eq 1 ]]; then
    echo "=== 3. E1 / E2 strong scaling (1/2/4/8 nodes) ==="
    for n in "${NODES[@]}"; do
        extra=()
        after_args "${E0_IDS}"
        extra+=("${AFTER[@]}")
        if [[ "${n}" -eq 1 || "${n}" -eq 8 ]]; then extra+=(--repeat); fi
        if [[ "${n}" -ge 8 ]]; then extra+=(--gssr); fi
        run_submit submit_encoder.sh E1 --nodes "${n}" "${extra[@]}"
        ENC_SERIES="$(join_dep "${ENC_SERIES}" "${LAST_IDS}")"
        run_submit submit_encoder.sh E2 --nodes "${n}" "${extra[@]}"
        ENC_SERIES="$(join_dep "${ENC_SERIES}" "${LAST_IDS}")"
    done
fi

if [[ "${DO_RL}" -eq 1 ]]; then
    echo "=== 3. R1 / R2 strong scaling (1/2/4/8 nodes) ==="
    for n in "${NODES[@]}"; do
        extra=()
        after_args "${R0_IDS}"
        extra+=("${AFTER[@]}")
        if [[ "${n}" -eq 1 || "${n}" -eq 8 ]]; then extra+=(--repeat); fi
        if [[ "${n}" -ge 8 ]]; then extra+=(--gssr); fi
        run_submit submit_rl.sh R1 --nodes "${n}" "${extra[@]}"
        RL_SERIES="$(join_dep "${RL_SERIES}" "${LAST_IDS}")"
        run_submit submit_rl.sh R2 --nodes "${n}" "${extra[@]}"
        RL_SERIES="$(join_dep "${RL_SERIES}" "${LAST_IDS}")"
    done
fi

ENC_NSTAR_DEP="${ENC_SERIES}"
RL_NSTAR_DEP="${RL_SERIES}"

if [[ "${DO_ENCODER}" -eq 1 ]]; then
    echo "=== 5. E3 stages 3+4, E4 checkpoint, E5 loader-only at n*=${NSTAR_NODES} ==="
    after_args "${ENC_NSTAR_DEP:-${E0_IDS}}"
    run_submit submit_encoder.sh E3 --stage 3 --nodes "${NSTAR_NODES}" --gssr "${AFTER[@]}"
    ENC_SERIES="$(join_dep "${ENC_SERIES}" "${LAST_IDS}")"
    run_submit submit_encoder.sh E3 --stage 4 --nodes "${NSTAR_NODES}" --gssr "${AFTER[@]}"
    ENC_SERIES="$(join_dep "${ENC_SERIES}" "${LAST_IDS}")"
    run_submit submit_encoder.sh E4 --nodes "${NSTAR_NODES}" --gssr "${AFTER[@]}"
    ENC_SERIES="$(join_dep "${ENC_SERIES}" "${LAST_IDS}")"
    run_submit submit_encoder.sh E5 --nodes 1 "${AFTER[@]}"
    ENC_SERIES="$(join_dep "${ENC_SERIES}" "${LAST_IDS}")"
fi

if [[ "${DO_RL}" -eq 1 ]]; then
    echo "=== 5. R3 learning check, R4 teacher refs, R5 video prompts ==="
    after_args "${RL_NSTAR_DEP:-${R0_IDS}}"
    run_submit submit_rl.sh R3 --gssr "${AFTER[@]}"
    RL_SERIES="$(join_dep "${RL_SERIES}" "${LAST_IDS}")"
    run_submit submit_rl.sh R4 --nodes 1 "${AFTER[@]}"
    RL_SERIES="$(join_dep "${RL_SERIES}" "${LAST_IDS}")"
    run_submit submit_rl.sh R4 --nodes 2 "${AFTER[@]}"
    RL_SERIES="$(join_dep "${RL_SERIES}" "${LAST_IDS}")"
    if [[ "${DO_R5}" -eq 1 ]]; then
        run_submit submit_rl.sh R5 "${AFTER[@]}"
        RL_SERIES="$(join_dep "${RL_SERIES}" "${LAST_IDS}")"
    fi
fi

echo
echo "=== submitted ==="
echo "  E6     ${E6_IDS}"
echo "  E0     ${E0_IDS}"
echo "  encoder series + E3–E5  ${ENC_SERIES}"
echo "  R0     ${R0_IDS}"
echo "  RL series + R3–R5       ${RL_SERIES}"
echo "  job list: ${JOBS_FILE}"
echo
echo "After the scaling jobs finish, pick n* from analysis.json and re-run"
echo "E3/E4 if --nstar-nodes=${NSTAR_NODES} was not the chosen scale:"
echo "  python3 scripts/pocus/analyse.py --evidence ${EVIDENCE}"
echo "  python3 scripts/pocus/wp5_results.py --analysis ${EVIDENCE}/analysis.json --out ${EVIDENCE}/WP5_results.md"
echo "  bash scripts/pocus/set_acls.sh"
echo
echo "LAUNCH_ALL_JOBS_FILE=${JOBS_FILE}"
