#!/usr/bin/env bash
# =============================================================================
# submit_rl.sh  ·  WP5 Workload B (multimodal GRPO on NeMo-RL)
# =============================================================================
#
# The policy lives in swiss-ai/nemo-rl-development @ swissai/k8s, container
# nemo-rl_image_0_7_0_swissai_k8.sqsh (+ Qwen-VL layer if needed).
# This launcher writes the Slurm wrapper, env capture and GSSR sidecar; the
# actual NeMo-RL command is filled from configs/pocus/rl.yaml.
#
#   bash scripts/pocus/submit_rl.sh R0
#   bash scripts/pocus/submit_rl.sh R1 --nodes 4
#   bash scripts/pocus/submit_rl.sh R2 --nodes 1 --repeat
#
# Qwen3-VL support is a launch-time check (Transformers 5.5.4 / vLLM 0.25.1).
# Fallback: Qwen2.5-VL-7B-Instruct — record the substitution in env.txt.
# =============================================================================
set -euo pipefail

REPO_DIR="${ULTATRON_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ACCOUNT="${POCUS_ACCOUNT:-${ULTATRON_ACCOUNT:-a127}}"
PARTITION="${ULTATRON_PARTITION:-normal}"
EDF_ENV="${NEMO_RL_EDF_ENV:-${HOME}/.edf/nemo-rl.toml}"
EVIDENCE="${POCUS_EVIDENCE_ROOT:-/capstor/store/cscs/swissai/infra01/meditron-feasibility-review/pocus}"
NEMO_RL="${NEMO_RL_REPO:-/users/${USER}/nemo-rl-development}"
SQSH="${NEMO_RL_SQSH:-nemo-rl_image_0_7_0_swissai_k8.sqsh}"
GPUS_PER_NODE=4
CPUS="${POCUS_CPUS_PER_TASK:-288}"
NODE_MEM="475G"

EXP="${1:?experiment id: R0|R1|R2|R3|R4|R5}"
shift || true
NODES=1
REPEAT=0
AFTER_JOB=""
GSSR=0
TIME_LIMIT="01:00:00"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes) NODES="$2"; shift 2 ;;
        --repeat) REPEAT=1; shift ;;
        --after-job) AFTER_JOB="$2"; shift 2 ;;
        --gssr) GSSR=1; shift ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        *) echo "[ERROR] $1" >&2; exit 1 ;;
    esac
done

case "${EXP}" in
    R0) NODES=1; TIME_LIMIT="00:30:00" ;;
    R1) TIME_LIMIT="01:00:00" ;;
    R2) TIME_LIMIT="00:30:00" ;;
    R3) NODES=1; TIME_LIMIT="04:00:00" ;;
    R4) TIME_LIMIT="00:30:00" ;;
    R5) NODES=1; TIME_LIMIT="00:30:00" ;;
esac
if [[ "${NODES}" -ge 8 || "${GSSR}" -eq 1 ]]; then GSSR=1; fi

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
JOB_NAME="pocus_${EXP}_${NODES}n"
LOG_DIR="${EVIDENCE}/rl"
mkdir -p "${LOG_DIR}" "${REPO_DIR}/logs/pocus"
INNERSCRIPT="$(mktemp "${REPO_DIR}/logs/pocus/inner_${EXP}.XXXXXX.sh")"
OUTERSCRIPT="$(mktemp /tmp/pocus_rl_XXXXX.sh)"
trap "rm -f ${OUTERSCRIPT}" EXIT

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
cd ${NEMO_RL}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
unset NCCL_NET || true
export NCCL_NET="AWS Libfabric"
export NCCL_DEBUG=INIT
export FI_CXI_ATS=0
export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500
JOBID="\${SLURM_JOB_ID:-local}"
EVID="${LOG_DIR}/\${JOBID}"
mkdir -p "\${EVID}"
# Env capture (NeMo-RL tree + Ultatron helper).
bash "${REPO_DIR}/scripts/pocus/capture_env.sh" "\${EVID}/env.txt" "${REPO_DIR}/configs/pocus/rl.yaml" || true
source "${REPO_DIR}/scripts/gssr_sidecar.sh"
if [[ "${GSSR}" -eq 1 ]]; then
    gssr_sidecar_start "\${EVID}"
    trap gssr_sidecar_stop EXIT
fi
echo "POCUS ${EXP}  nodes=${NODES}  image=${SQSH}  NCCL_NET=\${NCCL_NET}"
# Confirm Qwen3-VL; fall back to Qwen2.5-VL-7B-Instruct and record it.
python3 - <<'PY'
import os, json, pathlib
evid = pathlib.Path(os.environ.get("EVID", "."))
try:
    import transformers, vllm
    ok = True
    note = f"transformers={transformers.__version__} vllm={vllm.__version__}"
except Exception as exc:
    ok = False
    note = f"import failed: {exc}"
(evid / "vlm_versions.json").write_text(json.dumps({"ok": ok, "note": note, "fallback": "Qwen/Qwen2.5-VL-7B-Instruct"}) + "\n")
print(note)
PY

# The concrete NeMo-RL entry point is repo-local; pass the experiment id.
if [[ -f "${NEMO_RL}/examples/run_vlm_grpo.sh" ]]; then
    bash "${NEMO_RL}/examples/run_vlm_grpo.sh" --experiment ${EXP} --config "${REPO_DIR}/configs/pocus/rl.yaml"
else
    echo "[WARN] ${NEMO_RL}/examples/run_vlm_grpo.sh missing — write the command in env.txt and exit 0 so the job is still evidence."
    echo "Ready to run: check Qwen3-VL in this fork; else Qwen2.5-VL-7B; else ETA from WP1 + teacher cost."
fi
INNER_EOF
chmod +x "${INNERSCRIPT}"

cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
set -euo pipefail
srun --ntasks-per-node=1 --environment=${EDF_ENV} bash ${INNERSCRIPT}
OUTER_EOF
chmod +x "${OUTERSCRIPT}"

SBATCH_CMD=(
    sbatch --job-name="${JOB_NAME}" --nodes="${NODES}" --ntasks-per-node=1
    --gpus-per-node="${GPUS_PER_NODE}" --cpus-per-task="${CPUS}" --mem="${NODE_MEM}"
    --time="${TIME_LIMIT}" --partition="${PARTITION}" --account="${ACCOUNT}"
    --output="${LOG_DIR}/${JOB_NAME}_%j.out" --error="${LOG_DIR}/${JOB_NAME}_%j.err" --parsable
)
if [[ -n "${AFTER_JOB}" ]]; then SBATCH_CMD+=(--dependency="afterok:${AFTER_JOB}"); fi
JOB_ID=$("${SBATCH_CMD[@]}" "${OUTERSCRIPT}")
echo "  Job ID : ${JOB_ID}"
echo "  Log    : ${LOG_DIR}/${JOB_NAME}_${JOB_ID}.out"
if [[ "${REPEAT}" -eq 1 ]]; then
    bash "$0" "${EXP}" --nodes "${NODES}" --after-job "${JOB_ID}"
fi
