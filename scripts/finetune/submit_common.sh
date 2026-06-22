#!/usr/bin/env bash
# Shared Slurm submission for per-experiment finetune launchers (no login-node Python).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ACCOUNT="${ULTATRON_ACCOUNT:-a127}"
PARTITION="${ULTATRON_PARTITION:-normal}"
EDF_ENV="${ULTATRON_EDF_ENV:-${HOME}/.edf/ultatron.toml}"
LOG_ROOT="${REPO_DIR}/logs/finetune"
DEFAULT_COMPARISON_CONFIG="${REPO_DIR}/configs/finetune/comparison_representative.yaml"

STORE="/capstor/store/cscs/swissai/a127/ultrasound"
DEFAULT_BACKBONES=(student_stage1 resnet50 vit_b_16 dinov3_l biomedclip usfm echocare)

_die() { echo "[ERROR] $*" >&2; exit 1; }
_info() { echo "[INFO]  $*"; }

# Login-node python3 is 3.6; finetune needs >=3.10. Prefer 3.12 (compute EDF) then 3.11.
_FINETUNE_PYTHON=""
for _py in python3.12 python3.11 python3; do
  if command -v "${_py}" >/dev/null 2>&1 && "${_py}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    _FINETUNE_PYTHON="${_py}"
    break
  fi
done
[[ -n "${_FINETUNE_PYTHON}" ]] || _die "Need Python >=3.10 (tried python3.12, python3.11, python3)"

_resolve_output_dir() {
  "${_FINETUNE_PYTHON}" -c "
import yaml
from pathlib import Path
repo = Path('${REPO_DIR}')
cfg_path = Path('${COMPARISON_CONFIG}')
if not cfg_path.is_absolute():
    cfg_path = repo / cfg_path
with cfg_path.open() as f:
    out = yaml.safe_load(f).get('output_dir', 'results/finetune/representative')
out = Path(out)
print(out if out.is_absolute() else repo / out)
"
}

_build_backbones_arg() {
  BACKBONES_ARG=""
  if [[ "${ALL_BACKBONES}" -eq 1 ]]; then
    return
  fi
  if [[ ${#BACKBONES[@]} -eq 0 ]]; then
    # Custom comparison YAMLs define their own backbone set (e.g. openus
    # instead of echocare); use every entry unless --backbones is given.
    if [[ "${COMPARISON_CONFIG}" != "${DEFAULT_COMPARISON_CONFIG}" ]]; then
      return
    fi
    BACKBONES=("${DEFAULT_BACKBONES[@]}")
  fi
  BACKBONES_ARG="--backbones ${BACKBONES[*]}"
}

_build_finetune_cmd() {
  local experiments="${1:-}"
  local parallel="${2:-0}"
  local num_gpus="${3:-}"

  FINETUNE_ARGS=(
    "--comparison-config" "${COMPARISON_CONFIG}"
    "--busi-root"    "${STORE}/raw/breast/BUSI"
    "--echonet-root" "${STORE}/raw/cardiac/EchoNet-Dynamic"
    "--camus-root"   "${STORE}/raw/cardiac/CAMUS"
    "--benin-root"   "${STORE}/raw/lung/Benin_Videos"
    "--rsa-root"     "${STORE}/raw/lung/RSA_Videos"
  )
  if [[ -n "${experiments}" ]]; then
    FINETUNE_ARGS+=(--experiments ${experiments})
  fi
  if [[ "${EVAL_ONLY}" -eq 1 ]]; then
    FINETUNE_ARGS+=(--eval-only)
  fi
  if [[ -n "${BACKBONES_ARG}" ]]; then
    # shellcheck disable=SC2206
    FINETUNE_ARGS+=(${BACKBONES_ARG})
  fi
  if [[ "${parallel}" -eq 1 ]]; then
    FINETUNE_ARGS+=(--parallel-experiments)
    if [[ -n "${num_gpus}" ]]; then
      FINETUNE_ARGS+=(--num-gpus "${num_gpus}")
    fi
  else
    FINETUNE_ARGS+=(--no-parallel-experiments)
  fi
}

_write_inner_script() {
  local log_label="$1"
  local experiments="${2:-}"
  local parallel="${3:-0}"
  local num_gpus="${4:-}"

  _build_finetune_cmd "${experiments}" "${parallel}" "${num_gpus}"
  COMPARISON_OUTPUT="$(_resolve_output_dir)"
  mkdir -p "${LOG_ROOT}" "${COMPARISON_OUTPUT}"
  INNER_SCRIPT="${LOG_ROOT}/.inner_finetune_${log_label}.sh"

  {
    echo '#!/bin/bash'
    echo 'set -euo pipefail'
    echo 'ulimit -c 0'
    echo "cd ${REPO_DIR}"
    echo 'export PYTHONPATH="'"${REPO_DIR}"':${PYTHONPATH:-}"'
    echo 'export PYTHONUNBUFFERED=1'
    echo 'export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr '"'"':'"'"' '"'"'\\n'"'"' | grep -v '"'"'aws-ofi-nccl'"'"' | paste -sd '"'"':'"'"' -)'
    echo 'export OMP_NUM_THREADS=4'
    echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True'
    echo 'echo "================================================================"'
    echo "echo \" Ultatron — finetune (${log_label})\""
    echo 'echo " Job    : ${SLURM_JOB_ID:-local}"'
    echo 'echo " Node   : $(hostname)"'
    echo 'echo " Start  : $(date)"'
    echo 'nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \'
    echo '    | awk '"'"'{print " GPU     :", $0}'"'"' || echo " GPU     : nvidia-smi unavailable"'
    echo 'echo "================================================================"'
    echo 'echo ""'
    echo "echo \"Config     : ${COMPARISON_CONFIG}\""
    echo "echo \"Results dir : ${COMPARISON_OUTPUT}\""
    echo 'echo "Checkpoints  : /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/Finetune/"'
    echo 'echo ""'
    echo "bash \"${REPO_DIR}/scripts/ensure_deps.sh\""
    echo "bash \"${REPO_DIR}/scripts/ensure_ablation_deps.sh\""
    echo "source \"${REPO_DIR}/scripts/setup_hf_cache.sh\""
    echo "source \"${REPO_DIR}/scripts/finetune/runtime_python.sh\""
    echo 'FINETUNE_PYTHON="$(resolve_finetune_python)"'
    echo 'echo "Python: ${FINETUNE_PYTHON} ($(${FINETUNE_PYTHON} --version 2>&1))"'
    echo ''
    printf '"${FINETUNE_PYTHON}" scripts/finetune.py '
    printf '%q ' "${FINETUNE_ARGS[@]}"
    echo ''
    echo 'FT_EXIT=$?'
    echo '"${FINETUNE_PYTHON}" scripts/finetune/report.py --dashboard || true'
    echo 'echo ""'
    echo 'echo "================================================================"'
    echo 'echo " FINETUNE COMPLETE  exit=${FT_EXIT}  $(date)"'
    echo '[[ ${FT_EXIT} -eq 0 ]] && echo " Report  : '"${COMPARISON_OUTPUT}"'/comparison_report.md"'
    echo '[[ ${FT_EXIT} -eq 0 ]] && echo " Charts  : '"${COMPARISON_OUTPUT}"'/charts/"'
    echo '[[ ${FT_EXIT} -eq 0 ]] && echo " Dashboard: '"${REPO_DIR}"'/results/finetune/_dashboard/"'
    echo 'echo "================================================================"'
    echo 'exit ${FT_EXIT}'
  } > "${INNER_SCRIPT}"
  chmod +x "${INNER_SCRIPT}"
}

_submit_slurm() {
  local job_name="$1"
  local gpus="${2:-1}"
  local cpus="${3:-}"
  [[ -z "${cpus}" ]] && cpus=$(( gpus * 4 > 16 ? gpus * 4 : 16 ))

  OUTER_SCRIPT="$(mktemp /tmp/ultatron_outer_finetune_XXXXX.sh)"
  trap 'rm -f "${OUTER_SCRIPT}"' RETURN

  cat > "${OUTER_SCRIPT}" <<EOF
#!/bin/bash
set -euo pipefail
srun --ntasks-per-node=1 \\
     --environment=${EDF_ENV} \\
     bash ${INNER_SCRIPT}
EOF
  chmod +x "${OUTER_SCRIPT}"

  local -a sbatch_cmd=(
    sbatch
    "--job-name=${job_name}"
    --nodes=1
    --ntasks-per-node=1
    "--gpus-per-node=${gpus}"
    "--cpus-per-task=${cpus}"
    --time=12:00:00
    "--partition=${PARTITION}"
    "--account=${ACCOUNT}"
    "--output=${LOG_ROOT}/${job_name}_%j.out"
    "--error=${LOG_ROOT}/${job_name}_%j.err"
    --parsable
  )
  if [[ -n "${AFTER_JOB}" ]]; then
    sbatch_cmd+=("--dependency=afterok:${AFTER_JOB}")
  fi

  local job_id
  if ! job_id="$("${sbatch_cmd[@]}" "${OUTER_SCRIPT}")"; then
    _die "sbatch failed (GH200 nodes have at most 4 GPUs)."
  fi
  _info "Submitted ${job_name}  job_id=${job_id}"
  _info "  log: ${LOG_ROOT}/${job_name}_${job_id}.out"
}

_parse_submit_args() {
  COMPARISON_CONFIG="${DEFAULT_COMPARISON_CONFIG}"
  EVAL_ONLY=0
  LOCAL=0
  AFTER_JOB=""
  ALL_BACKBONES=0
  BACKBONES=()
  GPUS=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --comparison-config) COMPARISON_CONFIG="$2"; shift 2 ;;
      --eval-only)         EVAL_ONLY=1; shift ;;
      --local)             LOCAL=1; shift ;;
      --after-job)         AFTER_JOB="$2"; shift 2 ;;
      --all-backbones)     ALL_BACKBONES=1; shift ;;
      --backbones)
        shift
        while [[ $# -gt 0 && "$1" != --* ]]; do
          BACKBONES+=("$1")
          shift
        done
        ;;
      --gpus)              GPUS="$2"; shift 2 ;;
      -h|--help)
        echo "Usage: submit_finetune_experiment <name> [--local] [--eval-only] [--backbones ...]"
        echo "       [--all-backbones] [--comparison-config PATH] [--after-job JOBID] [--gpus N]"
        exit 0
        ;;
      *)
        _die "Unknown argument: $1"
        ;;
    esac
  done

  if [[ "${ALL_BACKBONES}" -eq 1 && ${#BACKBONES[@]} -gt 0 ]]; then
    _die "Use either --all-backbones or --backbones, not both"
  fi
  if [[ ! -f "${COMPARISON_CONFIG}" ]]; then
    _die "Comparison config not found: ${COMPARISON_CONFIG}"
  fi
}

submit_finetune_experiment() {
  local experiment="$1"
  shift
  _parse_submit_args "$@"
  _build_backbones_arg
  _write_inner_script "${experiment}" "${experiment}" 0 ""

  if [[ "${LOCAL}" -eq 1 ]]; then
    local log_path="${LOG_ROOT}/finetune_${experiment}_local_$(date +%Y%m%d_%H%M%S).log"
    _info "Running locally on $(hostname) (log → ${log_path})"
    bash "${INNER_SCRIPT}" 2>&1 | tee "${log_path}"
    return "${PIPESTATUS[0]}"
  fi

  _submit_slurm "ultatron_ft_${experiment}" 1
}

submit_finetune_all() {
  _parse_submit_args "$@"
  _build_backbones_arg

  local n_experiments=5
  if [[ -f "${COMPARISON_CONFIG}" ]]; then
    n_experiments="$(awk '/^experiments:/{f=1;next} f&&/^  - /{c++} f&&/^[^ #]/{exit} END{print c+0}' "${COMPARISON_CONFIG}")"
    [[ "${n_experiments}" -eq 0 ]] && n_experiments=5
  fi
  if [[ -n "${GPUS}" ]]; then
    local gpus="${GPUS}"
  else
    local gpus="${n_experiments}"
  fi
  (( gpus > 4 )) && gpus=4

  _write_inner_script "all" "" 1 "${gpus}"

  if [[ "${LOCAL}" -eq 1 ]]; then
    local log_path="${LOG_ROOT}/finetune_all_local_$(date +%Y%m%d_%H%M%S).log"
    _info "Running locally on $(hostname) (log → ${log_path})"
    bash "${INNER_SCRIPT}" 2>&1 | tee "${log_path}"
    return "${PIPESTATUS[0]}"
  fi

  _submit_slurm "ultatron_finetune" "${gpus}"
}
