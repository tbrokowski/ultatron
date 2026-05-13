#!/usr/bin/env bash
# Submit Ultratron foundation-model hyperparameter tuning runs.
#
# Default behavior writes one train YAML + one Slurm wrapper per requested
# variant, but does not submit. Add --submit to enqueue jobs.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="${USER:-$(id -un)}"

HPARAM_CONFIG="configs/hparam_tuning.yaml"
MANIFEST="/capstor/scratch/cscs/${USER_NAME}/ultrasound/manifests/cardiac_breast_brain_maternal_lung_train.jsonl"
BASE_CONFIGS=""

PROFILE="single"
VARIANTS=""
VARIANTS_SET=0
RUN_PREFIX="family"
STEPS_OVERRIDE=""
PHASE_SPLIT=""
REDUCED_IMAGE_CROPS=0
IMAGE_GLOBAL_CROPS=""
IMAGE_LOCAL_CROPS=""
NUM_WORKERS_OVERRIDE=""

GENERATED_ROOT="/capstor/scratch/cscs/${USER_NAME}/ultrasound/hparam_tuning"
CONFIG_DIR="${GENERATED_ROOT}/configs"
SLURM_DIR="${GENERATED_ROOT}/slurm"
CKPT_ROOT="/capstor/scratch/cscs/${USER_NAME}/ultrasound/checkpoints/hparam_stability"
LOG_ROOT="/capstor/scratch/cscs/${USER_NAME}/ultrasound/logs/hparam_stability"

NODES=8
GPUS_PER_NODE=4
CPUS_PER_TASK=64
TIME_LIMIT="12:00:00"
PARTITION="normal"
ACCOUNT="a127"
EDF_ENV="${HOME}/.edf/ultatron.toml"
TRAIN_PYTHON="python3"
JOB_PREFIX="uhp"
NO_7B=1

SUBMIT=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./submit_hparam_tuning.sh [options]

Common:
  --submit                      Generate files and submit jobs with sbatch.
  --dry-run                     Print what would be generated/submitted.
  --profile NAME                single | sentinel | optimizer | masking | loss | architecture | trainability | all.
  --variants CSV                Explicit comma-separated variant names.
  --list-variants               Print available variants.

Run shape:
  --steps N                     Override curriculum.total_training_steps for all selected variants.
  --phase-split P1,P2,P3        Phase lengths as fractions or percentages. Example: 0.15,0.25,0.60.
  --reduced-image-crops         Shorthand for --image-global-crops 2 --image-local-crops 4 unless explicitly set.
  --image-global-crops N        Override transforms.image.n_global_crops.
  --image-local-crops N         Override transforms.image.n_local_crops.
  --num-workers N               Override loaders.num_workers.

Data/config:
  --hparam-config PATH          YAML defining defaults, variants, and profiles.
  --manifest PATH               Manifest JSONL passed to scripts/train.py.
  --base-configs CSV            Optional _base_ list for advanced runs. Empty by default.

Outputs:
  --repo-dir DIR                 Repo path used inside generated launch scripts.
  --run-prefix NAME             Prefix for generated run names. Default: family.
  --generated-root DIR           Root for generated train YAMLs and Slurm scripts.
  --config-dir DIR              Where generated train YAMLs are written.
  --slurm-dir DIR               Where generated Slurm scripts are written.
  --ckpt-root DIR               Root checkpoint directory. Variant subdirs are created below it.
  --log-root DIR                Root log directory. Variant subdirs are created below it.

Slurm/resources:
  --nodes N
  --gpus-per-node N
  --cpus-per-task N
  --time HH:MM:SS
  --partition NAME
  --account NAME                Use empty string to omit #SBATCH --account.
  --edf-env PATH                CSCS EDF environment for srun --environment.
  --train-python BIN            Python executable inside the EDF container.
  --with-7b                     Do not pass --no-7b to scripts/train.py.

Examples:
  # Generate, inspect, do not submit:
  ./submit_hparam_tuning.sh --profile sentinel

  # Submit the current best-practice stable baseline:
  ./submit_hparam_tuning.sh --submit --variants stable_baseline

  # Submit a first stability comparison set:
  ./submit_hparam_tuning.sh --submit --profile sentinel

  # Submit a 4h reduced-crop stability screen:
  ./submit_hparam_tuning.sh --submit --profile sentinel \
    --steps 2400 --time 04:00:00 --phase-split 0.15,0.25,0.60 \
    --reduced-image-crops

  # Submit a 12h full-crop confirmation run with the default 8k-step schedule:
  ./submit_hparam_tuning.sh --submit --profile sentinel \
    --steps 8000 --time 12:00:00 --phase-split 0.15,0.25,0.60

  # Use a different manifest and output root:
  ./submit_hparam_tuning.sh --submit \
    --manifest /path/to/train.jsonl \
    --ckpt-root /capstor/scratch/cscs/$USER/ultrasound/checkpoints/my_sweep \
    --log-root /capstor/scratch/cscs/$USER/ultrasound/logs/my_sweep
EOF
}

list_variants() {
  python3 - "$HPARAM_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
for name, spec in cfg.get("variants", {}).items():
    print(f"{name:<18} {spec.get('description', '')}")
PY
}

profile_variants() {
  python3 - "$HPARAM_CONFIG" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
profile = sys.argv[2]
profiles = cfg.get("profiles", {})
if profile not in profiles:
    raise SystemExit(f"Unknown profile: {profile}. Available: {', '.join(sorted(profiles))}")
print(",".join(profiles[profile]))
PY
}

quote() {
  printf '%q' "$1"
}

run_name_for_variant() {
  local variant="$1"
  if [[ -n "$RUN_PREFIX" ]]; then
    printf '%s_%s\n' "$RUN_PREFIX" "$variant"
  else
    printf '%s\n' "$variant"
  fi
}

validate_variant() {
  python3 - "$HPARAM_CONFIG" "$1" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
variant = sys.argv[2]
if variant not in cfg.get("variants", {}):
    raise SystemExit(f"Unknown variant: {variant}. Use --list-variants.")
PY
}

write_train_config() {
  local variant="$1" run_name="$2" config_path="$3"
  local base_csv
  base_csv="$BASE_CONFIGS"
  python3 - "$HPARAM_CONFIG" "$variant" "$run_name" "$config_path" "$base_csv" \
    "$STEPS_OVERRIDE" "$PHASE_SPLIT" "$REDUCED_IMAGE_CROPS" \
    "$IMAGE_GLOBAL_CROPS" "$IMAGE_LOCAL_CROPS" "$NUM_WORKERS_OVERRIDE" <<'PY'
import copy, sys, yaml
from pathlib import Path

(
    cfg_path,
    variant,
    run_name,
    out_path,
    base_csv,
    steps_override,
    phase_split,
    reduced_image_crops,
    image_global_crops,
    image_local_crops,
    num_workers_override,
) = sys.argv[1:12]
root = yaml.safe_load(open(cfg_path))
variants = root.get("variants", {})
if variant not in variants:
    raise SystemExit(f"Unknown variant: {variant}")

def deep_merge(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base

def parse_phase_split(value: str):
    raw = [x.strip() for x in value.replace("/", ",").split(",") if x.strip()]
    if len(raw) != 3:
        raise SystemExit("--phase-split must contain exactly three values: P1,P2,P3")
    parts = [float(x) for x in raw]
    total = sum(parts)
    if total > 1.5:
        parts = [x / 100.0 for x in parts]
        total = sum(parts)
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("--phase-split values must sum to 1.0 or 100")
    return parts

bases = [x.strip() for x in base_csv.split(",") if x.strip()]
out = {
    "experiment_name": f"hparam_stability_{run_name}",
}
if bases:
    out["_base_"] = bases
deep_merge(out, copy.deepcopy(root.get("defaults", {})))
deep_merge(out, variants[variant].get("overrides") or {})

if steps_override:
    out.setdefault("curriculum", {})["total_training_steps"] = int(steps_override)

if phase_split:
    p1, p2, p3 = parse_phase_split(phase_split)
    train = out.setdefault("train", {})
    train["phase1_frac"] = p1
    train["phase2_frac"] = p1 + p2
    train["phase3_frac"] = p1 + p2 + p3

if reduced_image_crops == "1":
    image_global_crops = image_global_crops or "2"
    image_local_crops = image_local_crops or "4"

if image_global_crops:
    out.setdefault("transforms", {}).setdefault("image", {})["n_global_crops"] = int(image_global_crops)
if image_local_crops:
    out.setdefault("transforms", {}).setdefault("image", {})["n_local_crops"] = int(image_local_crops)
if num_workers_override:
    out.setdefault("loaders", {})["num_workers"] = int(num_workers_override)

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Generated by submit_hparam_tuning.sh\n")
    f.write(f"# Hparam config: {cfg_path}\n")
    f.write(f"# Variant: {variant}\n")
    launch_overrides = []
    if steps_override:
        launch_overrides.append(f"steps={steps_override}")
    if phase_split:
        launch_overrides.append(f"phase_split={phase_split}")
    if reduced_image_crops == "1":
        launch_overrides.append("reduced_image_crops=true")
    if image_global_crops:
        launch_overrides.append(f"image_global_crops={image_global_crops}")
    if image_local_crops:
        launch_overrides.append(f"image_local_crops={image_local_crops}")
    if num_workers_override:
        launch_overrides.append(f"num_workers={num_workers_override}")
    if launch_overrides:
        f.write(f"# Launch overrides: {', '.join(launch_overrides)}\n")
    desc = variants[variant].get("description")
    if desc:
        f.write(f"# Description: {desc}\n")
    yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
PY
}

write_slurm_scripts() {
  local variant="$1" run_name="$2" config_path="$3" ckpt_dir="$4" log_dir="$5"
  local inner_path="${SLURM_DIR}/${run_name}.inner.sh"
  local sbatch_path="${SLURM_DIR}/${run_name}.sbatch"
  local no_7b_arg=""
  [[ "$NO_7B" -eq 1 ]] && no_7b_arg="--no-7b"

  cat > "$inner_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd $(quote "$REPO_DIR")
ulimit -c 0
export PYTHONPATH="$(quote "$REPO_DIR"):\${PYTHONPATH:-}"
# The EDF container exposes aws-ofi-nccl, but on this launch path the CXI/EFA
# device plugin is not usable. Remove that plugin path and force NCCL sockets.
export LD_LIBRARY_PATH="\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)"
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-8}"

mkdir -p $(quote "$ckpt_dir") $(quote "$log_dir")

export MASTER_ADDR="\${MASTER_ADDR:-\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)}"
export MASTER_PORT="\${MASTER_PORT:-29500}"

echo "variant      : ${variant}"
echo "run name     : ${run_name}"
echo "config       : ${config_path}"
echo "checkpoint   : ${ckpt_dir}"
echo "manifest     : ${MANIFEST}"
echo "node rank    : \${SLURM_NODEID:-0} / \${SLURM_NNODES:-1}"
echo "started      : \$(date)"

$(quote "$TRAIN_PYTHON") -m torch.distributed.run \\
    --nnodes="\${SLURM_NNODES:-1}" \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --node_rank="\${SLURM_NODEID:-0}" \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \\
    --rdzv_id="\${SLURM_JOB_ID:-${run_name}}" \\
    scripts/train.py \\
    --config $(quote "$config_path") \\
    --ckpt-dir $(quote "$ckpt_dir") \\
    --log-dir $(quote "$log_dir") \\
    ${no_7b_arg} \\
    --manifest $(quote "$MANIFEST")
EOF
  chmod +x "$inner_path"

  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=${JOB_PREFIX}_${run_name}"
    echo "#SBATCH --nodes=${NODES}"
    echo "#SBATCH --ntasks-per-node=1"
    echo "#SBATCH --gpus-per-node=${GPUS_PER_NODE}"
    echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
    echo "#SBATCH --time=${TIME_LIMIT}"
    [[ -n "$PARTITION" ]] && echo "#SBATCH --partition=${PARTITION}"
    [[ -n "$ACCOUNT" ]] && echo "#SBATCH --account=${ACCOUNT}"
    echo "#SBATCH --output=${log_dir}/%x_%j.out"
    echo "#SBATCH --error=${log_dir}/%x_%j.err"
    echo
    echo "set -euo pipefail"
    if [[ -n "$EDF_ENV" ]]; then
      echo "srun --ntasks-per-node=1 --environment=$(quote "$EDF_ENV") bash $(quote "$inner_path")"
    else
      echo "srun --ntasks-per-node=1 bash $(quote "$inner_path")"
    fi
  } > "$sbatch_path"
  chmod +x "$sbatch_path"

  printf '%s\n' "$sbatch_path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; VARIANTS_SET=1; shift 2 ;;
    --steps) STEPS_OVERRIDE="$2"; shift 2 ;;
    --phase-split) PHASE_SPLIT="$2"; shift 2 ;;
    --reduced-image-crops) REDUCED_IMAGE_CROPS=1; shift ;;
    --image-global-crops) IMAGE_GLOBAL_CROPS="$2"; shift 2 ;;
    --image-local-crops) IMAGE_LOCAL_CROPS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS_OVERRIDE="$2"; shift 2 ;;
    --hparam-config) HPARAM_CONFIG="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --base-configs) BASE_CONFIGS="$2"; shift 2 ;;
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --generated-root) GENERATED_ROOT="$2"; CONFIG_DIR="${2}/configs"; SLURM_DIR="${2}/slurm"; shift 2 ;;
    --config-dir) CONFIG_DIR="$2"; shift 2 ;;
    --slurm-dir) SLURM_DIR="$2"; shift 2 ;;
    --ckpt-root) CKPT_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE="$2"; shift 2 ;;
    --cpus-per-task) CPUS_PER_TASK="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --account) ACCOUNT="$2"; shift 2 ;;
    --edf-env) EDF_ENV="$2"; shift 2 ;;
    --train-python) TRAIN_PYTHON="$2"; shift 2 ;;
    --job-prefix) JOB_PREFIX="$2"; shift 2 ;;
    --with-7b) NO_7B=0; shift ;;
    --list-variants) list_variants; exit 0 ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$VARIANTS_SET" -eq 0 ]]; then
  VARIANTS="$(profile_variants "$PROFILE")"
fi

IFS=',' read -r -a variant_array <<< "$VARIANTS"

echo "profile       : ${PROFILE}"
echo "variants      : ${VARIANTS}"
echo "hparam config : ${HPARAM_CONFIG}"
echo "manifest      : ${MANIFEST}"
if [[ -n "$BASE_CONFIGS" ]]; then
  echo "base configs  : ${BASE_CONFIGS}"
else
  echo "base configs  : <none>"
fi
echo "config dir    : ${CONFIG_DIR}"
echo "slurm dir     : ${SLURM_DIR}"
echo "generated root: ${GENERATED_ROOT}"
echo "repo dir      : ${REPO_DIR}"
echo "ckpt root     : ${CKPT_ROOT}"
echo "log root      : ${LOG_ROOT}"
echo "resources     : nodes=${NODES}, gpus/node=${GPUS_PER_NODE}, cpus/task=${CPUS_PER_TASK}, time=${TIME_LIMIT}"
if [[ -n "$STEPS_OVERRIDE" ]]; then
  echo "steps override: ${STEPS_OVERRIDE}"
fi
if [[ -n "$PHASE_SPLIT" ]]; then
  echo "phase split   : ${PHASE_SPLIT}"
fi
if [[ "$REDUCED_IMAGE_CROPS" -eq 1 ]]; then
  echo "image crops   : reduced (${IMAGE_GLOBAL_CROPS:-2} global, ${IMAGE_LOCAL_CROPS:-4} local)"
elif [[ -n "$IMAGE_GLOBAL_CROPS" || -n "$IMAGE_LOCAL_CROPS" ]]; then
  echo "image crops   : ${IMAGE_GLOBAL_CROPS:-config default} global, ${IMAGE_LOCAL_CROPS:-config default} local"
else
  echo "image crops   : config default"
fi
if [[ -n "$NUM_WORKERS_OVERRIDE" ]]; then
  echo "num workers   : ${NUM_WORKERS_OVERRIDE}"
fi
echo "submit        : ${SUBMIT}"
echo "dry run       : ${DRY_RUN}"
echo

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$CONFIG_DIR" "$SLURM_DIR"
fi

for raw_variant in "${variant_array[@]}"; do
  variant="${raw_variant#"${raw_variant%%[![:space:]]*}"}"
  variant="${variant%"${variant##*[![:space:]]}"}"
  [[ -n "$variant" ]] || continue
  validate_variant "$variant"

  run_name="$(run_name_for_variant "$variant")"
  config_path="${CONFIG_DIR}/${run_name}.yaml"
  ckpt_dir="${CKPT_ROOT}/${run_name}"
  log_dir="${LOG_ROOT}/${run_name}"

  echo "[${run_name}] variant=${variant}"
  echo "  config: ${config_path}"
  echo "  ckpt  : ${ckpt_dir}"
  echo "  logs  : ${log_dir}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  dry-run: no files written"
    continue
  fi

  mkdir -p "$ckpt_dir" "$log_dir"
  write_train_config "$variant" "$run_name" "$config_path"
  sbatch_path="$(write_slurm_scripts "$variant" "$run_name" "$config_path" "$ckpt_dir" "$log_dir")"
  echo "  sbatch: ${sbatch_path}"

  if [[ "$SUBMIT" -eq 1 ]]; then
    job_id="$(sbatch --parsable "$sbatch_path")"
    echo "  job   : ${job_id}"
  else
    echo "  submit with: sbatch --parsable ${sbatch_path}"
  fi
  echo
done
