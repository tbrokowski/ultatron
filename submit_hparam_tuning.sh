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
SUMMARY_DIR="${GENERATED_ROOT}/summaries"
SUBMISSION_DIR="${GENERATED_ROOT}/submissions"
CKPT_ROOT="/capstor/scratch/cscs/${USER_NAME}/ultrasound/checkpoints/hparam_stability"
LOG_ROOT="/capstor/scratch/cscs/${USER_NAME}/ultrasound/logs/hparam_stability"
SWEEP_ID=""

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
SUMMARY_JOB=0
ALL_PARALLEL=0

usage() {
  cat <<'EOF'
Usage:
  ./submit_hparam_tuning.sh [options]

Common:
  --submit                      Generate files and submit jobs with sbatch.
  --all-parallel                Submit baseline + every configured isolated ablation as separate jobs.
  --summary                     Submit a dependent summary job after selected jobs finish.
  --no-summary                  Do not submit a dependent summary job.
  --dry-run                     Print what would be generated/submitted.
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
  --hparam-config PATH          YAML defining defaults and variants.
  --manifest PATH               Manifest JSONL passed to scripts/train.py.
  --base-configs CSV            Optional _base_ list for advanced runs. Empty by default.

Outputs:
  --repo-dir DIR                 Repo path used inside generated launch scripts.
  --run-prefix NAME             Prefix for generated run names. Default: family.
  --sweep-id ID                 Stable ID for this submission. Default: UTC timestamp + prefix + selected/all.
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
  --job-prefix NAME             Slurm job-name prefix. Default: uhp.
  --with-7b                     Do not pass --no-7b to scripts/train.py.

Examples:
  # Generate, inspect, do not submit:
  ./submit_hparam_tuning.sh --dry-run

  # Submit the current best-practice stable baseline:
  ./submit_hparam_tuning.sh --submit

  # Submit a hand-picked stability comparison set:
  ./submit_hparam_tuning.sh --submit --summary \
    --variants stable_baseline,lr_5e6,lr_5e5,video_mask_065,cross_loss_half,gram_w2

  # Submit every configured ablation in parallel and summarize after all finish:
  ./submit_hparam_tuning.sh --all-parallel --run-prefix curated_8k

  # Submit a 4h reduced-crop stability screen:
  ./submit_hparam_tuning.sh --submit \
    --variants stable_baseline,lr_5e6,lr_5e5,video_mask_065,cross_loss_half,gram_w2 \
    --steps 2400 --time 04:00:00 --phase-split 0.15,0.25,0.60 \
    --reduced-image-crops

  # Submit a 12h full-crop confirmation run with the default 8k-step schedule:
  ./submit_hparam_tuning.sh --submit \
    --variants stable_baseline,lr_5e6,lr_5e5,video_mask_065,cross_loss_half,gram_w2 \
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

all_variants() {
  python3 - "$HPARAM_CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(",".join(cfg.get("variants", {}).keys()))
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
  local inner_path="${RUN_SLURM_DIR}/${run_name}.inner.sh"
  local sbatch_path="${RUN_SLURM_DIR}/${run_name}.sbatch"
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
export HF_HUB_OFFLINE="\${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="\${TRANSFORMERS_OFFLINE:-1}"

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

write_summary_job() {
  local manifest_path="$1"
  local summary_dir="$2"
  local summary_py="${summary_dir}/summarize_hparam_sweep.py"
  local sbatch_path="${summary_dir}/summary.sbatch"

  mkdir -p "$summary_dir"
  cat > "$summary_py" <<'PY'
#!/usr/bin/env python3
import csv
import datetime as dt
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(manifest.open("r", encoding="utf-8"), delimiter="\t"))
job_ids = [row["job_id"] for row in rows if row.get("job_id")]
job_id_set = set(job_ids)


def _run_sacct(ids):
    if not ids:
        return {}
    cmd = [
        "sacct",
        "-P",
        "-n",
        "-j",
        ",".join(ids),
        "--format=JobIDRaw,JobName,State,ElapsedRaw,Start,End,ExitCode",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return {}
    if proc.returncode != 0:
        return {}
    statuses = {}
    for line in proc.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        jid, name, state, elapsed, start, end, exit_code = parts[:7]
        if jid not in job_id_set:
            continue
        try:
            elapsed_raw = int(elapsed) if elapsed else None
        except ValueError:
            elapsed_raw = None
        statuses[jid] = {
            "job_name": name,
            "state": state,
            "elapsed_sec": elapsed_raw,
            "start": start,
            "end": end,
            "exit_code": exit_code,
        }
    return statuses


def _metric_direction(name):
    lower = name.lower()
    min_hints = ("loss", "error", "rmse", "mae", "hd", "ece")
    return "min" if any(hint in lower for hint in min_hints) else "max"


def _read_metrics(log_dir):
    path = Path(log_dir) / "metrics.jsonl"
    result = {
        "metrics_path": str(path),
        "metrics_found": path.exists(),
        "n_metric_rows": 0,
        "first_metric_ts": None,
        "last_metric_ts": None,
        "last_step": None,
        "last_phase": None,
        "best": {},
    }
    if not path.exists():
        return result
    ignore = {"ts", "lr", "step", "phase", "stage", "n_align_pairs"}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            result["n_metric_rows"] += 1
            ts = row.get("ts")
            if isinstance(ts, (int, float)) and math.isfinite(ts):
                if result["first_metric_ts"] is None:
                    result["first_metric_ts"] = ts
                result["last_metric_ts"] = ts
            result["last_step"] = row.get("step", result["last_step"])
            result["last_phase"] = row.get("phase", result["last_phase"])
            for key, value in row.items():
                if key in ignore or not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                direction = _metric_direction(key)
                current = result["best"].get(key)
                better = (
                    current is None
                    or (direction == "min" and value < current["value"])
                    or (direction == "max" and value > current["value"])
                )
                if better:
                    result["best"][key] = {
                        "value": value,
                        "step": row.get("step"),
                        "phase": row.get("phase"),
                        "direction": direction,
                    }
    return result


statuses = _run_sacct(job_ids)
jobs = []
for row in rows:
    jid = row.get("job_id", "")
    metrics = _read_metrics(row.get("log_dir", ""))
    status = statuses.get(jid, {"state": "UNKNOWN", "elapsed_sec": None, "start": "", "end": "", "exit_code": ""})
    observed = None
    if metrics["first_metric_ts"] is not None and metrics["last_metric_ts"] is not None:
        observed = max(0.0, metrics["last_metric_ts"] - metrics["first_metric_ts"])
    jobs.append({
        **row,
        "slurm": status,
        "metrics": metrics,
        "observed_metric_runtime_sec": observed,
    })

elapsed = [job["slurm"].get("elapsed_sec") for job in jobs if isinstance(job["slurm"].get("elapsed_sec"), int)]
states = {}
for job in jobs:
    state = job["slurm"].get("state", "UNKNOWN")
    states[state] = states.get(state, 0) + 1
failed_states = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
failed = [job for job in jobs if job["slurm"].get("state", "").split()[0] in failed_states]

best_by_metric = {}
for job in jobs:
    for metric, spec in job["metrics"].get("best", {}).items():
        current = best_by_metric.get(metric)
        direction = spec["direction"]
        better = (
            current is None
            or (direction == "min" and spec["value"] < current["value"])
            or (direction == "max" and spec["value"] > current["value"])
        )
        if better:
            best_by_metric[metric] = {
                **spec,
                "run_name": job["run_name"],
                "variant": job["variant"],
                "job_id": job.get("job_id", ""),
            }


def _fmt_seconds(value):
    if value is None:
        return ""
    value = int(value)
    h, rem = divmod(value, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


summary = {
    "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "manifest": str(manifest),
    "n_jobs": len(jobs),
    "state_counts": states,
    "failed_jobs": failed,
    "average_slurm_elapsed_sec": statistics.mean(elapsed) if elapsed else None,
    "median_slurm_elapsed_sec": statistics.median(elapsed) if elapsed else None,
    "best_by_metric": best_by_metric,
    "jobs": jobs,
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

lines = []
lines.append("# Hparam Tuning Summary")
lines.append("")
lines.append(f"- Generated UTC: `{summary['generated_utc']}`")
lines.append(f"- Manifest: `{manifest}`")
lines.append(f"- Jobs: `{len(jobs)}`")
lines.append(f"- State counts: `{states}`")
lines.append(f"- Average Slurm runtime: `{_fmt_seconds(summary['average_slurm_elapsed_sec'])}`")
lines.append(f"- Median Slurm runtime: `{_fmt_seconds(summary['median_slurm_elapsed_sec'])}`")
lines.append("")
lines.append("## Jobs")
lines.append("")
lines.append("| Run | Variant | Job ID | State | Elapsed | End | Last step | Last phase |")
lines.append("|---|---|---:|---|---:|---|---:|---:|")
for job in jobs:
    slurm = job["slurm"]
    metrics = job["metrics"]
    lines.append(
        f"| `{job['run_name']}` | `{job['variant']}` | `{job.get('job_id', '')}` | "
        f"`{slurm.get('state', 'UNKNOWN')}` | `{_fmt_seconds(slurm.get('elapsed_sec'))}` | "
        f"`{slurm.get('end', '')}` | `{metrics.get('last_step')}` | `{metrics.get('last_phase')}` |"
    )
lines.append("")
if failed:
    lines.append("## Failed Or Non-Completed")
    lines.append("")
    for job in failed:
        lines.append(
            f"- `{job['run_name']}` job `{job.get('job_id', '')}`: "
            f"{job['slurm'].get('state')} exit `{job['slurm'].get('exit_code')}`"
        )
    lines.append("")
lines.append("## Best Metrics")
lines.append("")
lines.append("| Metric | Direction | Best value | Run | Step | Phase |")
lines.append("|---|---|---:|---|---:|---:|")
for metric in sorted(best_by_metric):
    spec = best_by_metric[metric]
    lines.append(
        f"| `{metric}` | `{spec['direction']}` | `{spec['value']:.6g}` | "
        f"`{spec['run_name']}` | `{spec.get('step')}` | `{spec.get('phase')}` |"
    )
lines.append("")
lines.append("## Artifact Roots")
lines.append("")
for key in ("config_path", "ckpt_dir", "log_dir", "sbatch_path"):
    lines.append(f"- `{key}`: see `{manifest.name}`")

(out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {out_dir / 'summary.md'}")
print(f"Wrote {out_dir / 'summary.json'}")
PY
  chmod +x "$summary_py"

  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=${JOB_PREFIX}_${SWEEP_ID}_summary"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --ntasks=1"
    echo "#SBATCH --cpus-per-task=4"
    echo "#SBATCH --time=00:30:00"
    [[ -n "$PARTITION" ]] && echo "#SBATCH --partition=${PARTITION}"
    [[ -n "$ACCOUNT" ]] && echo "#SBATCH --account=${ACCOUNT}"
    echo "#SBATCH --output=${summary_dir}/summary_%j.out"
    echo "#SBATCH --error=${summary_dir}/summary_%j.err"
    echo
    echo "set -euo pipefail"
    echo "python3 $(quote "$summary_py") $(quote "$manifest_path") $(quote "$summary_dir")"
  } > "$sbatch_path"
  chmod +x "$sbatch_path"
  printf '%s\n' "$sbatch_path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    --all-parallel) ALL_PARALLEL=1; SUBMIT=1; SUMMARY_JOB=1; VARIANTS_SET=0; shift ;;
    --summary) SUMMARY_JOB=1; shift ;;
    --no-summary) SUMMARY_JOB=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
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
    --sweep-id) SWEEP_ID="$2"; shift 2 ;;
    --generated-root) GENERATED_ROOT="$2"; CONFIG_DIR="${2}/configs"; SLURM_DIR="${2}/slurm"; SUMMARY_DIR="${2}/summaries"; SUBMISSION_DIR="${2}/submissions"; shift 2 ;;
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
  if [[ "$ALL_PARALLEL" -eq 1 ]]; then
    VARIANTS="$(all_variants)"
  else
    VARIANTS="stable_baseline"
  fi
fi

IFS=',' read -r -a variant_array <<< "$VARIANTS"

if [[ -z "$SWEEP_ID" ]]; then
  selection_label="selected"
  [[ "$ALL_PARALLEL" -eq 1 ]] && selection_label="all"
  safe_prefix="${RUN_PREFIX//[^A-Za-z0-9_.-]/_}"
  [[ -n "$safe_prefix" ]] || safe_prefix="run"
  SWEEP_ID="$(date -u +%Y%m%dT%H%M%SZ)_${safe_prefix}_${selection_label}"
fi
SWEEP_ID="${SWEEP_ID//[^A-Za-z0-9_.-]/_}"
RUN_PREFIX="${RUN_PREFIX//[^A-Za-z0-9_.-]/_}"

RUN_CONFIG_DIR="${CONFIG_DIR}/${SWEEP_ID}"
RUN_SLURM_DIR="${SLURM_DIR}/${SWEEP_ID}"
RUN_SUMMARY_DIR="${SUMMARY_DIR}/${SWEEP_ID}"
RUN_SUBMISSION_DIR="${SUBMISSION_DIR}/${SWEEP_ID}"
SUBMISSION_MANIFEST="${RUN_SUBMISSION_DIR}/jobs.tsv"

echo "variants      : ${VARIANTS}"
echo "all parallel  : ${ALL_PARALLEL}"
echo "sweep id      : ${SWEEP_ID}"
echo "hparam config : ${HPARAM_CONFIG}"
echo "manifest      : ${MANIFEST}"
if [[ -n "$BASE_CONFIGS" ]]; then
  echo "base configs  : ${BASE_CONFIGS}"
else
  echo "base configs  : <none>"
fi
echo "config dir    : ${RUN_CONFIG_DIR}"
echo "slurm dir     : ${RUN_SLURM_DIR}"
echo "summary dir   : ${RUN_SUMMARY_DIR}"
echo "submission tsv: ${SUBMISSION_MANIFEST}"
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
echo "summary job   : ${SUMMARY_JOB}"
echo "dry run       : ${DRY_RUN}"
echo

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$RUN_CONFIG_DIR" "$RUN_SLURM_DIR" "$RUN_SUMMARY_DIR" "$RUN_SUBMISSION_DIR"
  printf 'sweep_id\trun_name\tvariant\tjob_id\tconfig_path\tckpt_dir\tlog_dir\tsbatch_path\tsubmitted_at_utc\n' > "$SUBMISSION_MANIFEST"
fi

job_ids=()
generated_count=0

for raw_variant in "${variant_array[@]}"; do
  variant="${raw_variant#"${raw_variant%%[![:space:]]*}"}"
  variant="${variant%"${variant##*[![:space:]]}"}"
  [[ -n "$variant" ]] || continue
  validate_variant "$variant"

  run_name="$(run_name_for_variant "$variant")"
  config_path="${RUN_CONFIG_DIR}/${run_name}.yaml"
  ckpt_dir="${CKPT_ROOT}/${SWEEP_ID}/${run_name}"
  log_dir="${LOG_ROOT}/${SWEEP_ID}/${run_name}"

  echo "[${run_name}] variant=${variant}"
  echo "  config: ${config_path}"
  echo "  ckpt  : ${ckpt_dir}"
  echo "  logs  : ${log_dir}"
  generated_count=$((generated_count + 1))

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
    job_id="${job_id%%;*}"
    job_ids+=("$job_id")
    echo "  job   : ${job_id}"
  else
    job_id=""
    echo "  submit with: sbatch --parsable ${sbatch_path}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$SWEEP_ID" "$run_name" "$variant" "$job_id" "$config_path" "$ckpt_dir" "$log_dir" "$sbatch_path" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "$SUBMISSION_MANIFEST"
  echo
done

summary_sbatch=""
summary_job_id=""
if [[ "$DRY_RUN" -eq 0 && "$SUBMIT" -eq 1 && "$SUMMARY_JOB" -eq 1 && "${#job_ids[@]}" -gt 0 ]]; then
  summary_sbatch="$(write_summary_job "$SUBMISSION_MANIFEST" "$RUN_SUMMARY_DIR")"
  old_ifs="$IFS"
  IFS=':'
  dependency_ids="${job_ids[*]}"
  IFS="$old_ifs"
  summary_job_id="$(sbatch --parsable --dependency="afterany:${dependency_ids}" "$summary_sbatch")"
  summary_job_id="${summary_job_id%%;*}"
  echo "[summary] sbatch: ${summary_sbatch}"
  echo "[summary] job   : ${summary_job_id}"
  echo "[summary] dependency: afterany:${dependency_ids}"
  echo
fi

echo "Submission artifacts"
echo "  sweep id      : ${SWEEP_ID}"
echo "  generated jobs: ${generated_count}"
echo "  configs       : ${RUN_CONFIG_DIR}"
echo "  slurm scripts : ${RUN_SLURM_DIR}"
echo "  checkpoints   : ${CKPT_ROOT}/${SWEEP_ID}"
echo "  logs          : ${LOG_ROOT}/${SWEEP_ID}"
echo "  manifest      : ${SUBMISSION_MANIFEST}"
echo "  summaries     : ${RUN_SUMMARY_DIR}"
if [[ -n "$summary_job_id" ]]; then
  echo "  summary job   : ${summary_job_id}"
fi
