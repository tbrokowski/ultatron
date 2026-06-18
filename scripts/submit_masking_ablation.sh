#!/usr/bin/env bash
# =============================================================================
# submit_masking_ablation.sh  ·  Stage-1 image masking strategy ablation
# =============================================================================
#
# Submits three parallel 5k-step stage-1-only student pretrain jobs, one for
# each masking strategy:
#   freq     — ultrasound frequency-band masking (Fourier patch energy)
#   spatial  — standard random patch zeroing
#   both     — frequency first, then additional spatial union masking
#
# After all three jobs finish a lightweight CPU summary job reads metrics.jsonl
# from each run, compares losses, and writes a recommendation.
#
# Usage:
#   # Generate all files, print paths, do NOT submit (default):
#   bash scripts/submit_masking_ablation.sh
#
#   # Generate AND submit all three training jobs + dependent summary:
#   bash scripts/submit_masking_ablation.sh --submit
#
#   # Override cluster resources:
#   bash scripts/submit_masking_ablation.sh --submit --nodes 4 --time 06:00:00
#
#   # Dry run — print what would be generated, write nothing:
#   bash scripts/submit_masking_ablation.sh --dry-run
#
# Artifacts are written to:
#   logs/pretrain/masking_ablation_<sweep_id>/
#     configs/mask_{freq,spatial,both}.yaml
#     inner_mask_{freq,spatial,both}.sh
#     outer_mask_{freq,spatial,both}.sbatch
#     compare_masking.py
#     summary.sbatch
#     jobs.tsv
#   (after jobs complete)
#     summary.md
#     summary.json
#
# Checkpoints land in:
#   /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/MaskingAblation/{freq,spatial,both}/
# =============================================================================

set -euo pipefail

REPO_DIR="/users/tbrokowski/Ultatron"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
LOG_ROOT="${REPO_DIR}/logs/pretrain"
BASE_CONFIG="${REPO_DIR}/configs/student/student_pretrain_pilot.yaml"
CKPT_ROOT="/capstor/store/cscs/swissai/a127/ultrasound/checkpoints/MaskingAblation"

NODES=8
GPUS_PER_NODE=4
CPUS=32
NODE_MEM="460G"
TIME_LIMIT="04:00:00"
ABLATION_STEPS=5000

SUBMIT=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --submit)   SUBMIT=1; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        --nodes)    NODES="$2"; shift 2 ;;
        --gpus)     GPUS_PER_NODE="$2"; shift 2 ;;
        --time)     TIME_LIMIT="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))
SWEEP_ID="$(date -u +%Y%m%dT%H%M%SZ)_masking_ablation"
RUN_DIR="${LOG_ROOT}/masking_ablation_${SWEEP_ID}"
CONFIG_DIR="${RUN_DIR}/configs"
VARIANTS=("freq" "spatial" "both")

echo "================================================================"
echo " Ultatron — Stage-1 Masking Strategy Ablation (${ABLATION_STEPS} steps)"
echo " Sweep ID : ${SWEEP_ID}"
echo " Variants : ${VARIANTS[*]}"
echo " Resources: ${NODES} nodes × ${GPUS_PER_NODE} GPU = ${TOTAL_GPUS} total"
echo " Time     : ${TIME_LIMIT}"
echo " Artifacts: ${RUN_DIR}"
echo " Ckpt root: ${CKPT_ROOT}"
echo " Submit   : ${SUBMIT}"
echo " Dry-run  : ${DRY_RUN}"
echo "================================================================"
echo ""

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] Would create: ${RUN_DIR}/{configs,inner_*,outer_*,compare_masking.py,summary.sbatch,jobs.tsv}"
    for v in "${VARIANTS[@]}"; do
        echo "[dry-run]   configs/mask_${v}.yaml"
        echo "[dry-run]   inner_mask_${v}.sh"
        echo "[dry-run]   outer_mask_${v}.sbatch"
        echo "[dry-run]   ckpt_dir: ${CKPT_ROOT}/${v}/"
    done
    exit 0
fi

mkdir -p "${CONFIG_DIR}"

# Initialize jobs.tsv manifest
JOBS_TSV="${RUN_DIR}/jobs.tsv"
printf 'sweep_id\tvariant\tjob_id\tconfig_path\tckpt_dir\tlog_dir\tsbatch_path\tsubmitted_at_utc\n' \
    > "${JOBS_TSV}"

# ---------------------------------------------------------------------------
# Step 1: Generate per-variant YAML configs
# ---------------------------------------------------------------------------

generate_config() {
    local variant="$1"
    local config_out="$2"
    local ckpt_dir="$3"

    python3 - "${BASE_CONFIG}" "${variant}" "${config_out}" "${ckpt_dir}" "${ABLATION_STEPS}" <<'PYEOF'
import sys, copy, yaml
from pathlib import Path

base_path, variant, out_path, ckpt_dir, steps_str = sys.argv[1:6]
steps = int(steps_str)

with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

def deep_merge(base, override):
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base

# Common overrides: 5k steps, stage 1 only
common = {
    "training": {
        "total_steps": steps,
        # All steps fall in stage 1 (image-only); zero out remaining stages.
        # _stage_for_step builds cumulative bounds: [0, steps, steps, steps, steps]
        # so every step 0..(steps-1) resolves to stage 1.
        "stage_fracs": [1.0, 0.0, 0.0, 0.0],
    },
    "curriculum": {
        "total_training_steps": steps,
        # Stage-1 masking ablation: keep ALP tier-1 only (no tier-2 @ 25% → OOM).
        "alp_stage_fracs": [1.0, 1.0],
    },
    "loaders": {
        "num_workers": 2,
    },
    "student_data": {
        "num_workers": 2,
    },
    "pretrain": {
        "ckpt_dir": ckpt_dir,
        "save_stage_end_ckpts": False,  # no meaningful stage boundary at step 5000
    },
}

# Per-variant masking overrides
mask_overrides = {
    "freq": {
        "transforms": {
            "image": {
                "mask_strategy": "freq",
                "freq_mask": {"mask_ratio": 0.40},
            }
        }
    },
    "spatial": {
        "transforms": {
            "image": {
                "mask_strategy": "spatial",
                "mask_ratio": 0.40,
                "spatial_mask_ratio": 0.40,
            }
        }
    },
    "both": {
        "transforms": {
            "image": {
                "mask_strategy": "both",
                "mask_ratio": 0.40,
                "spatial_mask_ratio": 0.40,
                "freq_mask": {"mask_ratio": 0.40},
            }
        }
    },
}

deep_merge(cfg, common)
deep_merge(cfg, mask_overrides[variant])

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"# Generated by submit_masking_ablation.sh\n")
    f.write(f"# Base config: {base_path}\n")
    f.write(f"# Variant: {variant}  Steps: {steps}  Stage: 1 only (image)\n")
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

print(f"  Wrote config: {out_path}")
PYEOF
}

echo "--- Generating configs ---"
for variant in "${VARIANTS[@]}"; do
    config_path="${CONFIG_DIR}/mask_${variant}.yaml"
    ckpt_dir="${CKPT_ROOT}/${variant}"
    generate_config "${variant}" "${config_path}" "${ckpt_dir}"
done
echo ""

# ---------------------------------------------------------------------------
# Step 2: Generate inner + outer SLURM scripts for each variant
# ---------------------------------------------------------------------------

echo "--- Generating SLURM scripts ---"
job_ids=()

for variant in "${VARIANTS[@]}"; do
    config_path="${CONFIG_DIR}/mask_${variant}.yaml"
    ckpt_dir="${CKPT_ROOT}/${variant}"
    log_dir="${ckpt_dir}/logs"
    inner_path="${RUN_DIR}/inner_mask_${variant}.sh"
    outer_path="${RUN_DIR}/outer_mask_${variant}.sbatch"
    job_name="ultatron_mask_${variant}"

    cat > "${inner_path}" << INNER_EOF
#!/bin/bash
set -euo pipefail
ulimit -c 0
cd ${REPO_DIR}
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export US_STUDENT_CONFIG="${config_path}"
export US_STUDENT_MODE=pretrain

export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export NCCL_NET=Socket
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export OMP_NUM_THREADS=8

export MASTER_ADDR=\$(scontrol show hostnames "\${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

echo "================================================================"
echo " Ultatron — Masking Ablation: ${variant}"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Nodes  : \${SLURM_NNODES:-1}  GPUs/node: ${GPUS_PER_NODE}"
echo " Config : ${config_path}"
echo " Ckpt   : ${ckpt_dir}"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""

bash "${REPO_DIR}/scripts/ensure_deps.sh"
source "${REPO_DIR}/scripts/setup_hf_cache.sh"

python3 -m torch.distributed.run \\
    --nnodes=\${SLURM_NNODES} \\
    --nproc_per_node=${GPUS_PER_NODE} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint="\${MASTER_ADDR}:\${MASTER_PORT}" \\
    --rdzv_id=\${SLURM_JOB_ID} \\
    -m tests.dataset_adapters.student_training_smoke

echo ""
echo "================================================================"
echo " MASKING ABLATION (${variant}) COMPLETE — \$(date)"
echo "================================================================"
INNER_EOF
    chmod +x "${inner_path}"

    cat > "${outer_path}" << OUTER_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${NODE_MEM}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${RUN_DIR}/${job_name}_%j.out
#SBATCH --error=${RUN_DIR}/${job_name}_%j.err
set -euo pipefail
srun --ntasks-per-node=1 \\
     --environment=${EDF_ENV} \\
     bash ${inner_path}
OUTER_EOF
    chmod +x "${outer_path}"

    echo "  variant=${variant}"
    echo "    config : ${config_path}"
    echo "    inner  : ${inner_path}"
    echo "    sbatch : ${outer_path}"
    echo "    ckpt   : ${ckpt_dir}"

    job_id=""
    if [[ "${SUBMIT}" -eq 1 ]]; then
        job_id="$(sbatch --parsable "${outer_path}")"
        job_ids+=("${job_id}")
        echo "    job_id : ${job_id}"
    else
        echo "    submit with: sbatch ${outer_path}"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${SWEEP_ID}" "${variant}" "${job_id}" \
        "${config_path}" "${ckpt_dir}" "${log_dir}" \
        "${outer_path}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "${JOBS_TSV}"
    echo ""
done

# ---------------------------------------------------------------------------
# Step 3: Write comparison Python script
# ---------------------------------------------------------------------------

COMPARE_PY="${RUN_DIR}/compare_masking.py"

cat > "${COMPARE_PY}" << 'COMPARE_EOF'
#!/usr/bin/env python3
"""
compare_masking.py  —  Summarise masking ablation results and recommend
the optimal strategy for the full Ultatron stage-1 pretrain.

Usage:
    python3 compare_masking.py <jobs_tsv> <output_dir>

Reads metrics.jsonl from each variant's {ckpt_dir}/logs/ directory,
computes per-variant loss statistics, and writes:
    summary.md   — human-readable comparison table + recommendation
    summary.json — machine-readable full results
"""
import csv
import datetime as dt
import json
import math
import statistics
import sys
from pathlib import Path

LAST_N_STEPS = 500          # window for stable final-loss estimate
PRIMARY_METRIC = "loss_masked"   # most diagnostic for masking strategy quality
SECONDARY_METRIC = "loss"        # total loss as tiebreaker

def _read_metrics(log_dir: Path) -> dict:
    path = log_dir / "metrics.jsonl"
    result = {
        "metrics_path": str(path),
        "metrics_found": path.exists(),
        "n_rows": 0,
        "last_step": None,
        "per_step": [],   # list of {step, loss, loss_global, loss_patch, loss_masked, ...}
    }
    if not path.exists():
        return result

    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            result["n_rows"] += 1
            step = row.get("step")
            if step is None:
                continue
            result["last_step"] = step
            rows.append(row)

    result["per_step"] = rows
    return result


def _window_stats(rows: list, key: str, last_n: int) -> dict:
    """Mean/std/min of `key` over the last `last_n` rows that have the key."""
    vals = [
        r[key] for r in rows
        if key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key])
    ]
    if not vals:
        return {"mean": None, "std": None, "min": None, "n": 0}
    window = vals[-last_n:]
    return {
        "mean": statistics.mean(window),
        "std": statistics.stdev(window) if len(window) > 1 else 0.0,
        "min": min(window),
        "n": len(window),
    }


def _rank_variants(variants: list, stats: dict, metric: str) -> list:
    """Return variant names sorted best-first (lower is better for loss metrics)."""
    def _key(v):
        m = stats[v].get(metric, {}).get("mean")
        return m if m is not None else float("inf")
    return sorted(variants, key=_key)


def _fmt(val, digits=6):
    if val is None:
        return "n/a"
    return f"{val:.{digits}g}"


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <jobs_tsv> <output_dir>", file=sys.stderr)
        sys.exit(1)

    jobs_tsv = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(jobs_tsv.open("r", encoding="utf-8"), delimiter="\t"))

    variants_ordered = [r["variant"] for r in rows]
    variant_meta = {r["variant"]: r for r in rows}

    # Read metrics for each variant
    all_metrics = {}
    for row in rows:
        variant = row["variant"]
        log_dir = Path(row["ckpt_dir"]) / "logs"
        all_metrics[variant] = _read_metrics(log_dir)

    # Compute windowed stats over last LAST_N_STEPS logged rows
    TRACK_KEYS = ["loss", "loss_global", "loss_patch", "loss_masked",
                  "loss_proto", "loss_ema"]
    stats = {}
    for variant in variants_ordered:
        rows_v = all_metrics[variant]["per_step"]
        stats[variant] = {k: _window_stats(rows_v, k, LAST_N_STEPS) for k in TRACK_KEYS}

    # Rank variants
    ranked_primary = _rank_variants(variants_ordered, stats, PRIMARY_METRIC)
    ranked_secondary = _rank_variants(variants_ordered, stats, SECONDARY_METRIC)
    best_variant = ranked_primary[0]

    # ---------------------------------------------------------------------------
    # Determine recommendation rationale
    # ---------------------------------------------------------------------------
    rationale_parts = []
    pm = stats[best_variant][PRIMARY_METRIC]
    if pm["mean"] is not None:
        rationale_parts.append(
            f"`{best_variant}` achieves the lowest mean `{PRIMARY_METRIC}` "
            f"({_fmt(pm['mean'])}) over the final {pm['n']} logged steps."
        )
    sm = stats[best_variant][SECONDARY_METRIC]
    if sm["mean"] is not None:
        rank_in_secondary = ranked_secondary.index(best_variant) + 1
        rationale_parts.append(
            f"It ranks #{rank_in_secondary} on total loss ({_fmt(sm['mean'])})."
        )

    # Check if "both" wins: that suggests combined masking is most informative
    # Check if "freq" wins: ultrasound-specific structure is the key signal
    # Check if "spatial" wins: simple masking is sufficient, saves compute
    strategy_notes = {
        "freq": (
            "Frequency-band masking forces the model to reconstruct ultrasound-specific "
            "spectral structure, which is the most diagnostic signal in echography. "
            "Recommended when you want the student to learn tissue echogenicity patterns."
        ),
        "spatial": (
            "Standard spatial masking is compute-efficient and well-validated in iBOT/DINOv2. "
            "Winning here suggests the model benefits from contiguous spatial context more than "
            "frequency-guided difficulty. Consider for faster full runs."
        ),
        "both": (
            "Combined masking applies frequency guidance first then adds spatial tokens, "
            "creating the hardest reconstruction task and the most diverse loss signal. "
            "Winning here suggests the joint curriculum (matching the full-run pilot default) "
            "is already optimal — no config change needed for the 300k run."
        ),
    }
    rationale_parts.append(strategy_notes.get(best_variant, ""))

    # ---------------------------------------------------------------------------
    # Write summary.md
    # ---------------------------------------------------------------------------
    lines = []
    lines.append("# Stage-1 Masking Strategy Ablation — Summary")
    lines.append("")
    lines.append(f"- Generated UTC: `{dt.datetime.now(dt.timezone.utc).isoformat()}`")
    lines.append(f"- Ablation steps: `{LAST_N_STEPS}` (final-window average over last logged rows)")
    lines.append(f"- Primary ranking metric: `{PRIMARY_METRIC}` (lower is better)")
    lines.append("")

    # Data availability
    lines.append("## Data Availability")
    lines.append("")
    lines.append("| Variant | metrics.jsonl found | Logged rows | Last step |")
    lines.append("|---|:---:|---:|---:|")
    for v in variants_ordered:
        m = all_metrics[v]
        lines.append(
            f"| `{v}` | {'yes' if m['metrics_found'] else '**NO**'} | "
            f"`{m['n_rows']}` | `{m['last_step']}` |"
        )
    lines.append("")

    # Per-metric comparison table
    lines.append("## Loss Comparison (mean over last 500 logged steps)")
    lines.append("")
    header = "| Metric | " + " | ".join(f"`{v}`" for v in variants_ordered) + " | Best |"
    sep = "|---|" + "---:|" * len(variants_ordered) + "---|"
    lines.append(header)
    lines.append(sep)
    for key in TRACK_KEYS:
        row_parts = []
        vals = [(v, stats[v][key]["mean"]) for v in variants_ordered]
        valid_vals = [(v, x) for v, x in vals if x is not None]
        best_v = min(valid_vals, key=lambda t: t[1])[0] if valid_vals else None
        for v, val in vals:
            cell = _fmt(val)
            if v == best_v:
                cell = f"**{cell}**"
            row_parts.append(cell)
        lines.append(f"| `{key}` | " + " | ".join(row_parts) + f" | `{best_v or 'n/a'}` |")
    lines.append("")

    # Ranking
    lines.append("## Rankings")
    lines.append("")
    lines.append(f"**By `{PRIMARY_METRIC}` (primary):** " +
                 " > ".join(f"`{v}`" for v in ranked_primary))
    lines.append("")
    lines.append(f"**By `{SECONDARY_METRIC}` (total loss):** " +
                 " > ".join(f"`{v}`" for v in ranked_secondary))
    lines.append("")

    # Recommendation
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**Use `mask_strategy: {best_variant}` for the full 300k run.**")
    lines.append("")
    for part in rationale_parts:
        if part:
            lines.append(part)
            lines.append("")

    # Per-variant detail tables
    lines.append("## Per-Variant Detail (last-500-step window)")
    lines.append("")
    for v in variants_ordered:
        lines.append(f"### `{v}`")
        lines.append("")
        lines.append("| Metric | Mean | Std | Min | N |")
        lines.append("|---|---:|---:|---:|---:|")
        for key in TRACK_KEYS:
            s = stats[v][key]
            lines.append(
                f"| `{key}` | {_fmt(s['mean'])} | {_fmt(s['std'])} | "
                f"{_fmt(s['min'])} | {s['n']} |"
            )
        lines.append("")

    summary_md = out_dir / "summary.md"
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {summary_md}")

    # ---------------------------------------------------------------------------
    # Write summary.json
    # ---------------------------------------------------------------------------
    summary = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "last_n_steps_window": LAST_N_STEPS,
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "recommendation": best_variant,
        "ranked_by_primary": ranked_primary,
        "ranked_by_secondary": ranked_secondary,
        "rationale": " ".join(p for p in rationale_parts if p),
        "variants": {
            v: {
                "metrics_found": all_metrics[v]["metrics_found"],
                "n_rows": all_metrics[v]["n_rows"],
                "last_step": all_metrics[v]["last_step"],
                "stats": stats[v],
            }
            for v in variants_ordered
        },
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {summary_json}")

    # Print recommendation to stdout for Slurm log visibility
    print("")
    print("=" * 64)
    print(f"  RECOMMENDATION: mask_strategy = {best_variant}")
    for part in rationale_parts:
        if part:
            print(f"  {part}")
    print("=" * 64)


if __name__ == "__main__":
    main()
COMPARE_EOF
chmod +x "${COMPARE_PY}"
echo "--- Wrote comparison script: ${COMPARE_PY} ---"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Generate summary sbatch job
# ---------------------------------------------------------------------------

SUMMARY_SBATCH="${RUN_DIR}/summary.sbatch"

cat > "${SUMMARY_SBATCH}" << SUMMARY_EOF
#!/bin/bash
#SBATCH --job-name=ultatron_mask_ablation_summary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${RUN_DIR}/summary_%j.out
#SBATCH --error=${RUN_DIR}/summary_%j.err
set -euo pipefail
echo "Running masking ablation comparison..."
echo "Jobs TSV : ${JOBS_TSV}"
echo "Output   : ${RUN_DIR}"
python3 ${COMPARE_PY} ${JOBS_TSV} ${RUN_DIR}
echo ""
echo "Results written to:"
echo "  ${RUN_DIR}/summary.md"
echo "  ${RUN_DIR}/summary.json"
SUMMARY_EOF
chmod +x "${SUMMARY_SBATCH}"
echo "--- Wrote summary sbatch: ${SUMMARY_SBATCH} ---"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Submit summary job with dependency (if --submit)
# ---------------------------------------------------------------------------

summary_job_id=""
if [[ "${SUBMIT}" -eq 1 && "${#job_ids[@]}" -gt 0 ]]; then
    old_ifs="$IFS"
    IFS=':'
    dep_ids="${job_ids[*]}"
    IFS="${old_ifs}"
    summary_job_id="$(sbatch --parsable \
        --dependency="afterany:${dep_ids}" \
        "${SUMMARY_SBATCH}")"
    echo "Submitted summary job: ${summary_job_id}"
    echo "  Dependency: afterany:${dep_ids}"
    echo ""
elif [[ "${SUBMIT}" -eq 0 ]]; then
    echo "Summary job not submitted (use --submit to enqueue)."
    echo "  Manual trigger after training jobs complete:"
    echo "    sbatch ${SUMMARY_SBATCH}"
    echo ""
fi

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

echo "================================================================"
echo " Submission complete"
echo "  Sweep ID   : ${SWEEP_ID}"
echo "  Artifacts  : ${RUN_DIR}"
echo "  Configs    : ${CONFIG_DIR}"
echo "  Jobs TSV   : ${JOBS_TSV}"
echo ""
for variant in "${VARIANTS[@]}"; do
    echo "  mask_${variant}"
    echo "    config : ${CONFIG_DIR}/mask_${variant}.yaml"
    echo "    inner  : ${RUN_DIR}/inner_mask_${variant}.sh"
    echo "    sbatch : ${RUN_DIR}/outer_mask_${variant}.sbatch"
    echo "    ckpt   : ${CKPT_ROOT}/${variant}"
done
echo ""
if [[ -n "${summary_job_id}" ]]; then
    echo "  Summary job : ${summary_job_id}"
    echo "  Results will appear at:"
    echo "    ${RUN_DIR}/summary.md"
    echo "    ${RUN_DIR}/summary.json"
else
    echo "  Summary sbatch: ${SUMMARY_SBATCH}"
    echo "  Run after training: python3 ${COMPARE_PY} ${JOBS_TSV} ${RUN_DIR}"
fi
echo "================================================================"
