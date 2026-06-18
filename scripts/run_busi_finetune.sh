#!/usr/bin/env bash
# =============================================================================
# run_busi_finetune.sh  ·  BUSI segmentation replication (USFM paper, MedIA 2024)
# =============================================================================
#
# Replicates the BUSI breast tumour vs background protocol from:
#   Jiao et al., "USFM: A universal ultrasound foundation model...", MedIA 2024
#   https://www.sciencedirect.com/science/article/pii/S1361841524001270
#
# Pipeline:
#   1. prepare_busi_usfm_split.py  → USFM SegBase layout (70/15/15 per class)
#   2. ensure_usfm_seg_deps.sh     → usdsgen + mmcv + mmsegmentation
#   3. Official USFM main.py       → UPerNet full fine-tune @ 512px, 400 epochs
#   4. Test-set metrics            → DSC, HD95, IoU, ACC, SEN (paper format)
#
# Variants (paper Table 2, Breast row):
#   upernet   UPerNet from scratch          (~66% DSC)
#   usfm      UPerNet + USFM_latest.pth    (~84% DSC)
#   all       both (default)
#
# Usage:
#   bash scripts/run_busi_finetune.sh --local
#   bash scripts/run_busi_finetune.sh --local --variant usfm --epochs 400
#   bash scripts/run_busi_finetune.sh --prepare-only
#   bash scripts/run_busi_finetune.sh --skip-prepare --local --variant usfm
#   bash scripts/run_busi_finetune.sh --after-job 2540153
#
# SLURM (default): submits 1 GPU × 24 h job to logs/busi_usfm/
# Local:          bash scripts/run_busi_finetune.sh --local
#
# Outputs:
#   dataset_exploration_outputs/busi_usfm/BUSI/          prepared split
#   dataset_exploration_outputs/busi_usfm/runs/{variant}/  checkpoints + metrics
#   dataset_exploration_outputs/busi_usfm/summary.json     collected test metrics
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCOUNT="a127"
PARTITION="normal"
EDF_ENV="/users/tbrokowski/.edf/ultatron.toml"
LOG_ROOT="${REPO_DIR}/logs/busi_usfm"

STORE="/capstor/store/cscs/swissai/a127/ultrasound"
BUSI_ROOT="${US_BUSI_ROOT:-${STORE}/raw/breast/BUSI}"
USFM_CKPT="${US_USFM_CHECKPOINT:-${STORE}/checkpoints/Ablations/USFM_latest.pth}"

USFM_VENDOR="${REPO_DIR}/finetune/backbones/vendor/usfm"
DATA_ROOT="${REPO_DIR}/dataset_exploration_outputs/busi_usfm/BUSI"
RUNS_ROOT="${REPO_DIR}/dataset_exploration_outputs/busi_usfm/runs"
SUMMARY_JSON="${REPO_DIR}/dataset_exploration_outputs/busi_usfm/summary.json"

VARIANT="all"
EPOCHS=400
BATCH_SIZE=16
NUM_WORKERS=4
IMG_SIZE=512
GPUS=1
SPLIT_SEED=42
TRAIN_RATIO=0.70
VAL_RATIO=0.15
RUN_LOCAL=false
PREPARE_ONLY=false
SKIP_PREPARE=false
FORCE_PREPARE=false
AFTER_JOB=""
TEST_ONLY=false
RESUME_CKPT=""

die()  { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO]  $*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --local)           RUN_LOCAL=true;            shift ;;
        --prepare-only)    PREPARE_ONLY=true;         shift ;;
        --skip-prepare)      SKIP_PREPARE=true;         shift ;;
        --force-prepare)     FORCE_PREPARE=true;        shift ;;
        --test-only)         TEST_ONLY=true;            shift ;;
        --variant)         VARIANT="$2";              shift 2 ;;
        --epochs)            EPOCHS="$2";               shift 2 ;;
        --batch-size)        BATCH_SIZE="$2";           shift 2 ;;
        --img-size)          IMG_SIZE="$2";             shift 2 ;;
        --gpus)              GPUS="$2";                 shift 2 ;;
        --busi-root)         BUSI_ROOT="$2";            shift 2 ;;
        --usfm-checkpoint)   USFM_CKPT="$2";            shift 2 ;;
        --split-seed)        SPLIT_SEED="$2";           shift 2 ;;
        --train-ratio)       TRAIN_RATIO="$2";          shift 2 ;;
        --val-ratio)         VAL_RATIO="$2";            shift 2 ;;
        --checkpoint)        RESUME_CKPT="$2";          shift 2 ;;
        --after-job)         AFTER_JOB="$2";            shift 2 ;;
        -h|--help)
            sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "Unknown argument: $1 (try --help)" ;;
    esac
done

case "${VARIANT}" in
    usfm|upernet|all) ;;
    *) die "--variant must be usfm, upernet, or all (got ${VARIANT})" ;;
esac

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"

# ── Inner compute script ──────────────────────────────────────────────────────
INNERSCRIPT="${LOG_ROOT}/.inner_busi_usfm.sh"

cat > "${INNERSCRIPT}" << INNER_EOF
#!/bin/bash
set -euo pipefail
ulimit -c 0
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export REPO_DIR="${REPO_DIR}"

export LD_LIBRARY_PATH=\$(echo "\${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd ':' -)
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

echo "================================================================"
echo " Ultatron — BUSI USFM replication"
echo " Job    : \${SLURM_JOB_ID:-local}"
echo " Node   : \$(hostname)"
echo " Start  : \$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \\
    | awk '{print " GPU     :", \$0}' || echo " GPU     : nvidia-smi unavailable"
echo "================================================================"
echo ""
echo "Variant(s) : ${VARIANT}"
echo "Epochs     : ${EPOCHS}"
echo "Batch size : ${BATCH_SIZE}"
echo "Img size   : ${IMG_SIZE}"
echo "BUSI root  : ${BUSI_ROOT}"
echo "Data out   : ${DATA_ROOT}"
echo "Runs out   : ${RUNS_ROOT}"
echo ""

die()  { echo "[ERROR] \$*" >&2; exit 1; }
info() { echo "[INFO]  \$*"; }

# ── 1. Prepare dataset ───────────────────────────────────────────────────────
if [[ "${SKIP_PREPARE}" != true ]]; then
    PREP_ARGS=(--busi-root "${BUSI_ROOT}" --out-root "${DATA_ROOT}"
               --seed ${SPLIT_SEED} --train-ratio ${TRAIN_RATIO} --val-ratio ${VAL_RATIO})
    [[ "${FORCE_PREPARE}" == true ]] && PREP_ARGS+=(--force)
    python3 scripts/prepare_busi_usfm_split.py "\${PREP_ARGS[@]}"
else
    info "Skipping data preparation (--skip-prepare)"
fi

[[ -d "${DATA_ROOT}/training_set/image" ]] || die "Prepared data not found: ${DATA_ROOT}"

if [[ "${PREPARE_ONLY}" == true ]]; then
    info "Prepare-only complete."
    exit 0
fi

# ── 2. USFM segmentation dependencies ───────────────────────────────────────
bash "${REPO_DIR}/scripts/ensure_usfm_seg_deps.sh"
bash "${REPO_DIR}/scripts/ensure_ablation_deps.sh" || true

# ── 3. Train / test ───────────────────────────────────────────────────────────
run_usfm_train() {
    local tag="\$1"
    local pretrained="\$2"
    local out="${RUNS_ROOT}/\${tag}"
    mkdir -p "\${out}"

    info "=== Training variant: \${tag} ==="
    info "Output: \${out}"

    cd "${USFM_VENDOR}"

    PRETRAIN_OVERRIDE="model.model_cfg.backbone.pretrained=null"
    if [[ -n "\${pretrained}" ]]; then
        [[ -f "\${pretrained}" ]] || die "Checkpoint not found: \${pretrained}"
        PRETRAIN_OVERRIDE="model.model_cfg.backbone.pretrained=\${pretrained}"
    fi

    if [[ "${TEST_ONLY}" == true ]]; then
        local ckpt="${RESUME_CKPT}"
        [[ -n "\${ckpt}" ]] || ckpt=\$(ls -t "\${out}"/best*.pth 2>/dev/null | head -1 || true)
        [[ -n "\${ckpt}" ]] || die "No checkpoint for test-only (set --checkpoint or train first)"
        info "Test-only mode: \${ckpt}"
        python3 main.py \\
            experiment=task/Seg \\
            data=Seg/BUSI \\
            model=Seg/Upernet \\
            mode=test \\
            "data.path.root=${DATA_ROOT}" \\
            "data.batch_size=${BATCH_SIZE}" \\
            "data.num_workers=${NUM_WORKERS}" \\
            "data.img_size=${IMG_SIZE}" \\
            "\${PRETRAIN_OVERRIDE}" \\
            "++output=\${out}" \\
            "model.resume=\${ckpt}" \\
            "train.only_resume_model=true" \\
            "L.devices=${GPUS}" \\
            "tag=\${tag}_test"
        return 0
    fi

    python3 main.py \\
        experiment=task/Seg \\
        data=Seg/BUSI \\
        model=Seg/Upernet \\
        mode=train \\
        "data.path.root=${DATA_ROOT}" \\
        "data.batch_size=${BATCH_SIZE}" \\
        "data.num_workers=${NUM_WORKERS}" \\
        "data.img_size=${IMG_SIZE}" \\
        "\${PRETRAIN_OVERRIDE}" \\
        "train.epochs=${EPOCHS}" \\
        "train.warmup_epochs=20" \\
        "train.base_lr=3e-4" \\
        "train.layer_decay=0.65" \\
        "train.val_freq=5" \\
        "train.weight_decay=0.05" \\
        "++output=\${out}" \\
        "L.devices=${GPUS}" \\
        "seed=${SPLIT_SEED}" \\
        "tag=\${tag}"

    local best=\$(ls -t "\${out}"/best*.pth 2>/dev/null | head -1 || true)
    [[ -n "\${best}" ]] || die "No best checkpoint saved under \${out}"

    info "Running test evaluation: \${best}"
    python3 main.py \\
        experiment=task/Seg \\
        data=Seg/BUSI \\
        model=Seg/Upernet \\
        mode=test \\
        "data.path.root=${DATA_ROOT}" \\
        "data.batch_size=${BATCH_SIZE}" \\
        "data.num_workers=${NUM_WORKERS}" \\
        "data.img_size=${IMG_SIZE}" \\
        "\${PRETRAIN_OVERRIDE}" \\
        "++output=\${out}" \\
        "model.resume=\${best}" \\
        "train.only_resume_model=true" \\
        "L.devices=${GPUS}" \\
        "tag=\${tag}_test"
}

EXIT=0
if [[ "${TEST_ONLY}" == true ]]; then
    case "${VARIANT}" in
        upernet) run_usfm_train "upernet_baseline" "" || EXIT=\$? ;;
        usfm)    run_usfm_train "usfm" "${USFM_CKPT}" || EXIT=\$? ;;
        all)     run_usfm_train "usfm" "${USFM_CKPT}" || EXIT=\$? ;;
    esac
elif [[ "${VARIANT}" == "upernet" || "${VARIANT}" == "all" ]]; then
    run_usfm_train "upernet_baseline" "" || EXIT=\$?
fi
if [[ ${EXIT} -eq 0 && "${TEST_ONLY}" != true && ( "${VARIANT}" == "usfm" || "${VARIANT}" == "all" ) ]]; then
    run_usfm_train "usfm" "${USFM_CKPT}" || EXIT=\$?
fi

# ── 4. Collect metrics ────────────────────────────────────────────────────────
python3 - <<'PY'
import csv
import json
from pathlib import Path

runs_root = Path("${RUNS_ROOT}")
summary_path = Path("${SUMMARY_JSON}")
paper_ref = {
    "upernet_baseline": {"DSC": 66.0, "HD95": 43.7, "IoU": 55.3, "ACC": 94.8, "SEN": 66.1},
    "usfm": {"DSC": 84.3, "HD95": 16.7, "IoU": 76.0, "ACC": 97.4, "SEN": 83.9},
    "ResUnet": {"DSC": 78.5, "HD95": 25.6, "IoU": 69.1, "ACC": 96.3, "SEN": 79.2},
}

def parse_segmetrics(run_dir):
    test_csv = run_dir / "test_segmetrics.csv"
    if test_csv.exists():
        out = {}
        for row in csv.DictReader(test_csv.open()):
            out[row["metrics"]] = {"mean": float(row["mean"]), "std": float(row["std"])}
        if out:
            return out
    for pattern in ("best*_test/segmetrics.csv", "best*/segmetrics.csv", "segmetrics.csv"):
        for p in run_dir.glob(pattern):
            rows = list(csv.DictReader(p.open()))
            if not rows:
                continue
            out = {}
            for row in rows:
                m = row.get("metrics") or row.get("metric")
                if not m:
                    continue
                try:
                    out[m] = {
                        "mean": float(row["mean"]),
                        "std": float(row["std"]),
                    }
                except (KeyError, ValueError):
                    pass
            if out:
                return out
    metrics_csv = run_dir / "allsegmetrics.csv"
    if metrics_csv.exists():
        rows = [r for r in csv.DictReader(metrics_csv.open()) if r.get("metrics")]
        if rows:
            last = rows[-1]
            return {last["metrics"]: {"mean": float(last["mean"]), "std": float(last["std"])}}
    test_dirs = sorted(run_dir.glob("best*_test"), key=lambda p: p.stat().st_mtime, reverse=True)
    for td in test_dirs:
        sm = td / "segmetrics.csv"
        if sm.exists():
            out = {}
            for row in csv.DictReader(sm.open()):
                out[row["metrics"]] = {"mean": float(row["mean"]), "std": float(row["std"])}
            if out:
                return out
    return None

summary = {
    "protocol": "USFM MedIA 2024 BUSI breast tumour vs background",
    "paper_url": "https://www.sciencedirect.com/science/article/pii/S1361841524001270",
    "data_root": "${DATA_ROOT}",
    "runs_root": str(runs_root),
    "variants": {},
    "paper_reference": paper_ref,
}

for variant_dir in sorted(runs_root.iterdir()) if runs_root.exists() else []:
    if not variant_dir.is_dir():
        continue
    tag = variant_dir.name
    metrics = parse_segmetrics(variant_dir)
    if metrics is None:
        continue
    dsc = metrics.get("Dice", {})
    summary["variants"][tag] = {
        "run_dir": str(variant_dir),
        "metrics": metrics,
        "DSC_pct": dsc.get("mean"),
        "DSC_std": dsc.get("std"),
    }

summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2))
print(f"[summary] Wrote {summary_path}")
for tag, v in summary["variants"].items():
    dsc = v.get("DSC_pct")
    ref = paper_ref.get(tag, {})
    ref_dsc = ref.get("DSC")
    extra = f"  (paper {ref_dsc}%)" if ref_dsc else ""
    print(f"  {tag}: DSC={dsc:.1f}%{extra}" if dsc is not None else f"  {tag}: no metrics")
PY

echo ""
echo "================================================================"
echo " BUSI USFM replication complete  exit=\${EXIT}  \$(date)"
echo " Summary : ${SUMMARY_JSON}"
echo " Runs    : ${RUNS_ROOT}"
echo "================================================================"
exit \${EXIT}
INNER_EOF

chmod +x "${INNERSCRIPT}"

# ── Local run ─────────────────────────────────────────────────────────────────
if [[ "${RUN_LOCAL}" == true ]]; then
    LOCAL_LOG="${LOG_ROOT}/busi_usfm_local_$(date +%Y%m%d_%H%M%S).log"
    info "Running locally (log → ${LOCAL_LOG})"
    bash "${INNERSCRIPT}" 2>&1 | tee "${LOCAL_LOG}"
    exit "${PIPESTATUS[0]}"
fi

# ── SLURM submit ──────────────────────────────────────────────────────────────
OUTERSCRIPT=$(mktemp /tmp/ultatron_outer_busi_usfm_XXXXX.sh)
trap "rm -f ${OUTERSCRIPT}" EXIT

cat > "${OUTERSCRIPT}" << OUTER_EOF
#!/bin/bash
set -euo pipefail
srun --ntasks-per-node=1 \\
     --environment=${EDF_ENV} \\
     bash ${INNERSCRIPT}
OUTER_EOF
chmod +x "${OUTERSCRIPT}"

DEPENDENCY_FLAG=""
if [[ -n "${AFTER_JOB}" ]]; then
    DEPENDENCY_FLAG="--dependency=afterok:${AFTER_JOB}"
    info "Will start after job ${AFTER_JOB} completes successfully."
fi

JOB_NAME="ultatron_busi_usfm"
TIME="24:00:00"

info "Submitting ${JOB_NAME}  (gpus=${GPUS}, time=${TIME}, variant=${VARIANT})"
JOB_ID=$(sbatch \
    ${DEPENDENCY_FLAG} \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --job-name="${JOB_NAME}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node="${GPUS}" \
    --cpus-per-task=16 \
    --time="${TIME}" \
    --output="${LOG_ROOT}/${JOB_NAME}_%j.out" \
    --error="${LOG_ROOT}/${JOB_NAME}_%j.err" \
    "${OUTERSCRIPT}")

info "Submitted job ${JOB_ID}"
info "  stdout: ${LOG_ROOT}/${JOB_NAME}_*.out"
info "  stderr: ${LOG_ROOT}/${JOB_NAME}_*.err"
info "  results: ${SUMMARY_JSON}"
