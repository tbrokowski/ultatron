#!/usr/bin/env bash
# =============================================================================
# download_prostate_mri_us_biopsy.sh  ·  Prostate MRI-US Biopsy (Kaggle Hub)
# =============================================================================
#
# Downloads dsptlp/prostate-mri-us-biopsy from Kaggle into:
#   /capstor/store/cscs/swissai/a127/ultrasound/raw/prostate/Prostate-MRI-US-Biopsy
#
# After download, all MRI DICOM series (Modality=MR) are deleted so only
# the ultrasound (Modality=US) data is retained.  MRI-registered biopsy
# overlay folders (*-BXmr-*) are also removed.
#
# Prerequisites:
#   Set your Kaggle credentials (found at https://www.kaggle.com/settings → API):
#     export KAGGLE_USERNAME="your_kaggle_username"
#     export KAGGLE_KEY="your_kaggle_api_key"
#   Or put them in ~/.kaggle/kaggle.json (mode 600):
#     {"username":"your_kaggle_username","key":"your_kaggle_api_key"}
#
# Usage:
#   sbatch scripts/download_prostate_mri_us_biopsy.sh       # submit as a batch job
#   bash   scripts/download_prostate_mri_us_biopsy.sh       # run interactively
#
# Monitor:
#   tail -f logs/download/prostate_mri_us_biopsy_<jobid>.out
#   squeue -u $USER
# =============================================================================
#SBATCH --job-name=prostate_mri_us_biopsy_download
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/users/tbrokowski/Ultatron/logs/download/prostate_mri_us_biopsy_%j.out
#SBATCH --error=/users/tbrokowski/Ultatron/logs/download/prostate_mri_us_biopsy_%j.err

set -euo pipefail

TARGET_DIR="/capstor/store/cscs/swissai/a127/ultrasound/raw/prostate/Prostate-MRI-US-Biopsy"
VENV="/users/tbrokowski/Ultatron/.venv"

mkdir -p "${TARGET_DIR}"
mkdir -p "/users/tbrokowski/Ultatron/logs/download"

echo "================================================================"
echo " Prostate-MRI-US-Biopsy download"
echo " Job    : ${SLURM_JOB_ID:-local}"
echo " Node   : $(hostname)"
echo " Target : ${TARGET_DIR}"
echo " Start  : $(date -Is)"
echo "================================================================"

source "${VENV}/bin/activate"

# Install kagglehub and pydicom if not already present
python3 -c "import kagglehub" 2>/dev/null || pip install -q kagglehub
python3 -c "import pydicom"   2>/dev/null || pip install -q pydicom

# kagglehub downloads to a cache dir; we redirect it to TARGET_DIR
export KAGGLE_CACHE_FOLDER="${TARGET_DIR}"

python3 - <<'PYEOF'
import os, shutil, pathlib, kagglehub, pydicom

target = pathlib.Path(os.environ["KAGGLE_CACHE_FOLDER"])

# ── Download ──────────────────────────────────────────────────────────────────
path = kagglehub.dataset_download("dsptlp/prostate-mri-us-biopsy")
print(f"Downloaded to cache: {path}")

# Move contents up to TARGET_DIR if they landed in a versioned subdirectory
if os.path.realpath(path) != os.path.realpath(str(target)):
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = target / item
        if not dst.exists():
            shutil.move(src, str(dst))
            print(f"Moved: {item}")
        else:
            print(f"Skip (exists): {item}")

print(f"\nDataset ready at: {target}")

# ── MRI pruning ───────────────────────────────────────────────────────────────
# Pass 1: remove DICOM series whose Modality tag is MR.
#   Walk all directories; only inspect leaf dirs that contain .dcm files
#   directly (glob, not rglob) to avoid double-counting nested series.

def get_modality(folder: pathlib.Path) -> str | None:
    """Return the DICOM Modality of the first readable .dcm in folder."""
    for dcm in folder.glob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
            return getattr(ds, "Modality", None)
        except Exception:
            continue
    return None

print("\n--- MRI pruning (pass 1: DICOM Modality=MR) ---")
removed_dcm, kept_dcm = 0, 0

# Collect all dirs first to avoid mutating the tree while walking
all_dirs = sorted(target.rglob("*"))
for series_dir in all_dirs:
    if not series_dir.is_dir() or not series_dir.exists():
        continue
    dcm_files = list(series_dir.glob("*.dcm"))
    if not dcm_files:
        continue
    modality = get_modality(series_dir)
    if modality == "MR":
        print(f"  Removing MR series : {series_dir.relative_to(target)}")
        shutil.rmtree(series_dir)
        removed_dcm += 1
    else:
        kept_dcm += 1

print(f"  DICOM series removed: {removed_dcm}  |  kept: {kept_dcm}")

# Pass 2: remove MRI-registered biopsy overlay folders (*-BXmr-*)
#   These folders contain BiopsyVectors.mrml files (no .dcm), so pass 1
#   does not catch them.  Pattern per TCIA naming convention.
print("\n--- MRI pruning (pass 2: BXmr biopsy overlay folders) ---")
removed_bxmr = 0
for bxmr_dir in sorted(target.rglob("*-BXmr-*")):
    if bxmr_dir.is_dir() and bxmr_dir.exists():
        print(f"  Removing BXmr dir  : {bxmr_dir.relative_to(target)}")
        shutil.rmtree(bxmr_dir)
        removed_bxmr += 1

print(f"  BXmr overlay dirs removed: {removed_bxmr}")

# ── Summary ───────────────────────────────────────────────────────────────────
remaining = sum(1 for _ in target.rglob("*") if _.is_file())
print(f"\n================================================================")
print(f" Pruning complete.")
print(f"   MR DICOM series removed : {removed_dcm}")
print(f"   BXmr overlay dirs removed: {removed_bxmr}")
print(f"   Ultrasound series kept  : {kept_dcm}")
print(f"   Total files remaining   : {remaining}")
print(f"   Final location          : {target}")
print(f"================================================================")
PYEOF

echo ""
echo "================================================================"
echo " Finished : $(date -Is)"
echo "================================================================"
