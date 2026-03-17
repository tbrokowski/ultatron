# CSCS Data Storage & Staging Guide
### Ultrasound Foundation Model — Dataset Management on Capstor

---

## Our Storage Paths

These are the two fixed paths you will use throughout this project. Replace `$USER` with your CSCS username where indicated.

| Purpose | Path |
|---|---|
| **Store** (permanent archive) | `/capstor/store/cscs/swissai/a127/ultrasound/` |
| **Scratch** (active training) | `/capstor/scratch/cscs/$USER/ultrasound/` |

> **Note on Store:** The `/capstor/store/cscs/swissai/a127/` directory is provisioned by CSCS for our project allocation (SwissAI project a127). You can freely create subdirectories inside `/capstor/store/cscs/swissai/a127/ultrasound/` using `mkdir`.

---

## Overview

CSCS provides two storage systems that serve different purposes. 

| Storage | Path | Purpose | Speed | Persistent? |
|---|---|---|---|---|
| **Store** | `/capstor/store/cscs/swissai/a127/ultrasound/` | Long-term archive | Slow | ✅ Yes — never purged |
| **Scratch** | `/capstor/scratch/cscs/$USER/ultrasound/` | Active training I/O | Fast | ⚠️ Purged after ~30 days inactivity |

**The golden rule: Store is your archive. Scratch is where you train.**

You never train directly from Store — it is too slow for the random-access patterns of a dataloader reading from 120 datasets simultaneously. Instead, you download data once to Store, then *stage* (copy) it to Scratch before running a job.

---

## Directory Structure

### Store — Permanent Archive

This is where all raw data lives permanently. It survives between projects, cluster maintenance, and long gaps between jobs.

```
/capstor/store/cscs/swissai/a127/ultrasound/
├── raw/                          # Original downloaded datasets — never modify these
│   ├── cardiac/
│   │   ├── CAMUS/
│   │   ├── EchoNet-Dynamic/
│   │   ├── EchoNet-LVH/
│   │   └── MIMIC-IV-ECHO/
│   ├── lung/
│   │   ├── COVIDx-US/
│   │   ├── LUS-multicenter-2025/
│   │   └── POCUS-LUS/
│   ├── breast/
│   │   ├── BUS-BRA/
│   │   ├── BUSI/
│   │   └── BrEaST/
│   ├── thyroid/
│   │   ├── TN3K/
│   │   ├── TN5000/
│   │   ├── DDTI/
│   │   └── TNSCUI/
│   ├── fetal/
│   │   ├── FETAL_PLANES_DB/
│   │   ├── HC18/
│   │   └── ACOUSLIC-AI/
│   ├── kidney/
│   ├── liver/
│   ├── ovarian/
│   ├── prostate/
│   ├── musculoskeletal/
│   └── multi_organ/
│
├── manifests/                    # Generated manifest .jsonl files
│   ├── us_foundation_train.jsonl
│   ├── us_foundation_val.jsonl
│   ├── us_foundation_test.jsonl
│   └── per_dataset/
│       ├── CAMUS_train.jsonl
│       └── ...
│
└── checkpoints/                  # Completed phase checkpoints
    ├── phase1/
    ├── phase2/
    └── phase3/
```

### Scratch — Active Training Workspace

This mirrors Store but lives on a high-speed Lustre filesystem optimised for training I/O. Files here are **purged if untouched for ~30 days**, so do not treat Scratch as permanent storage.

```
/capstor/scratch/cscs/$USER/ultrasound/
├── raw/                          # Staged copy of datasets for the current training run
│   ├── cardiac/
│   ├── lung/
│   ├── breast/
│   └── ...                       # Same structure as Store/raw/
│
├── manifests/                    # Working copy of manifests
│   ├── us_foundation_train.jsonl
│   └── ...
│
├── alp_cache/                    # ALP saliency maps written during Phase 3 training
│   ├── CA/                       # Bucketed by first 2 chars of sample_id
│   │   └── CAMUS__patient0001__0.pt
│   └── EC/
│       └── EchoNet-Dynamic__0X1A2B3C__0.pt
│
└── checkpoints/                  # In-progress training checkpoints
    └── current_run/
```

> **Why bucket the ALP cache?** The `alp_cache/` directory can accumulate 500,000+ small `.pt` files during Phase 3 training. Putting them all in one flat directory degrades Lustre performance. Bucketing by the first 2 characters of `sample_id` keeps each subdirectory manageable.

---

## Step-by-Step: First-Time Setup

### Step 1 — Create the directory structure in Store

Run this once when you first join the project. Check whether these directories already exist before running — if a teammate has already set them up, you can skip this step.

```bash
STORE=/capstor/store/cscs/swissai/a127/ultrasound

mkdir -p $STORE/raw/cardiac
mkdir -p $STORE/raw/lung
mkdir -p $STORE/raw/breast
mkdir -p $STORE/raw/thyroid
mkdir -p $STORE/raw/fetal
mkdir -p $STORE/raw/kidney
mkdir -p $STORE/raw/liver
mkdir -p $STORE/raw/ovarian
mkdir -p $STORE/raw/prostate
mkdir -p $STORE/raw/musculoskeletal
mkdir -p $STORE/raw/multi_organ
mkdir -p $STORE/manifests/per_dataset
mkdir -p $STORE/checkpoints/phase1
mkdir -p $STORE/checkpoints/phase2
mkdir -p $STORE/checkpoints/phase3
```

### Step 2 — Create your personal Scratch workspace

Each student sets up their own Scratch directory. Scratch is per-user, so everyone does this independently.

```bash
SCRATCH=/capstor/scratch/cscs/$USER/ultrasound

mkdir -p $SCRATCH/raw
mkdir -p $SCRATCH/manifests/per_dataset
mkdir -p $SCRATCH/alp_cache
mkdir -p $SCRATCH/checkpoints/current_run
```

---

## Downloading Datasets to Store

All datasets go into Store first. **Never download directly to Scratch** — it will eventually be purged.

### General pattern

```bash
# Set the Store path
STORE=/capstor/store/cscs/swissai/a127/ultrasound

# Navigate to the correct anatomy family before downloading
cd $STORE/raw/<anatomy_family>/
```

### Datasets with direct download links (wget)

```bash
STORE=/capstor/store/cscs/swissai/a127/ultrasound

# Example: HC18 (fetal head)
cd $STORE/raw/fetal/
wget -r -np -nH --cut-dirs=3 -P HC18/ https://zenodo.org/record/<record_id>/files/

# Example: COVIDx-US (lung)
cd $STORE/raw/lung/
wget -O COVIDx-US.zip https://github.com/nrc-cnrc/COVID-US/archive/refs/heads/master.zip
unzip COVIDx-US.zip -d COVIDx-US/
rm COVIDx-US.zip
```

### Datasets requiring Kaggle CLI

First, configure your Kaggle credentials (one-time setup):

```bash
pip install kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/   # Download this from Kaggle → Settings → API → Create Token
chmod 600 ~/.kaggle/kaggle.json
```

Then download:

```bash
STORE=/capstor/store/cscs/swissai/a127/ultrasound

# Example: BUSI (breast)
cd $STORE/raw/breast/
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p BUSI/ --unzip

# Example: TN3K (thyroid)
cd $STORE/raw/thyroid/
kaggle datasets download -d haifangong/thyroid-ultrasound-cine-clip -p TN3K/ --unzip
```

### Datasets requiring PhysioNet credentials (MIMIC-IV-ECHO)

MIMIC requires credentialed access. You must complete the PhysioNet data use agreement and CITI training before downloading.

```bash
STORE=/capstor/store/cscs/swissai/a127/ultrasound

cd $STORE/raw/cardiac/
wget -r -N -c -np \
  --user <your_physionet_username> \
  --ask-password \
  -P MIMIC-IV-ECHO/ \
  https://physionet.org/files/mimic-iv-echo/0.1/
```

### Uploading from your local machine

If you have already downloaded a dataset locally and need to transfer it to CSCS:

```bash
# Upload a single dataset
rsync -avh --progress \
  /local/path/to/EchoNet-Dynamic/ \
  <username>@daint.cscs.ch:/capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/EchoNet-Dynamic/

# Upload an entire anatomy family at once
rsync -avh --progress \
  /local/path/to/thyroid/ \
  <username>@daint.cscs.ch:/capstor/store/cscs/swissai/a127/ultrasound/raw/thyroid/
```

> **Always use `rsync` instead of `scp`.** If the connection drops, `rsync` will resume from where it left off. `scp` will restart the entire transfer from scratch.

### Verify the download

After downloading any dataset, do a quick sanity check before moving on:

```bash
STORE=/capstor/store/cscs/swissai/a127/ultrasound

# Check total size
du -sh $STORE/raw/cardiac/EchoNet-Dynamic/

# Count files
find $STORE/raw/cardiac/EchoNet-Dynamic/ -type f | wc -l

# Spot-check the internal structure looks correct
ls $STORE/raw/cardiac/EchoNet-Dynamic/
# Expected output: Videos/  FileList.csv  VolumeTracings.csv
```

---

## Staging Data from Store to Scratch

Before each training run, copy the datasets you need from Store to Scratch. You do not need to stage everything at once — only what your current training phase requires.

### Stage a single dataset

```bash
STORE=/capstor/store/cscs/swissai/a127/ultrasound
SCRATCH=/capstor/scratch/cscs/$USER/ultrasound

rsync -avh --progress \
  $STORE/raw/cardiac/CAMUS/ \
  $SCRATCH/raw/cardiac/CAMUS/
```

### Stage all datasets for a training run

Save the following as `stage_datasets.sh` and run it before submitting a training job:

```bash
#!/bin/bash
# stage_datasets.sh
# Run this before launching a training job to copy data from Store to Scratch.

STORE=/capstor/store/cscs/swissai/a127/ultrasound
SCRATCH=/capstor/scratch/cscs/$USER/ultrasound

DATASETS=(
  # Cardiac
  "cardiac/CAMUS"
  "cardiac/EchoNet-Dynamic"
  "cardiac/EchoNet-LVH"
  "cardiac/MIMIC-IV-ECHO"
  # Lung
  "lung/COVIDx-US"
  "lung/LUS-multicenter-2025"
  # Breast
  "breast/BUS-BRA"
  "breast/BUSI"
  # Thyroid
  "thyroid/TN3K"
  "thyroid/TN5000"
  "thyroid/DDTI"
  # Fetal
  "fetal/FETAL_PLANES_DB"
  "fetal/HC18"
  # Add more datasets here as needed
)

for DATASET in "${DATASETS[@]}"; do
  echo "Staging $DATASET ..."
  mkdir -p $SCRATCH/raw/$DATASET
  rsync -ah --progress $STORE/raw/$DATASET/ $SCRATCH/raw/$DATASET/
  echo "Done: $DATASET"
done

# Also stage manifests
rsync -avh \
  $STORE/manifests/ \
  $SCRATCH/manifests/

echo "All datasets staged successfully."
```

---

## Dataset Adapter Exploration Scripts

Once BUSI and COVIDx-US are downloaded to Store (see above), you can run
exploratory scripts to sanity-check adapters, manifests, and transforms and
to generate PNG visualisations of images / videos and crops.

From the project root:

```bash
# BUSI breast ultrasound (images + masks + SSL crops)
python -m tests.dataset_adapters.busi_explore

# COVIDx-US lung ultrasound (video frames + SSL clips)
python -m tests.dataset_adapters.covidx_us_explore
```

These scripts expect the canonical Store locations:

- BUSI: `/capstor/store/cscs/swissai/a127/ultrasound/raw/breast/BUSI`
- COVIDx-US: `/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/COVIDx-US`

You can override these with:

```bash
export US_BUSI_ROOT=/custom/path/to/BUSI
export US_COVIDX_ROOT=/custom/path/to/COVIDx-US
```

Each script will:

- Build a small per-dataset manifest in the current working directory
  under a shared `dataset_exploration_outputs/` root
  (for example, `dataset_exploration_outputs/busi/`).
- Print basic manifest statistics and label distributions.
- Save PNG snapshots of:
  - Raw BUSI images and masks.
  - BUSI SSL crops (global/local views).
  - COVIDx-US raw frames (first/middle/last frame).
  - COVIDx-US transformed clip frames from the video SSL pipeline.

Open the generated PNGs locally (or via VS Code / Jupyter) to visually
inspect that adapters, label mappings, and transforms behave as expected
before launching larger training runs.

Run it:

```bash
chmod +x stage_datasets.sh
./stage_datasets.sh
```

### Set Lustre striping for large video datasets

For datasets containing large video files (EchoNet, MIMIC-IV-ECHO), set Lustre striping **before** copying data in. This distributes large files across multiple storage targets for faster parallel reads during training.

```bash
SCRATCH=/capstor/scratch/cscs/$USER/ultrasound

# Set striping BEFORE staging — this must come first
lfs setstripe -c 4 $SCRATCH/raw/cardiac/EchoNet-Dynamic/
lfs setstripe -c 4 $SCRATCH/raw/cardiac/MIMIC-IV-ECHO/

# Verify striping is set correctly
lfs getstripe $SCRATCH/raw/cardiac/EchoNet-Dynamic/

# Then stage the data
rsync -avh --progress \
  /capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/EchoNet-Dynamic/ \
  $SCRATCH/raw/cardiac/EchoNet-Dynamic/
```

> **Important:** Set striping on the directory *before* copying files into it. Striping only applies to files created after the stripe count is set — it has no effect on files that already exist.

---

## Keeping Store and Scratch in Sync

After a training phase completes, archive outputs back to Store so nothing is lost when Scratch is eventually purged.

```bash
STORE=/capstor/store/cscs/swissai/a127/ultrasound
SCRATCH=/capstor/scratch/cscs/$USER/ultrasound

# Archive completed phase checkpoint back to Store
rsync -avh --progress \
  $SCRATCH/checkpoints/current_run/ \
  $STORE/checkpoints/phase1/

# Archive ALP cache if you want to resume Phase 3 later
rsync -avh --progress \
  $SCRATCH/alp_cache/ \
  $STORE/alp_cache/

# Archive updated manifests
rsync -avh --progress \
  $SCRATCH/manifests/ \
  $STORE/manifests/
```

---

## Updating data_config.yaml

Once your data is staged on Scratch, point the config at your Scratch paths. The `$USER` variable will be expanded automatically by your shell when you source the config.

```yaml
# data_config.yaml

manifest:
  path: "/capstor/scratch/cscs/$USER/ultrasound/manifests/us_foundation_train.jsonl"
  val_path: "/capstor/scratch/cscs/$USER/ultrasound/manifests/us_foundation_val.jsonl"
  root_remap: {}   # Leave empty — manifests already point to Scratch paths

datasets:
  # Cardiac
  CAMUS:            "/capstor/scratch/cscs/$USER/ultrasound/raw/cardiac/CAMUS"
  EchoNet-Dynamic:  "/capstor/scratch/cscs/$USER/ultrasound/raw/cardiac/EchoNet-Dynamic"
  EchoNet-LVH:      "/capstor/scratch/cscs/$USER/ultrasound/raw/cardiac/EchoNet-LVH"
  MIMIC-IV-ECHO:    "/capstor/scratch/cscs/$USER/ultrasound/raw/cardiac/MIMIC-IV-ECHO"
  # Lung
  COVIDx-US:        "/capstor/scratch/cscs/$USER/ultrasound/raw/lung/COVIDx-US"
  # Breast
  BUS-BRA:          "/capstor/scratch/cscs/$USER/ultrasound/raw/breast/BUS-BRA"
  # Thyroid
  TN3K:             "/capstor/scratch/cscs/$USER/ultrasound/raw/thyroid/TN3K"
  TN5000:           "/capstor/scratch/cscs/$USER/ultrasound/raw/thyroid/TN5000"
  DDTI:             "/capstor/scratch/cscs/$USER/ultrasound/raw/thyroid/DDTI"
  # Fetal
  FETAL_PLANES_DB:  "/capstor/scratch/cscs/$USER/ultrasound/raw/fetal/FETAL_PLANES_DB"
  HC18:             "/capstor/scratch/cscs/$USER/ultrasound/raw/fetal/HC18"
```

If you need to rebuild manifests while data is still on Store (before staging), use `root_remap` to redirect all paths temporarily without editing the manifest files:

```yaml
manifest:
  root_remap:
    "/capstor/scratch/cscs/$USER/ultrasound": "/capstor/store/cscs/swissai/a127/ultrasound"
```

---

## Pre-Training Checklist

Run through this before every training job to catch problems before they cost you GPU hours.

```bash
STORE=/capstor/store/cscs/swissai/a127/ultrasound
SCRATCH=/capstor/scratch/cscs/$USER/ultrasound

# 1. Confirm staged datasets exist and are non-empty
du -sh $SCRATCH/raw/cardiac/CAMUS/
du -sh $SCRATCH/raw/cardiac/EchoNet-Dynamic/
# Repeat for each dataset in your run

# 2. Confirm manifests are present on Scratch
ls -lh $SCRATCH/manifests/

# 3. Spot-check that a manifest path resolves to a real file
head -1 $SCRATCH/manifests/us_foundation_train.jsonl | \
  python3 -c "import sys,json; e=json.load(sys.stdin); print(e['image_paths'][0])"
# Copy the printed path and verify it exists:
# ls <printed path>

# 4. Check your Scratch quota
lfs quota -u $USER /capstor/scratch/

# 5. Check Store usage for the project
du -sh $STORE/

# 6. Check Scratch files haven't been purged (last access time)
lfs find $SCRATCH/raw/cardiac/CAMUS/ -maxdepth 1 -atime +25
# If this returns files, they are close to the purge threshold.
# Reset access times with:
# find $SCRATCH/raw/ -type f -exec touch {} +
```

---

## Quick Reference Card

| Task | Command |
|---|---|
| Check Store usage | `du -sh /capstor/store/cscs/swissai/a127/ultrasound/` |
| Check Scratch quota | `lfs quota -u $USER /capstor/scratch/` |
| Upload from local machine | `rsync -avh --progress /local/data/ <user>@daint.cscs.ch:/capstor/store/cscs/swissai/a127/ultrasound/raw/<anatomy>/DatasetName/` |
| Stage one dataset to Scratch | `rsync -avh --progress /capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/CAMUS/ /capstor/scratch/cscs/$USER/ultrasound/raw/cardiac/CAMUS/` |
| Stage all datasets | `./stage_datasets.sh` |
| Set Lustre striping | `lfs setstripe -c 4 /capstor/scratch/cscs/$USER/ultrasound/raw/cardiac/EchoNet-Dynamic/` |
| Check striping | `lfs getstripe <path>` |
| Count files in a dataset | `find /capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/CAMUS/ -type f \| wc -l` |
| Prevent Scratch purge | `find /capstor/scratch/cscs/$USER/ultrasound/raw/ -type f -exec touch {} +` |
| Archive checkpoint to Store | `rsync -avh /capstor/scratch/cscs/$USER/ultrasound/checkpoints/current_run/ /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/phase1/` |

---

## Common Mistakes to Avoid

**Downloading directly to Scratch.** Scratch is purged after ~30 days of inactivity. If you download 10 TB to Scratch and then take a break, it will be gone. Always download to Store first.

**Forgetting to set Lustre striping before copying video data.** Striping must be set on the directory before any files are created inside it. Copying first and setting striping afterwards has no effect on existing files.

**Letting Scratch files expire mid-run.** If you stage data weeks before the training job, files may be purged by then. Run the `touch` command in the checklist above to reset access times before submitting a job.

**Training from Store paths.** If you accidentally point `data_config.yaml` at Store paths instead of Scratch paths, training will run but will be severely I/O bottlenecked. Always double-check your config before submitting a job.

**Not archiving back to Store.** At the end of each training phase, always rsync your checkpoints and manifests back to Store. Scratch is not a backup.

**Creating folders at the wrong level.** You cannot create directories at `/capstor/store/` or `/capstor/store/cscs/` — those levels are managed by CSCS. You can only create directories inside our allocated path: `/capstor/store/cscs/swissai/a127/ultrasound/`.