#!/usr/bin/env bash
# Delete legacy finetune artifact dirs and prepare the canonical results tree.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

echo "[clean] Removing legacy finetune output directories..."
rm -rf dataset_exploration_outputs/finetune
rm -rf dataset_exploration_outputs/finetune_openus_seg
rm -rf dataset_exploration_outputs/finetune_fetal_planes
rm -rf finetune/outputs
rm -rf eval_results
rm -f logs/finetune/.inner_finetune_*.sh

mkdir -p results/finetune/representative
mkdir -p results/finetune/openus_seg
mkdir -p results/finetune/fetal_planes
mkdir -p results/finetune/_dashboard

echo "[clean] Done. Logs preserved in logs/finetune/"
echo "[clean] New results root: results/finetune/"
echo "[clean] Backfill: python scripts/finetune/report.py --from-logs --all-sweeps"
