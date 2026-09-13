#!/usr/bin/env bash
# download.sh  ·  Login-node download for POCUS bench datasets (spec §2.2)
#
#   bash scripts/pocus/download.sh
#
# Reuses the a127 US-365K copy when present.  Writes $ROOT/PROVENANCE.md.
set -euo pipefail

# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/account.sh"
ROOT="${RAW_ROOT}"
DATE_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "${ROOT}"
cd "${ROOT}"

echo "POCUS raw root: ${ROOT}"

if [[ -d "${US365K_A127}/metadata" ]]; then
    echo "US-365K: reusing ${US365K_A127}"
    mkdir -p "${ROOT}/US-365K"
    ln -sfn "${US365K_A127}" "${ROOT}/US-365K/a127"
else
    huggingface-cli download JJY-0823/US-365K --repo-type dataset --local-dir "${ROOT}/US-365K"
fi

huggingface-cli download jannisborn/COVID-BLUES --repo-type dataset --local-dir "${ROOT}/COVID-BLUES"

if command -v kaggle >/dev/null 2>&1; then
    kaggle datasets download -d xiaoweixumedicalai/cardiacudc-dataset -p "${ROOT}/CardiacUDC" --unzip || \
        echo "[WARN] CardiacUDC Kaggle download failed — check licence / credentials"
    kaggle datasets download -d aspirexxx/iugc-ultrasound-video-dataset-miccai-2024 -p "${ROOT}/IUGC2024" --unzip || \
        echo "[WARN] IUGC2024 Kaggle download failed — compare file counts with Zenodo 10.5281/zenodo.17655183 (774 videos)"
else
    echo "[WARN] kaggle CLI not found; skip CardiacUDC / IUGC2024"
fi

count_files() {
    local d="$1"
    if [[ -d "$d" ]]; then
        find "$d" -type f | wc -l
    else
        echo 0
    fi
}

{
    echo "# POCUS bench provenance"
    echo
    echo "Downloaded: ${DATE_UTC}"
    echo "Root: ${ROOT}"
    echo
    echo "## US-365K"
    echo "- Source: https://huggingface.co/datasets/JJY-0823/US-365K"
    echo "- Alps copy: ${US365K_A127}"
    echo "- Licence: Academic research use per the paper; no licence on the HF card [TBC]"
    echo "- Files: $(count_files "${ROOT}/US-365K")"
    echo
    echo "## COVID-BLUES"
    echo "- Source: https://huggingface.co/datasets/jannisborn/COVID-BLUES"
    echo "- Licence: CC BY-NC-ND 4.0 — do not redistribute shards or derived data"
    echo "- Files: $(count_files "${ROOT}/COVID-BLUES")"
    echo
    echo "## CardiacUDC"
    echo "- Source: https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset"
    echo "- Licence: Apache-2.0 [TBC on the Kaggle page]"
    echo "- Native format: nii.gz (Ultatron adapter)"
    echo "- Files: $(count_files "${ROOT}/CardiacUDC")"
    echo
    echo "## IUGC 2024"
    echo "- Kaggle mirror: https://www.kaggle.com/datasets/aspirexxx/iugc-ultrasound-video-dataset-miccai-2024"
    echo "- Original: Zenodo DOI 10.5281/zenodo.17655183 (774 videos)"
    echo "- Licence: CC BY 4.0 (cite the Zenodo record)"
    echo "- Files: $(count_files "${ROOT}/IUGC2024")"
    echo "- Integrity: compare video count with 774 if the Kaggle mirror has no checksum"
} > "${ROOT}/PROVENANCE.md"

echo "Wrote ${ROOT}/PROVENANCE.md"
