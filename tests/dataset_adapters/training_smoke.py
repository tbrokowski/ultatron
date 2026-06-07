"""
tests/dataset_adapters/training_smoke.py
=========================================
Multi-dataset, multi-phase training smoke test.

Tests all four training phases (DINOv3 image SSL, V-JEPA2 video SSL,
cross-modal alignment, downstream fine-tuning) using combined data sampled
from every registered dataset adapter.

Usage (from project root with the .venv active):

    python -m tests.dataset_adapters.training_smoke

Environment overrides:
    US_SMOKE_DEVICE       Force device  (e.g. "cuda:0", "cpu")
    US_BUSI_ROOT          Override BUSI data root
    US_ECHONET_ROOT       Override EchoNet-Dynamic data root
    US_BENIN_ROOT         Override Benin-LUS data root
    US_SKIP_PHASE1=1      Skip Phase 1 image SSL smoke
    US_SKIP_PHASE2=1      Skip Phase 2 video SSL smoke
    US_SKIP_PHASE3=1      Skip Phase 3 alignment smoke
    US_SKIP_PHASE4=1      Skip Phase 4 downstream heads smoke

Each dataset builder respects a corresponding US_<NAME>_ROOT env var and
falls back to the default CSCS store path.  Datasets whose root directory
is absent on the current machine are silently skipped (SKIP in the manifest
summary, not a failure).
"""
from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.adapters.busi import BUSIAdapter
from data.adapters.tn3k import TN3KAdapter
from data.adapters.cardiac.camus import CAMUSAdapter
from data.adapters.cardiac.echonet import EchoNetDynamicAdapter
from data.adapters.cardiac.echonet_pediatric import EchoNetPediatricAdapter
from data.adapters.cardiac.echonet_lvh import EchoNetLVHAdapter
from data.adapters.cardiac.mimic_echo import MIMICEchoAdapter
from data.adapters.cardiac.mimic_lvvol_a4c import MIMICLVVolA4CAdapter
from data.adapters.cardiac.ted import TEDAdapter
from data.adapters.cardiac.unity import UnityAdapter
from data.adapters.cardiac.cardiacudc import CardiacUDCAdapter
from data.adapters.cardiac.echocp import EchoCPAdapter
from data.adapters.breast.breast_adapter import BrEaSTAdapter
from data.adapters.breast.buid_adapter import BUIDAdapter
from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
from data.adapters.breast.busv_adapter import BUSVAdapter
from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
from data.adapters.lung.benin_lus import BeninLUSAdapter
from data.adapters.lung.rsa_lus import RSALUSAdapter
from data.adapters.liver.aul import AULAdapter
from data.adapters.liver.us105 import US105Adapter
from data.adapters.maternal_fetal.acouslic import ACOUSLICAIAdapter
from data.adapters.maternal_fetal.fetal_abdominal_structures import FASSAdapter
from data.adapters.maternal_fetal.fetal_planes_db import FetalPlanesDBAdapter
from data.adapters.maternal_fetal.focus import FOCUSAdapter
from data.adapters.maternal_fetal.fpus23 import FPUS23Adapter
from data.adapters.maternal_fetal.fugc import FUGCAdapter
from data.adapters.maternal_fetal.fh_ps_aop import FHPSAOPAdapter
from data.adapters.maternal_fetal.hc18 import HC18Adapter
from data.adapters.maternal_fetal.iugc2024 import IUGC2024Adapter
from data.adapters.maternal_fetal.jnu_ifm import JNUIFMAdapter
from data.adapters.maternal_fetal.large_scale_fetal_head_biometry import LargeScaleFetalHeadBiometryAdapter
from data.adapters.maternal_fetal.maternal_fetal_us_video_intrapartum import MaternalFetalUSVideoIntrapartumAdapter
from data.adapters.maternal_fetal.pbf_us1 import PBFUS1Adapter
from data.adapters.maternal_fetal.psfhs import PSFHSAdapter
from data.adapters.cubs import CUBSAdapter
from data.adapters.common_carotid import CommonCarotidArteryImagesAdapter
from data.adapters.brain_3d_us_neuroimages import ThreeDUSNeuroimagesAdapter
from data.adapters.bite import BITEAdapter
from data.adapters.remind_brain_ius import REMINDBrainIUSAdapter
from data.adapters.resect import RESECTAdapter
from data.adapters.remind2reg import ReMIND2RegAdapter
from data.adapters.stu_hospital import STUHospitalAdapter
from data.adapters.annotated_heterogeneous_us_db import AnnotatedHeterogeneousUSDBAdapter
from data.adapters.erdes import ERDESAdapter
from data.adapters.dermatologic_skin_lesions import DermatologicSkinLesionsAdapter
from data.schema.manifest import ManifestWriter, USManifestEntry, load_manifest
from data.pipeline.dataset import ImageSSLDataset, VideoSSLDataset
from data.pipeline.downstream_dataset import DownstreamDataset, PatientLevelDataset
from data.pipeline.datamodule import USFoundationDataModule
from data.pipeline.transforms import (
    ImageSSLTransformConfig,
    VideoSSLTransformConfig,
    MASK_STRATEGY_FREQ,
)
from models.branches.image_branch import ImageBranch
from models.branches.video_branch import build_video_branch
from models.branches.shared import CrossBranchDistillation, PrototypeHead
from models.registry import build_image_backbone
from models.heads.classification_head import LinearClsHead
from models.heads.segmentation_head import LinearSegHead
from models.heads.regression_head import RegressionHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("training_smoke")

# ── Paths ─────────────────────────────────────────────────────────────────────
_STORE = Path("/capstor/store/cscs/swissai/a127/ultrasound/raw")

# Cardiac
_DEFAULT_CAMUS_ROOT          = _STORE / "cardiac" / "CAMUS"
_DEFAULT_ECHONET_ROOT        = _STORE / "cardiac" / "EchoNet-Dynamic"
_DEFAULT_ECHONET_PED_ROOT    = _STORE / "cardiac" / "EchoNet-Pediatric"
_DEFAULT_ECHONET_LVH_ROOT    = _STORE / "cardiac" / "EchoNet-LVH"
_DEFAULT_MIMIC_ECHO_ROOT     = _STORE / "cardiac" / "MIMIC-IV-Echo"
_DEFAULT_MIMIC_LVVOL_ROOT    = _STORE / "cardiac" / "MIMIC-IV-Echo-LVVol-A4C"
_DEFAULT_TED_ROOT            = _STORE / "cardiac" / "TED"
_DEFAULT_UNITY_ROOT          = _STORE / "cardiac" / "Unity"
_DEFAULT_CARDIACUDC_ROOT     = _STORE / "cardiac" / "CardiacUDC"
_DEFAULT_ECHOCP_ROOT         = _STORE / "cardiac" / "EchoCP"

# Breast / Thyroid
_DEFAULT_BUSI_ROOT           = _STORE / "breast"  / "BUSI"
_DEFAULT_BREASST_ROOT        = _STORE / "breast"  / "BrEaST"
_DEFAULT_BUID_ROOT           = _STORE / "breast"  / "BUID"
_DEFAULT_BUSBRA_ROOT         = _STORE / "breast"  / "BUSBRA"
_DEFAULT_BUS_UC_ROOT         = _STORE / "breast"  / "BUS_UC"
_DEFAULT_BUS_UCLM_ROOT       = _STORE / "breast"  / "BUS-UCLM" / "BUS-UCLM"
_DEFAULT_BUSV_ROOT           = _STORE / "breast"  / "Miccai 2022 BUV Dataset"
_DEFAULT_GDPH_ROOT           = _STORE / "breast"  / "GDPH&SYSUCC"
_DEFAULT_CNRPT_ROOT          = _STORE / "breast"  / "Chinese US-Report Dataset (Breast)"
_DEFAULT_TN3K_ROOT           = _STORE / "thyroid" / "TN3K"

# Lung
_DEFAULT_BENIN_ROOT          = _STORE / "lung"    / "Benin_Videos"
_DEFAULT_RSA_ROOT            = _STORE / "lung"    / "RSA_Videos"

# Liver
_DEFAULT_AUL_ROOT            = _STORE / "liver"   / "AUL"
_DEFAULT_105US_ROOT          = _STORE / "liver"   / "105US"

# Fetal
_DEFAULT_ACOUSLIC_ROOT       = _STORE / "fetal"   / "ACOUSLIC"
_DEFAULT_FASS_ROOT           = _STORE / "fetal"   / "fetal-abdominal-structures-segmentation"
_DEFAULT_FETAL_PLANES_ROOT   = _STORE / "fetal"   / "FETAL-PLANES-DB"
_DEFAULT_FOCUS_ROOT          = _STORE / "fetal"   / "FOCUS"
_DEFAULT_FPUS23_ROOT         = _STORE / "fetal"   / "FPUS23"
_DEFAULT_FUGC_ROOT           = _STORE / "fetal"   / "FUGC"
_DEFAULT_FH_PS_AOP_ROOT      = _STORE / "fetal"   / "FH-PS-AOP"
_DEFAULT_HC18_ROOT           = _STORE / "fetal"   / "HC18"
_DEFAULT_IUGC2024_ROOT       = _STORE / "fetal"   / "IUGC-2024"
_DEFAULT_JNU_IFM_ROOT        = _STORE / "fetal"   / "JNU-IFM"
_DEFAULT_LSFHB_ROOT          = _STORE / "fetal"   / "large-scale-fetal-head-biometry"
_DEFAULT_MF_INTRAPARTUM_ROOT = _STORE / "fetal"   / "maternal-fetal-us-video-intrapartum"
_DEFAULT_PBF_US1_ROOT        = _STORE / "fetal"   / "PBF-US1"
_DEFAULT_PSFHS_ROOT          = _STORE / "fetal"   / "PSFHS"

# Vascular / Carotid
_DEFAULT_CUBS_ROOT           = _STORE / "vascular-carotid" / "CUBS"
_DEFAULT_CAROTID_ROOT        = _STORE / "vascular-carotid" / "Common-Carotid-Artery-Ultrasound-Images"

# Brain
_DEFAULT_3D_NEURO_ROOT       = _STORE / "brain"   / "3D-US-Neuroimages-Dataset"
_DEFAULT_BITE_ROOT           = _STORE / "brain"   / "BITE"
_DEFAULT_REMIND_ROOT         = _STORE / "brain"   / "REMIND-Brain-iUS"
_DEFAULT_RESECT_ROOT         = _STORE / "brain"   / "RESECT"
_DEFAULT_REMIND2REG_ROOT     = _STORE / "brain"   / "ReMIND2Reg"

# Multi-organ / Ocular / Skin
_DEFAULT_STU_ROOT            = _STORE / "multi_organ" / "STU-Hospital-master"
_DEFAULT_AHUS_ROOT           = _STORE / "multi_organ" / "annotated_heterogeneous_us_db"
_DEFAULT_ERDES_ROOT          = _STORE / "ocular"  / "ERDES"
_DEFAULT_DERM_ROOT           = _STORE / "skin"    / "Dermatologic-US-Skin-Lesions"

_SMOKE_OUT  = _ROOT / "dataset_exploration_outputs" / "smoke"
_SMOKE_CFG  = _ROOT / "configs" / "smoke" / "multi_dataset_smoke.yaml"
_COMBINED_MANIFEST = _SMOKE_OUT / "combined_smoke_manifest.jsonl"

N_SMOKE_ENTRIES = 32   # entries per dataset in the combined manifest
N_SMOKE_BATCHES = 2    # forward passes per phase


# ── Device ───────────────────────────────────────────────────────────────────

def _auto_device() -> str:
    """Auto-select: respect US_SMOKE_DEVICE, otherwise prefer CUDA."""
    env = os.environ.get("US_SMOKE_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        dev = "cuda:0"
        log.info("CUDA available — using %s (%s)",
                 dev, torch.cuda.get_device_name(0))
        return dev
    log.warning("CUDA not available — running on CPU (will be slow)")
    return "cpu"


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _root(env_var: str, default: Path) -> Optional[Path]:
    env = os.environ.get(env_var)
    p = Path(env) if env else default
    return p if p.exists() else None


def _build_camus_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_CAMUS_ROOT", _DEFAULT_CAMUS_ROOT)
    if root is None:
        log.warning("CAMUS root not found — skipping")
        return []
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        log.warning("SimpleITK not installed — skipping CAMUS")
        return []
    entries: List[USManifestEntry] = []
    for e in CAMUSAdapter(root).iter_entries():
        # Prefer image entries for Phase 1 image SSL coverage
        if e.modality_type in ("image", "pseudo_video"):
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("CAMUS: %d entries", len(entries))
    return entries


def _build_busi_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BUSI_ROOT", _DEFAULT_BUSI_ROOT)
    if root is None:
        log.warning("BUSI root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BUSIAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BUSI: %d entries", len(entries))
    return entries


def _build_echonet_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_ECHONET_ROOT", _DEFAULT_ECHONET_ROOT)
    if root is None:
        log.warning("EchoNet root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in EchoNetDynamicAdapter(root).iter_entries():
        if e.split == "train":
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("EchoNet: %d entries", len(entries))
    return entries


def _build_benin_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BENIN_ROOT", _DEFAULT_BENIN_ROOT)
    if root is None:
        log.warning("Benin-LUS root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BeninLUSAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("Benin-LUS: %d entries", len(entries))
    return entries


def _build_echonet_ped_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_ECHONET_PED_ROOT", _DEFAULT_ECHONET_PED_ROOT)
    if root is None:
        log.warning("EchoNet-Pediatric root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in EchoNetPediatricAdapter(root).iter_entries():
        if e.split == "train":
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("EchoNet-Pediatric: %d entries", len(entries))
    return entries


def _build_ted_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_TED_ROOT", _DEFAULT_TED_ROOT)
    if root is None:
        log.warning("TED root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    # Only take 'video' modality entries for the smoke manifest (ED/ES images
    # are a by-product of the same file; video entries are sufficient here).
    for e in TEDAdapter(root).iter_entries():
        if e.modality_type == "video":
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("TED: %d entries", len(entries))
    return entries


# ── Cardiac (new) ─────────────────────────────────────────────────────────────

def _build_echonet_lvh_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_ECHONET_LVH_ROOT", _DEFAULT_ECHONET_LVH_ROOT)
    if root is None:
        log.warning("EchoNet-LVH root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in EchoNetLVHAdapter(root).iter_entries():
        if e.split == "train":
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("EchoNet-LVH: %d entries", len(entries))
    return entries


def _build_mimic_echo_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_MIMIC_ECHO_ROOT", _DEFAULT_MIMIC_ECHO_ROOT)
    if root is None:
        log.warning("MIMIC-IV-Echo root not found — skipping")
        return []
    try:
        import pydicom  # noqa: F401
    except ImportError:
        log.warning("pydicom not installed — skipping MIMIC-IV-Echo")
        return []
    entries: List[USManifestEntry] = []
    for e in MIMICEchoAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("MIMIC-IV-Echo: %d entries", len(entries))
    return entries


def _build_mimic_lvvol_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_MIMIC_LVVOL_ROOT", _DEFAULT_MIMIC_LVVOL_ROOT)
    if root is None:
        log.warning("MIMIC-IV-Echo-LVVol-A4C root not found — skipping")
        return []
    try:
        import pydicom  # noqa: F401
    except ImportError:
        log.warning("pydicom not installed — skipping MIMIC-IV-Echo-LVVol-A4C")
        return []
    entries: List[USManifestEntry] = []
    for e in MIMICLVVolA4CAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("MIMIC-IV-Echo-LVVol-A4C: %d entries", len(entries))
    return entries


def _build_unity_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_UNITY_ROOT", _DEFAULT_UNITY_ROOT)
    if root is None:
        log.warning("Unity-Echo root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in UnityAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("Unity-Echo: %d entries", len(entries))
    return entries


def _build_cardiacudc_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_CARDIACUDC_ROOT", _DEFAULT_CARDIACUDC_ROOT)
    if root is None:
        log.warning("CardiacUDC root not found — skipping")
        return []
    try:
        import nibabel  # noqa: F401
    except ImportError:
        log.warning("nibabel not installed — skipping CardiacUDC")
        return []
    entries: List[USManifestEntry] = []
    for e in CardiacUDCAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("CardiacUDC: %d entries", len(entries))
    return entries


def _build_echocp_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_ECHOCP_ROOT", _DEFAULT_ECHOCP_ROOT)
    if root is None:
        log.warning("EchoCP root not found — skipping")
        return []
    try:
        import nibabel  # noqa: F401
    except ImportError:
        log.warning("nibabel not installed — skipping EchoCP")
        return []
    entries: List[USManifestEntry] = []
    for e in EchoCPAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("EchoCP: %d entries", len(entries))
    return entries


# ── Breast / Thyroid (new) ────────────────────────────────────────────────────

def _build_breasst_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BREASST_ROOT", _DEFAULT_BREASST_ROOT)
    if root is None:
        log.warning("BrEaST root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BrEaSTAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BrEaST: %d entries", len(entries))
    return entries


def _build_buid_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BUID_ROOT", _DEFAULT_BUID_ROOT)
    if root is None:
        log.warning("BUID root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BUIDAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BUID: %d entries", len(entries))
    return entries


def _build_busbra_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BUSBRA_ROOT", _DEFAULT_BUSBRA_ROOT)
    if root is None:
        log.warning("BUS-BRA root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BUSBRAAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BUS-BRA: %d entries", len(entries))
    return entries


def _build_bus_uc_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BUS_UC_ROOT", _DEFAULT_BUS_UC_ROOT)
    if root is None:
        log.warning("BUS-UC root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BUSUCAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BUS-UC: %d entries", len(entries))
    return entries


def _build_bus_uclm_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BUS_UCLM_ROOT", _DEFAULT_BUS_UCLM_ROOT)
    if root is None:
        log.warning("BUS-UCLM root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BUSUCLMAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BUS-UCLM: %d entries", len(entries))
    return entries


def _build_busv_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BUSV_ROOT", _DEFAULT_BUSV_ROOT)
    if root is None:
        log.warning("BUSV root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BUSVAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BUSV: %d entries", len(entries))
    return entries


def _build_gdph_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_GDPH_ROOT", _DEFAULT_GDPH_ROOT)
    if root is None:
        log.warning("GDPH-SYSUCC root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in GDPHSYSUCCAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("GDPH-SYSUCC: %d entries", len(entries))
    return entries


def _build_cnrpt_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_CNRPT_ROOT", _DEFAULT_CNRPT_ROOT)
    if root is None:
        log.warning("Chinese-US-Report-Breast root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in ChineseUSReportBreastAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("Chinese-US-Report-Breast: %d entries", len(entries))
    return entries


def _build_tn3k_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_TN3K_ROOT", _DEFAULT_TN3K_ROOT)
    if root is None:
        log.warning("TN3K root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in TN3KAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("TN3K: %d entries", len(entries))
    return entries


# ── Lung (new) ────────────────────────────────────────────────────────────────

def _build_rsa_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_RSA_ROOT", _DEFAULT_RSA_ROOT)
    if root is None:
        log.warning("RSA-LUS root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in RSALUSAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("RSA-LUS: %d entries", len(entries))
    return entries


# ── Liver (new) ───────────────────────────────────────────────────────────────

def _build_aul_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_AUL_ROOT", _DEFAULT_AUL_ROOT)
    if root is None:
        log.warning("AUL root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in AULAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("AUL: %d entries", len(entries))
    return entries


def _build_105us_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_105US_ROOT", _DEFAULT_105US_ROOT)
    if root is None:
        log.warning("105US root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in US105Adapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("105US: %d entries", len(entries))
    return entries


# ── Fetal (new) ───────────────────────────────────────────────────────────────

def _build_acouslic_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_ACOUSLIC_ROOT", _DEFAULT_ACOUSLIC_ROOT)
    if root is None:
        log.warning("ACOUSLIC-AI root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in ACOUSLICAIAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("ACOUSLIC-AI: %d entries", len(entries))
    return entries


def _build_fass_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_FASS_ROOT", _DEFAULT_FASS_ROOT)
    if root is None:
        log.warning("FASS root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in FASSAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("FASS: %d entries", len(entries))
    return entries


def _build_fetal_planes_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_FETAL_PLANES_ROOT", _DEFAULT_FETAL_PLANES_ROOT)
    if root is None:
        log.warning("FETAL-PLANES-DB root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in FetalPlanesDBAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("FETAL-PLANES-DB: %d entries", len(entries))
    return entries


def _build_focus_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_FOCUS_ROOT", _DEFAULT_FOCUS_ROOT)
    if root is None:
        log.warning("FOCUS root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in FOCUSAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("FOCUS: %d entries", len(entries))
    return entries


def _build_fpus23_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_FPUS23_ROOT", _DEFAULT_FPUS23_ROOT)
    if root is None:
        log.warning("FPUS23 root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in FPUS23Adapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("FPUS23: %d entries", len(entries))
    return entries


def _build_fugc_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_FUGC_ROOT", _DEFAULT_FUGC_ROOT)
    if root is None:
        log.warning("FUGC root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in FUGCAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("FUGC: %d entries", len(entries))
    return entries


def _build_fh_ps_aop_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_FH_PS_AOP_ROOT", _DEFAULT_FH_PS_AOP_ROOT)
    if root is None:
        log.warning("FH-PS-AOP root not found — skipping")
        return []
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        log.warning("SimpleITK not installed — skipping FH-PS-AOP")
        return []
    entries: List[USManifestEntry] = []
    for e in FHPSAOPAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("FH-PS-AOP: %d entries", len(entries))
    return entries


def _build_hc18_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_HC18_ROOT", _DEFAULT_HC18_ROOT)
    if root is None:
        log.warning("HC18 root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in HC18Adapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("HC18: %d entries", len(entries))
    return entries


def _build_iugc2024_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_IUGC2024_ROOT", _DEFAULT_IUGC2024_ROOT)
    if root is None:
        log.warning("IUGC2024 root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in IUGC2024Adapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("IUGC2024: %d entries", len(entries))
    return entries


def _build_jnu_ifm_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_JNU_IFM_ROOT", _DEFAULT_JNU_IFM_ROOT)
    if root is None:
        log.warning("JNU-IFM root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in JNUIFMAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("JNU-IFM: %d entries", len(entries))
    return entries


def _build_lsfhb_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_LSFHB_ROOT", _DEFAULT_LSFHB_ROOT)
    if root is None:
        log.warning("Large-Scale-Fetal-Head-Biometry root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in LargeScaleFetalHeadBiometryAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("Large-Scale-Fetal-Head-Biometry: %d entries", len(entries))
    return entries


def _build_mf_intrapartum_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_MF_INTRAPARTUM_ROOT", _DEFAULT_MF_INTRAPARTUM_ROOT)
    if root is None:
        log.warning("maternal-fetal-us-video-intrapartum root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in MaternalFetalUSVideoIntrapartumAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("maternal-fetal-us-video-intrapartum: %d entries", len(entries))
    return entries


def _build_pbf_us1_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_PBF_US1_ROOT", _DEFAULT_PBF_US1_ROOT)
    if root is None:
        log.warning("PBF-US1 root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in PBFUS1Adapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("PBF-US1: %d entries", len(entries))
    return entries


def _build_psfhs_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_PSFHS_ROOT", _DEFAULT_PSFHS_ROOT)
    if root is None:
        log.warning("PSFHS root not found — skipping")
        return []
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        log.warning("SimpleITK not installed — skipping PSFHS")
        return []
    entries: List[USManifestEntry] = []
    for e in PSFHSAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("PSFHS: %d entries", len(entries))
    return entries


# ── Vascular / Carotid (new) ──────────────────────────────────────────────────

def _build_cubs_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_CUBS_ROOT", _DEFAULT_CUBS_ROOT)
    if root is None:
        log.warning("CUBS root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in CUBSAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("CUBS: %d entries", len(entries))
    return entries


def _build_carotid_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_CAROTID_ROOT", _DEFAULT_CAROTID_ROOT)
    if root is None:
        log.warning("Common-Carotid root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in CommonCarotidArteryImagesAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("Common-Carotid: %d entries", len(entries))
    return entries


# ── Brain / Multi-organ / Ocular / Skin (new) ─────────────────────────────────

def _build_3d_neuro_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_3D_NEURO_ROOT", _DEFAULT_3D_NEURO_ROOT)
    if root is None:
        log.warning("3D-US-Neuroimages root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in ThreeDUSNeuroimagesAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("3D-US-Neuroimages: %d entries", len(entries))
    return entries


def _build_bite_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BITE_ROOT", _DEFAULT_BITE_ROOT)
    if root is None:
        log.warning("BITE root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BITEAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BITE: %d entries", len(entries))
    return entries


def _build_remind_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_REMIND_ROOT", _DEFAULT_REMIND_ROOT)
    if root is None:
        log.warning("REMIND-Brain-iUS root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in REMINDBrainIUSAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("REMIND-Brain-iUS: %d entries", len(entries))
    return entries


def _build_resect_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_RESECT_ROOT", _DEFAULT_RESECT_ROOT)
    if root is None:
        log.warning("RESECT root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in RESECTAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("RESECT: %d entries", len(entries))
    return entries


def _build_remind2reg_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_REMIND2REG_ROOT", _DEFAULT_REMIND2REG_ROOT)
    if root is None:
        log.warning("ReMIND2Reg root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in ReMIND2RegAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("ReMIND2Reg: %d entries", len(entries))
    return entries


def _build_stu_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_STU_ROOT", _DEFAULT_STU_ROOT)
    if root is None:
        log.warning("STU-Hospital-master root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in STUHospitalAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("STU-Hospital-master: %d entries", len(entries))
    return entries


def _build_ahus_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_AHUS_ROOT", _DEFAULT_AHUS_ROOT)
    if root is None:
        log.warning("annotated_heterogeneous_us_db root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in AnnotatedHeterogeneousUSDBAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("annotated_heterogeneous_us_db: %d entries", len(entries))
    return entries


def _build_erdes_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_ERDES_ROOT", _DEFAULT_ERDES_ROOT)
    if root is None:
        log.warning("ERDES root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in ERDESAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("ERDES: %d entries", len(entries))
    return entries


def _build_derm_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_DERM_ROOT", _DEFAULT_DERM_ROOT)
    if root is None:
        log.warning("Dermatologic-US-Skin-Lesions root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in DermatologicSkinLesionsAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("Dermatologic-US-Skin-Lesions: %d entries", len(entries))
    return entries


def build_combined_manifest(force: bool = False) -> Path:
    """Build (or reuse) the combined smoke manifest."""
    _SMOKE_OUT.mkdir(parents=True, exist_ok=True)

    if _COMBINED_MANIFEST.exists() and not force:
        log.info("Reusing existing manifest: %s", _COMBINED_MANIFEST)
        return _COMBINED_MANIFEST

    _BUILDERS = [
        # Cardiac
        _build_camus_entries,
        _build_echonet_entries,
        _build_echonet_ped_entries,
        _build_echonet_lvh_entries,
        _build_mimic_echo_entries,
        _build_mimic_lvvol_entries,
        _build_ted_entries,
        _build_unity_entries,
        _build_cardiacudc_entries,
        _build_echocp_entries,
        # Breast / Thyroid
        _build_busi_entries,
        _build_breasst_entries,
        _build_buid_entries,
        _build_busbra_entries,
        _build_bus_uc_entries,
        _build_bus_uclm_entries,
        _build_busv_entries,
        _build_gdph_entries,
        _build_cnrpt_entries,
        _build_tn3k_entries,
        # Lung
        _build_benin_entries,
        _build_rsa_entries,
        # Liver
        _build_aul_entries,
        _build_105us_entries,
        # Fetal
        _build_acouslic_entries,
        _build_fass_entries,
        _build_fetal_planes_entries,
        _build_focus_entries,
        _build_fpus23_entries,
        _build_fugc_entries,
        _build_fh_ps_aop_entries,
        _build_hc18_entries,
        _build_iugc2024_entries,
        _build_jnu_ifm_entries,
        _build_lsfhb_entries,
        _build_mf_intrapartum_entries,
        _build_pbf_us1_entries,
        _build_psfhs_entries,
        # Vascular / Carotid
        _build_cubs_entries,
        _build_carotid_entries,
        # Brain
        _build_3d_neuro_entries,
        _build_bite_entries,
        _build_remind_entries,
        _build_resect_entries,
        _build_remind2reg_entries,
        # Multi-organ / Ocular / Skin
        _build_stu_entries,
        _build_ahus_entries,
        _build_erdes_entries,
        _build_derm_entries,
    ]

    all_entries: List[USManifestEntry] = []
    for builder in _BUILDERS:
        try:
            all_entries.extend(builder())
        except Exception as exc:
            log.warning("Builder %s failed — skipping: %s", builder.__name__, exc)

    if not all_entries:
        raise RuntimeError("No entries found — check dataset paths.")

    with ManifestWriter(_COMBINED_MANIFEST) as w:
        for e in all_entries:
            w.write(e)

    log.info("Combined manifest written: %d entries → %s",
             len(all_entries), _COMBINED_MANIFEST)
    return _COMBINED_MANIFEST


# ── Config / DataModule helpers ───────────────────────────────────────────────

def load_smoke_config() -> dict:
    with open(_SMOKE_CFG) as f:
        return yaml.safe_load(f)


def build_datamodule(cfg: dict) -> USFoundationDataModule:
    img_cfg = ImageSSLTransformConfig(
        n_global_crops=cfg["transforms"]["image"]["n_global_crops"],
        n_local_crops=cfg["transforms"]["image"]["n_local_crops"],
        max_global_crop_px=cfg["transforms"]["image"]["max_global_crop_px"],
        min_crop_px=cfg["transforms"]["image"]["min_crop_px"],
        mask_strategy=MASK_STRATEGY_FREQ,
    )
    vid_cfg = VideoSSLTransformConfig(
        n_frames=cfg["transforms"]["video"]["n_frames"],
        max_crop_px=cfg["transforms"]["video"]["max_crop_px"],
        min_crop_px=cfg["transforms"]["video"]["min_crop_px"],
    )
    dm = USFoundationDataModule(
        manifest_path=str(_COMBINED_MANIFEST),
        image_batch_size=cfg["loaders"]["image_batch_size"],
        video_batch_size=cfg["loaders"]["video_batch_size"],
        num_workers=cfg["loaders"]["num_workers"],
        pin_memory=cfg["loaders"].get("pin_memory", False),
        image_cfg=img_cfg,
        video_cfg=vid_cfg,
        total_training_steps=cfg["curriculum"]["total_training_steps"],
        image_samples_per_epoch=cfg["curriculum"]["image_samples_per_epoch"],
        video_samples_per_epoch=cfg["curriculum"]["video_samples_per_epoch"],
    )
    dm.setup()
    log.info(
        "DataModule ready — image entries: %d | video entries: %d",
        len(dm._image_entries), len(dm._video_entries),
    )
    return dm


def _to_dev(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


# ── Native-resolution collate for DownstreamDataset ──────────────────────────

def _downstream_collate(samples: list) -> dict:
    """
    Collate DownstreamDataset samples at native resolution.

    Images in a batch will generally have different sizes — that is by design
    (the system is resolution-agnostic).  We pad each image to the batch-max
    (H, W) with zeros and produce a boolean padding_mask (B, ph, pw) where
    True = valid patch.  The DINOv3 backbone then ignores padding tokens via
    the attention bias we just fixed.
    """
    patch_size = 16

    # Determine batch-max spatial dims
    max_h = max(s["image"].shape[-2] for s in samples)
    max_w = max(s["image"].shape[-1] for s in samples)

    # Round up to patch-grid multiples so ph/pw are integers
    max_h = ((max_h + patch_size - 1) // patch_size) * patch_size
    max_w = ((max_w + patch_size - 1) // patch_size) * patch_size

    ph = max_h // patch_size
    pw = max_w // patch_size
    B  = len(samples)
    C  = samples[0]["image"].shape[0]

    images       = torch.zeros(B, C, max_h, max_w)
    padding_mask = torch.zeros(B, ph, pw, dtype=torch.bool)

    for i, s in enumerate(samples):
        _, h, w = s["image"].shape
        images[i, :, :h, :w] = s["image"]
        # Mark patches that are fully covered by the actual image as valid
        vh = h // patch_size
        vw = w // patch_size
        padding_mask[i, :vh, :vw] = True

    # Collate all other fields
    out: dict = {"image": images, "padding_mask": padding_mask}
    for key in samples[0]:
        if key == "image":
            continue
        vals = [s[key] for s in samples]
        v0 = vals[0]
        try:
            if isinstance(v0, torch.Tensor):
                out[key] = torch.stack(vals)
            elif isinstance(v0, (int, float)):
                out[key] = torch.tensor(vals)
            elif isinstance(v0, bool):
                out[key] = torch.tensor(vals, dtype=torch.bool)
            else:
                out[key] = vals       # lists, strings, dicts, LabelTargets etc.
        except Exception:
            out[key] = vals
    return out


# ── Phase 1: Image SSL ────────────────────────────────────────────────────────

def phase1_smoke(dm: USFoundationDataModule, device: str) -> None:
    log.info("=== Phase 1: Image SSL (DINOv3-S) ===")
    dtype = torch.float32

    student = build_image_backbone("dinov3_s", dtype=dtype)
    teacher = build_image_backbone("dinov3_s", dtype=dtype)
    branch = ImageBranch(student=student, teacher=teacher).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(branch.student.parameters(), lr=1e-4)

    loader = dm.image_loader()
    branch.train()
    n = 0
    for batch in loader:
        batch = _to_dev(batch, device)
        global_crops = batch["global_crops"].to(dtype)   # (B, 2, C, H, W)
        local_crops  = batch.get("local_crops")
        patch_mask   = batch.get("patch_mask")

        opt.zero_grad()

        # Teacher on clean crop (no padding mask needed — uniform squares)
        t_out = branch.forward_teacher(global_crops[:, 1])
        # Student on masked crop
        s_out = branch.forward_student(global_crops[:, 0])

        loss = cosine_loss(s_out["cls"], t_out["cls"])

        # Patch-level loss (if patch tokens available)
        if "patch_tokens" in s_out and "patch_tokens" in t_out:
            loss = loss + 0.5 * cosine_loss(
                s_out["patch_tokens"].mean(1),
                t_out["patch_tokens"].mean(1),
            )

        # Local crops
        if local_crops is not None:
            local_crops = local_crops.to(dtype)
            for i in range(local_crops.shape[1]):
                s_loc = branch.forward_student(local_crops[:, i])
                loss = loss + 0.3 * cosine_loss(s_loc["cls"], t_out["cls"])

        loss.backward()
        nn.utils.clip_grad_norm_(branch.student.parameters(), 1.0)
        opt.step()
        branch.update_teacher(momentum=0.9995)

        log.info("  Phase1 batch=%d  loss=%.4f  cls.shape=%s",
                 n + 1, loss.item(), tuple(s_out["cls"].shape))
        assert torch.isfinite(loss), f"Non-finite loss at batch {n+1}"
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    assert n > 0, "Phase 1: no image batches yielded — check manifest/stream split"
    log.info("Phase 1 PASS (%d batches)", n)


# ── Phase 2: Video SSL ────────────────────────────────────────────────────────

def phase2_smoke(dm: USFoundationDataModule, device: str) -> None:
    log.info("=== Phase 2: Video SSL (V-JEPA2) ===")
    if not dm._video_entries:
        log.warning("Phase 2 SKIP — no video entries in manifest")
        return

    dtype = torch.float32
    branch = build_video_branch(dtype=dtype, device=device)
    opt = torch.optim.AdamW(branch.student.parameters(), lr=1e-4)

    loader = dm.video_loader()
    branch.train()
    n = 0
    for batch in loader:
        batch = _to_dev(batch, device)
        full_clip  = batch["full_clips"].to(dtype)          # (B, T, C, H, W)
        vis_clip   = batch["visible_clips"].to(dtype)
        tube_mask  = batch.get("tube_masks")
        pad_mask   = batch.get("padding_masks")
        valid_fr   = batch.get("valid_frames")

        opt.zero_grad()
        t_out = branch.forward_teacher(full_clip, padding_mask=pad_mask,
                                       valid_frames=valid_fr)
        s_out = branch.forward_student(vis_clip, tube_mask=tube_mask,
                                       padding_mask=pad_mask,
                                       valid_frames=valid_fr)

        loss = cosine_loss(s_out["clip_cls"], t_out["clip_cls"])
        loss.backward()
        nn.utils.clip_grad_norm_(branch.student.parameters(), 1.0)
        opt.step()
        branch.update_teacher(momentum=0.9995)

        log.info("  Phase2 batch=%d  loss=%.4f  clip_cls.shape=%s",
                 n + 1, loss.item(), tuple(s_out["clip_cls"].shape))
        assert torch.isfinite(loss), f"Non-finite loss at batch {n+1}"
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    if n == 0:
        log.warning("Phase 2 SKIP — video loader yielded no batches")
        return
    log.info("Phase 2 PASS (%d batches)", n)


# ── Phase 3: Cross-modal alignment ───────────────────────────────────────────

def phase3_smoke(dm: USFoundationDataModule, device: str) -> None:
    log.info("=== Phase 3: Cross-modal Alignment ===")
    if not dm._video_entries:
        log.warning("Phase 3 SKIP — no video entries in manifest")
        return

    dtype = torch.float32

    # Image branch
    img_student = build_image_backbone("dinov3_s", dtype=dtype)
    img_teacher = build_image_backbone("dinov3_s", dtype=dtype)
    img_branch = ImageBranch(img_student, img_teacher).to(device=device, dtype=dtype)

    # Video branch
    vid_branch = build_video_branch(dtype=dtype, device=device)

    D_img = img_branch.embed_dim              # 384 for dinov3_s
    D_vid = vid_branch.student.hidden_size    # 1024 for vjepa2_l
    align_dim = 256

    cross = CrossBranchDistillation(img_dim=D_img, vid_dim=D_vid,
                                    align_dim=align_dim).to(device=device, dtype=dtype)
    # PrototypeHead works in a single shared space.
    # Video tokens (D_vid) are projected to D_img before assignment.
    proto     = PrototypeHead(embed_dim=D_img, n_prototypes=64).to(device=device, dtype=dtype)
    vid_to_img = nn.Linear(D_vid, D_img, bias=False).to(device=device, dtype=dtype)

    params = (
        list(img_branch.student.parameters())
        + list(vid_branch.student.parameters())
        + list(cross.parameters())
        + list(proto.parameters())
        + list(vid_to_img.parameters())
    )
    opt = torch.optim.AdamW(params, lr=1e-4)

    img_branch.train()
    vid_branch.train()
    cross.train()
    proto.train()
    vid_to_img.train()

    n = 0
    for dual in dm.combined_loader():
        img_batch = _to_dev(dual.image_batch, device)
        vid_batch = _to_dev(dual.video_batch, device)

        global_crops = img_batch["global_crops"].to(dtype)   # (B, 2, C, H, W)
        full_clip    = vid_batch["full_clips"].to(dtype)      # (B, T, C, H, W)
        vis_clip     = vid_batch["visible_clips"].to(dtype)
        tube_mask    = vid_batch.get("tube_masks")
        pad_mask_vid = vid_batch.get("padding_masks")

        opt.zero_grad()

        # Image arm
        t_img = img_branch.forward_teacher(global_crops[:, 1])
        s_img = img_branch.forward_student(global_crops[:, 0])
        loss_img = cosine_loss(s_img["cls"], t_img["cls"])

        # Video arm
        t_vid = vid_branch.forward_teacher(full_clip, padding_mask=pad_mask_vid)
        s_vid = vid_branch.forward_student(vis_clip, tube_mask=tube_mask,
                                           padding_mask=pad_mask_vid)
        loss_vid = cosine_loss(s_vid["clip_cls"], t_vid["clip_cls"])

        # Cross-branch distillation
        img_patches = t_img["patch_tokens"]                   # (B, N, D_img)
        vid_tubes   = s_vid.get("tube_tokens",
                       s_vid["clip_cls"].unsqueeze(1))        # (B, M, D_vid)
        loss_cross  = cross(img_patches, vid_tubes)

        # Prototype consistency: project video to img dim before assignment
        vid_tubes_proj = vid_to_img(vid_tubes)                # (B, M, D_img)
        loss_proto = proto.consistency_loss(img_patches, vid_tubes_proj)

        loss = loss_img + loss_vid + loss_cross + 0.5 * loss_proto
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        img_branch.update_teacher()
        vid_branch.update_teacher()

        log.info(
            "  Phase3 batch=%d  loss=%.4f  "
            "(img=%.3f vid=%.3f cross=%.3f proto=%.3f)",
            n + 1, loss.item(), loss_img.item(),
            loss_vid.item(), loss_cross.item(), loss_proto.item(),
        )
        assert torch.isfinite(loss), f"Non-finite loss at batch {n+1}"
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    if n == 0:
        log.warning("Phase 3 SKIP — combined loader yielded no batches")
        return
    log.info("Phase 3 PASS (%d batches)", n)


# ── Phase 4: Downstream heads ─────────────────────────────────────────────────

def _build_backbone_frozen(device: str, dtype: torch.dtype) -> nn.Module:
    bb = build_image_backbone("dinov3_s", dtype=dtype).to(device=device, dtype=dtype)
    for p in bb.parameters():
        p.requires_grad_(False)
    bb.eval()
    return bb


def _phase4_classification_smoke(
    busi_entries: List[USManifestEntry], device: str
) -> None:
    """Binary malignancy classification on BUSI."""
    entries = [e for e in busi_entries if e.task_type != "ssl_only"][:16]
    if not entries:
        log.warning("Phase4/cls SKIP — no BUSI supervised entries")
        return

    dtype = torch.float32
    bb = _build_backbone_frozen(device, dtype)
    D  = bb.hidden_size
    head = LinearClsHead(embed_dim=D, n_classes=1).to(device=device, dtype=dtype)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-3)

    ds = DownstreamDataset(entries, active_head_ids=["breast_malignancy_cls"])
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=_downstream_collate)

    head.train()
    n = 0
    for batch in loader:
        imgs     = batch["image"].to(device=device, dtype=dtype)
        pad_mask = batch.get("padding_mask")
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=device)

        opt.zero_grad()
        with torch.no_grad():
            feats = bb(imgs, padding_mask=pad_mask)
        logits = head(feats["cls"])                             # (B, 1)
        cls_label = batch.get("cls_label")
        if cls_label is None or (isinstance(cls_label, torch.Tensor) and (cls_label < 0).all()):
            loss = logits.mean() * 0.0
        else:
            lbl = (cls_label if isinstance(cls_label, torch.Tensor)
                   else torch.tensor(cls_label)).to(device=device, dtype=dtype)
            lbl = lbl.float().unsqueeze(1).clamp(0, 1)
            loss = F.binary_cross_entropy_with_logits(logits, lbl)
        loss.backward()
        opt.step()

        log.info("  Phase4/cls batch=%d  loss=%.4f  logits.shape=%s",
                 n + 1, loss.item(), tuple(logits.shape))
        assert torch.isfinite(loss)
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    log.info("Phase4/classification PASS (%d batches)", n)


def _phase4_segmentation_smoke(
    busi_entries: List[USManifestEntry], device: str
) -> None:
    """Lesion segmentation on BUSI."""
    entries = [e for e in busi_entries
               if e.task_type in ("seg", "seg_cls") and e.seg_mask_paths][:8]
    if not entries:
        log.warning("Phase4/seg SKIP — no BUSI segmentation entries")
        return

    dtype = torch.float32
    bb   = _build_backbone_frozen(device, dtype)
    D    = bb.hidden_size
    head = LinearSegHead(embed_dim=D, n_classes=1).to(device=device, dtype=dtype)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-3)

    ds = DownstreamDataset(entries, active_head_ids=["breast_lesion_seg"])
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=_downstream_collate)

    head.train()
    n = 0
    for batch in loader:
        imgs     = batch["image"].to(device=device, dtype=dtype)
        pad_mask = batch.get("padding_mask")
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        B, _, H, W = imgs.shape
        ph, pw = H // 16, W // 16
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=device)

        opt.zero_grad()
        with torch.no_grad():
            feats = bb(imgs, padding_mask=pad_mask)
        patch_tokens = feats["patch_tokens"]                  # (B, N, D)
        seg_logits   = head(patch_tokens, ph=ph, pw=pw)       # (B, 1, H, W)

        seg_mask = batch.get("seg_mask")
        if seg_mask is not None and seg_mask.shape[-1] == W:
            seg_mask = seg_mask.to(device=device, dtype=dtype)
            if seg_mask.shape[1] != 1:
                seg_mask = seg_mask[:, :1]
            loss = F.binary_cross_entropy_with_logits(seg_logits, seg_mask)
        else:
            loss = seg_logits.mean() * 0.0
        loss.backward()
        opt.step()

        log.info("  Phase4/seg batch=%d  loss=%.4f  seg_logits.shape=%s",
                 n + 1, loss.item(), tuple(seg_logits.shape))
        assert torch.isfinite(loss)
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    log.info("Phase4/segmentation PASS (%d batches)", n)


def _phase4_regression_smoke(
    echonet_entries: List[USManifestEntry], device: str
) -> None:
    """Ejection-fraction regression on EchoNet."""
    if not echonet_entries:
        log.warning("Phase4/reg SKIP — no EchoNet entries")
        return

    dtype = torch.float32
    bb   = _build_backbone_frozen(device, dtype)
    D    = bb.hidden_size
    head = RegressionHead(embed_dim=D, output_min=0.0, output_max=100.0).to(
        device=device, dtype=dtype)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-3)

    ds = DownstreamDataset(echonet_entries, active_head_ids=["cardiac_ef_regression"])
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=_downstream_collate)

    head.train()
    n = 0
    for batch in loader:
        imgs     = batch["image"].to(device=device, dtype=dtype)
        pad_mask = batch.get("padding_mask")
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=device)

        opt.zero_grad()
        with torch.no_grad():
            feats = bb(imgs, padding_mask=pad_mask)
        pred = head(feats["cls"])                             # (B, 1)

        # EF target from source_meta or label_targets
        label_targets = batch.get("label_targets", [])
        ef_vals = []
        for lt in label_targets:
            if hasattr(lt, "head_id") and lt.head_id == "cardiac_ef_regression":
                ef_vals.append(lt.value)
        if ef_vals:
            target = torch.tensor(ef_vals, device=device, dtype=dtype).unsqueeze(1)
            loss = F.smooth_l1_loss(pred, target)
        else:
            loss = pred.mean() * 0.0
        loss.backward()
        opt.step()

        log.info("  Phase4/reg batch=%d  loss=%.4f  pred.shape=%s",
                 n + 1, loss.item(), tuple(pred.shape))
        assert torch.isfinite(loss)
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    log.info("Phase4/regression PASS (%d batches)", n)


def _patient_collate(samples: list) -> dict:
    """
    Collate PatientLevelDataset samples.

    Each sample has:
      frames:      (max_frames, C, H, W)
      frame_mask:  (max_frames,) bool
      label_targets: list[LabelTarget]

    Frames are padded to batch-max (H, W) preserving native resolution.
    """
    patch_size = 16
    max_h = max(s["frames"].shape[-2] for s in samples)
    max_w = max(s["frames"].shape[-1] for s in samples)
    max_h = ((max_h + patch_size - 1) // patch_size) * patch_size
    max_w = ((max_w + patch_size - 1) // patch_size) * patch_size

    F_  = samples[0]["frames"].shape[0]
    C   = samples[0]["frames"].shape[1]
    B   = len(samples)
    ph, pw = max_h // patch_size, max_w // patch_size

    frames_t    = torch.zeros(B, F_, C, max_h, max_w)
    frame_masks = torch.zeros(B, F_, dtype=torch.bool)
    pad_masks   = torch.zeros(B, ph, pw, dtype=torch.bool)

    for i, s in enumerate(samples):
        h, w = s["frames"].shape[-2], s["frames"].shape[-1]
        frames_t[i, :, :, :h, :w] = s["frames"]
        frame_masks[i]              = s["frame_mask"]
        vh, vw = h // patch_size, w // patch_size
        pad_masks[i, :vh, :vw]     = True

    return {
        "frames":        frames_t,
        "frame_mask":    frame_masks,
        "padding_mask":  pad_masks,
        "label_targets": [s["label_targets"] for s in samples],
    }


def _phase4_patient_cls_smoke(
    benin_entries: List[USManifestEntry], device: str
) -> None:
    """Patient-level TB classification on Benin-LUS."""
    if not benin_entries:
        log.warning("Phase4/patient_cls SKIP — no Benin entries")
        return

    dtype = torch.float32
    bb   = _build_backbone_frozen(device, dtype)
    D    = 384
    head = LinearClsHead(embed_dim=D, n_classes=1).to(device=device, dtype=dtype)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-3)

    ds = PatientLevelDataset(
        benin_entries,
        active_head_ids=["lus_patient_tb"],
        max_frames=4,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=_patient_collate)

    head.train()
    n = 0
    for batch in loader:
        frames    = batch["frames"].to(device=device, dtype=dtype)  # (B, F, C, H, W)
        fm        = batch["frame_mask"].to(device=device)            # (B, F)
        pad_mask  = batch["padding_mask"].to(device=device)          # (B, ph, pw)
        B, F_, C_, H, W = frames.shape

        if C_ == 1:
            frames = frames.repeat(1, 1, 3, 1, 1)

        # Flatten frames, run backbone, mean-pool valid frames for patient repr
        frames_flat = frames.view(B * F_, frames.shape[2], H, W)
        pm_flat     = pad_mask.unsqueeze(1).expand(B, F_, -1, -1
                       ).reshape(B * F_, *pad_mask.shape[1:])

        opt.zero_grad()
        with torch.no_grad():
            feats_flat = bb(frames_flat, padding_mask=pm_flat)
        cls_flat     = feats_flat["cls"].view(B, F_, D)             # (B, F, D)
        fm_f         = fm.float().unsqueeze(-1)                     # (B, F, 1)
        patient_feat = (cls_flat * fm_f).sum(1) / fm_f.sum(1).clamp(min=1)

        logits = head(patient_feat)                                 # (B, 1)

        label_targets_list = batch.get("label_targets", [])
        tb_vals = []
        for patient_targets in label_targets_list:
            for lt in (patient_targets if isinstance(patient_targets, list) else []):
                if hasattr(lt, "head_id") and lt.head_id == "lus_patient_tb":
                    tb_vals.append(float(lt.value))
                    break
        if tb_vals:
            target = torch.tensor(tb_vals, device=device, dtype=dtype).unsqueeze(1)
            target = target.clamp(0, 1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        else:
            loss = logits.mean() * 0.0
        loss.backward()
        opt.step()

        log.info("  Phase4/patient_cls batch=%d  loss=%.4f  logits.shape=%s",
                 n + 1, loss.item(), tuple(logits.shape))
        assert torch.isfinite(loss)
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    log.info("Phase4/patient_classification PASS (%d batches)", n)


def phase4_smoke(
    dm: USFoundationDataModule,
    busi_entries: List[USManifestEntry],
    echonet_entries: List[USManifestEntry],
    benin_entries: List[USManifestEntry],
    device: str,
) -> None:
    log.info("=== Phase 4: Downstream Heads ===")
    _phase4_classification_smoke(busi_entries, device)
    _phase4_segmentation_smoke(busi_entries, device)
    _phase4_regression_smoke(echonet_entries, device)
    _phase4_patient_cls_smoke(benin_entries, device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = _auto_device()
    log.info("Device: %s", device)

    # Always rebuild so new datasets are included
    log.info("Building combined smoke manifest …")
    build_combined_manifest(force=True)

    # Cache per-dataset entries for Phase 4 downstream heads
    busi_entries    = _build_busi_entries()
    echonet_entries = _build_echonet_entries()
    benin_entries   = _build_benin_entries()

    # DataModule
    cfg = load_smoke_config()
    dm  = build_datamodule(cfg)

    results = {}

    def _run(name: str, fn, *args):
        skip_var = f"US_SKIP_{name.upper().replace(' ', '_')}"
        if os.environ.get(skip_var, "0") == "1":
            log.info("Skipping %s (env %s=1)", name, skip_var)
            results[name] = "SKIP"
            return
        try:
            fn(*args)
            results[name] = "PASS"
        except Exception:
            results[name] = "FAIL"
            log.error("%s FAILED:\n%s", name, traceback.format_exc())

    _run("PHASE1", phase1_smoke, dm, device)
    _run("PHASE2", phase2_smoke, dm, device)
    _run("PHASE3", phase3_smoke, dm, device)
    _run("PHASE4", phase4_smoke, dm, busi_entries, echonet_entries, benin_entries, device)

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    for phase, status in results.items():
        icon = "✓" if status == "PASS" else ("–" if status == "SKIP" else "✗")
        print(f"  {icon}  {phase:<12}  {status}")
    print("=" * 60)

    if any(v == "FAIL" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
