"""
data/adapters/__init__.py
Adapter registry and convenience imports.
"""
from pathlib import Path
from typing import Optional

from .base import BaseAdapter

# ── Cardiac ───────────────────────────────────────────────────────────────────
from .cardiac.camus             import CAMUSAdapter
from .cardiac.echonet           import EchoNetDynamicAdapter
from .cardiac.echonet_pediatric import EchoNetPediatricAdapter
from .cardiac.echonet_lvh       import EchoNetLVHAdapter
from .cardiac.mimic_echo        import MIMICEchoAdapter
from .cardiac.mimic_lvvol_a4c   import MIMICLVVolA4CAdapter
from .cardiac.ted               import TEDAdapter
from .cardiac.unity             import UnityAdapter
from .cardiac.cardiacudc        import CardiacUDCAdapter
from .cardiac.echocp            import EchoCPAdapter
from .cardiac.echocardiogram_uci   import EchocardiogramUCIAdapter
from .cardiac.mimic_echoqa         import MIMICEchoQAAdapter
from .cardiac.cactus               import CACTUSAdapter
from .cardiac.mitea                import MITEAAdapter
from .cardiac.physionet_cardiac     import PhysioNetCardiacAdapter

# ── Breast ────────────────────────────────────────────────────────────────────
from .breast.busi                       import BUSIAdapter
from .breast.breast_adapter             import BrEaSTAdapter
from .breast.buid_adapter               import BUIDAdapter
from .breast.bus_b_adapter              import BUSBAdapter
from .breast.bus_bra_adapter            import BUSBRAAdapter
from .breast.bus_uc_adapter             import BUSUCAdapter
from .breast.bus_uclm_adapter           import BUSUCLMAdapter
from .breast.busv_adapter               import BUSVAdapter
from .breast.chinese_us_report_adapter  import ChineseUSReportBreastAdapter
from .breast.gdph_sysucc_adapter        import GDPHSYSUCCAdapter
from .breast.s1_adapter                 import S1Adapter
from .breast.busi_whu                   import BUSIWHUAdapter
from .breast.midi_b                     import MidiBAdapter
from .breast.stanford_bus               import STAnfordBUSAdapter

# ── Lung ──────────────────────────────────────────────────────────────────────
from .lung.benin_lus       import BeninLUSAdapter
from .lung.covidx_us       import COVIDxUSAdapter
from .lung.lus_multicenter import LUSMulticenterAdapter
from .lung.open_pocus      import OpenPOCUSAdapter
from .lung.rsa_lus         import RSALUSAdapter
from .lung.luss_phantom    import LUSSPhantomAdapter
from .lung.lung_database   import LungDatabaseAdapter
from .lung.pocus_covid     import PocusCovidAdapter
from .lung.benin_videos    import BeninVideosAdapter
from .lung.covid_blues     import COVIDBLUESAdapter
from .lung.pocus_lus       import POCUSLUSAdapter
from .lung.ultrasound_lus  import ULTRASOUNDLUSAdapter
from .lung.lus_data          import LUSDataAdapter

# ── Liver ─────────────────────────────────────────────────────────────────────
from .liver.aul               import AULAdapter
from .liver.us105             import US105Adapter
from .liver.fatty_liver_bmode import FattyLiverBmodeAdapter
from .liver.liver_cv_project  import LiverCVProjectAdapter
from .liver.lepset            import LEPsetAdapter
from .liver.ctrus             import CTRUSAdapter
from .liver.bmode_ceus_liver  import BModeCEUSLiverAdapter
from .liver.behsof              import BEHSOFAdapter
from .liver.ultrasound_elastography_liver_cancer import UltrasoundElastographyLiverCancerAdapter

# ── Maternal / fetal ──────────────────────────────────────────────────────────
from .maternal_fetal.acouslic                        import ACOUSLICAIAdapter
from .maternal_fetal.fetal_abdominal_structures      import FASSAdapter
from .maternal_fetal.fetal_planes_db                 import FetalPlanesDBAdapter
from .maternal_fetal.focus                           import FOCUSAdapter
from .maternal_fetal.fpus23                          import FPUS23Adapter
from .maternal_fetal.fugc                            import FUGCAdapter
from .maternal_fetal.fh_ps_aop                       import FHPSAOPAdapter
from .maternal_fetal.hc18                            import HC18Adapter
from .maternal_fetal.iugc2024                        import IUGC2024Adapter
from .maternal_fetal.jnu_ifm                         import JNUIFMAdapter
from .maternal_fetal.large_scale_fetal_head_biometry import LargeScaleFetalHeadBiometryAdapter
from .maternal_fetal.maternal_fetal_us_video_intrapartum import MaternalFetalUSVideoIntrapartumAdapter
from .maternal_fetal.pbf_us1                         import PBFUS1Adapter
from .maternal_fetal.psfhs                           import PSFHSAdapter
from .maternal_fetal.ultrasound_fetus              import UltrasoundFetusAdapter
from .maternal_fetal.oc4us                         import OC4USAdapter
from .maternal_fetal.fast_u_net                    import FastUNetAdapter

# ── Muscle / MSK ──────────────────────────────────────────────────────────────
from .muscle.stmus_nda           import STMUSNDAAdapter
from .muscle.fallmud             import FALLMUDAdapter
from .muscle.luminous            import LUMINOUSAdapter
from .muscle.deep_mtj            import DeepMTJAdapter
from .muscle.knee_us_jocohs       import KneeUSJoCoHSAdapter
from .muscle.tus_rec             import TUSRECAdapter
from .muscle.tus_rec_val         import TUSRECValAdapter
from .muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
from .muscle.msk_heckmatt          import MSKHeckmattAdapter
from .muscle.msk_nmd               import MSKNMDAdapter
from .muscle.open_hip_dysplasia    import OpenHipDysplasiaAdapter

# ── Gallbladder / GI ──────────────────────────────────────────────────────────
from .gallbladder.gist514_db                          import GIST514DBAdapter
from .gallbladder.regensburg_pediatric_appendicitis   import RegensburgPediatricAppendicitisAdapter
from .gallbladder.gbcu                                import GBCUAdapter

# ── Abdomen ───────────────────────────────────────────────────────────────────
from .abdomen.abdomen_us import AbdomenUSAdapter
from .abdomen.abdomen_us_liver import AbdomenUSLiverAdapter
from .abdomen.cptac_pda  import CPTACPDAAdapter

# ── Kidney ────────────────────────────────────────────────────────────────────
from .kidney.normal_kidney_cv import NormalKidneyCVAdapter
from .kidney.kidney_us        import KidneyUSAdapter

# ── Thyroid ───────────────────────────────────────────────────────────────────
from .thyroid.segthy  import SegthyAdapter
from .thyroid.ddti    import DDTIAdapter
from .thyroid.tg3k    import TG3KAdapter
from .thyroid.tn3k    import TN3KAdapter
from .thyroid.tn5000  import TN5000Adapter
from .thyroid.tnscui  import TNSCUIAdapter
from .thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter

# ── Ovarian ───────────────────────────────────────────────────────────────────
from .ovarian.mmotu3d                 import MMOTU3DAdapter
from .ovarian.mmotu2d                 import MMOTU2DAdapter
from .ovarian.polycystic_ovary_telkom import PolycysticOvaryTelkomAdapter
from .ovarian.pcosgen                 import PCOSGenAdapter

# ── Prostate ──────────────────────────────────────────────────────────────────
from .prostate.openpros import OpenProsAdapter
from .prostate.museg import MuSegAdapter
from .prostate.muregpro import MuRegProAdapter
from .prostate.micro_ultrasound_prostate_seg import MicroUltrasoundProstateSegAdapter
from .prostate.mri_us_biopsy import ProstateMRIUSBiopsyAdapter

# ── Vascular / carotid ──────────────────────────────────────────────────────────
from .vascular.cubs           import CUBSAdapter
from .vascular.common_carotid import CommonCarotidArteryImagesAdapter

# ── Brain ───────────────────────────────────────────────────────────────────────
from .brain.bratious              import BraTioUSAdapter
from .brain.bite                  import BITEAdapter
from .brain.brain_3d_us_neuroimages import ThreeDUSNeuroimagesAdapter
from .brain.remind_brain_ius      import REMINDBrainIUSAdapter
from .brain.resect                import RESECTAdapter
from .brain.remind2reg            import ReMIND2RegAdapter

# ── Multi-organ ─────────────────────────────────────────────────────────────────
from .multi_organ.annotated_heterogeneous_us_db import AnnotatedHeterogeneousUSDBAdapter
from .multi_organ.stu_hospital                  import STUHospitalAdapter
from .multi_organ.us_365k                       import US365KAdapter
from .multi_organ.usanotai                       import USAnotAIAdapter

# ── Ocular ──────────────────────────────────────────────────────────────────────
from .ocular.erdes import ERDESAdapter

# ── Skin ────────────────────────────────────────────────────────────────────────
from .skin.dermatologic_skin_lesions import DermatologicSkinLesionsAdapter

# ── Nerve ───────────────────────────────────────────────────────────────────────
from .nerve.optic_nerve_sheaths  import OpticNerveSheathsAdapter
from .nerve.us_guided_anesthesia import USGuidedAnesthesiaAdapter

# ── Generic mask-pair factory ─────────────────────────────────────────────────
from .generic_mask import GenericMaskPairAdapter, _make_generic


# Registry: dataset_id -> adapter class
ADAPTER_REGISTRY = {
    # Cardiac — fully labelled
    "CAMUS":                    CAMUSAdapter,
    "EchoNet-Dynamic":          EchoNetDynamicAdapter,
    "EchoNet-Pediatric":        EchoNetPediatricAdapter,
    "EchoNet-LVH":              EchoNetLVHAdapter,
    "MIMIC-IV-ECHO":            MIMICEchoAdapter,
    "MIMIC-IV-Echo-LVVol-A4C":  MIMICLVVolA4CAdapter,
    "TED":                      TEDAdapter,
    "Unity-Echo":               UnityAdapter,
    "CardiacUDC":               CardiacUDCAdapter,
    "EchoCP":                   EchoCPAdapter,
    "Echocardiogram-UCI":       EchocardiogramUCIAdapter,
    "MIMIC-EchoQA":             MIMICEchoQAAdapter,
    "CACTUS":                   CACTUSAdapter,
    "MITEA":                    MITEAAdapter,
    "PhysioNet-cardiac":        PhysioNetCardiacAdapter,
    # Breast / thyroid
    "BUSI":                                    BUSIAdapter,
    "TN3K":                                    TN3KAdapter,
    "BrEaST":                                  BrEaSTAdapter,
    "BUID":                                    BUIDAdapter,
    "BUS-B":                                   BUSBAdapter,
    "BUS-BRA":                                 BUSBRAAdapter,
    "BUS-UC":                                  BUSUCAdapter,
    "BUS-UCLM":                                BUSUCLMAdapter,
    "BUSV":                                    BUSVAdapter,
    "GDPH-SYSUCC":                             GDPHSYSUCCAdapter,
    "Chinese-US-Report-Breast":                ChineseUSReportBreastAdapter,
    "S1":                                      S1Adapter,
    "busi-whu":                                BUSIWHUAdapter,
    "midi-b":                                  MidiBAdapter,
    "STAnford-BUS":                            STAnfordBUSAdapter,
    # Vascular / carotid
    "CUBS":                                    CUBSAdapter,
    "Common-Carotid-Artery-Ultrasound-Images": CommonCarotidArteryImagesAdapter,
    # Brain / multi-organ / ocular / skin
    "3D-US-Neuroimages-Dataset":               ThreeDUSNeuroimagesAdapter,
    "BITE":                                    BITEAdapter,
    "REMIND-Brain-iUS":                        REMINDBrainIUSAdapter,
    "RESECT":                                  RESECTAdapter,
    "ReMIND2Reg":                              ReMIND2RegAdapter,
    "STU-Hospital-master":                     STUHospitalAdapter,
    "annotated_heterogeneous_us_db":           AnnotatedHeterogeneousUSDBAdapter,
    "US-365K":                                 US365KAdapter,
    "USAnotAI-master":                         USAnotAIAdapter,
    "ERDES":                                   ERDESAdapter,
    "Dermatologic-US-Skin-Lesions":            DermatologicSkinLesionsAdapter,
    # Lung
    "Benin-LUS":                BeninLUSAdapter,
    "COVIDx-US":                COVIDxUSAdapter,
    "LUS-multicenter-2025":     LUSMulticenterAdapter,
    "OpenPOCUS":                OpenPOCUSAdapter,
    "RSA-LUS":                  RSALUSAdapter,
    "LUSS-PHANTOM":             LUSSPhantomAdapter,
    "Lung-Database":            LungDatabaseAdapter,
    "Pocus-covid":              PocusCovidAdapter,
    "BeninVideos":              BeninVideosAdapter,
    "COVID-BLUES":              COVIDBLUESAdapter,
    "POCUS-LUS":                POCUSLUSAdapter,
    "ULTRASOUND-LUS":           ULTRASOUNDLUSAdapter,
    "LUS-data":                 LUSDataAdapter,
    # Liver
    "AUL":                      AULAdapter,
    "105US":                    US105Adapter,
    "fatty-liver-bmode":        FattyLiverBmodeAdapter,
    "liver-CV-project":         LiverCVProjectAdapter,
    "LEPset":                   LEPsetAdapter,
    "B-mode-CEUS-liver":        BModeCEUSLiverAdapter,
    "C-TRUS":                   CTRUSAdapter,
    "BEHSOF":                   BEHSOFAdapter,
    "ultrasound-elastography-liver-cancer": UltrasoundElastographyLiverCancerAdapter,
    # Maternal / fetal
    "ACOUSLIC-AI":                         ACOUSLICAIAdapter,
    "FASS":                                FASSAdapter,
    "FETAL_PLANES_DB":                     FetalPlanesDBAdapter,
    "FOCUS":                               FOCUSAdapter,
    "FPUS23":                              FPUS23Adapter,
    "FUGC":                                FUGCAdapter,
    "FH-PS-AOP":                           FHPSAOPAdapter,
    "HC18":                                HC18Adapter,
    "IUGC2024":                            IUGC2024Adapter,
    "JNU-IFM":                             JNUIFMAdapter,
    "Large-Scale-Fetal-Head-Biometry":     LargeScaleFetalHeadBiometryAdapter,
    "maternal-fetal-us-video-intrapartum": MaternalFetalUSVideoIntrapartumAdapter,
    "PBF-US1":                             PBFUS1Adapter,
    "PSFHS":                               PSFHSAdapter,
    "ultrasound-fetus-dataset":            UltrasoundFetusAdapter,
    "OC4US":                               OC4USAdapter,
    "Fast-U-Net":                          FastUNetAdapter,
    # Muscle / MSK
    "STMUS-NDA":        STMUSNDAAdapter,
    "FALLMUD":          FALLMUDAdapter,
    "LUMINOUS":         LUMINOUSAdapter,
    "deepMTJ":          DeepMTJAdapter,
    "KneeUSJoCoHS":     KneeUSJoCoHSAdapter,
    "TUS-REC":          TUSRECAdapter,
    "TUS-REC-Val":      TUSRECValAdapter,
    "SpinalCordInjuryUS": SpinalCordInjuryUSAdapter,
    "msk-heckmatt-radboud": MSKHeckmattAdapter,
    "msk-nmd-radboud":      MSKNMDAdapter,
    "open-hip-dysplasia":   OpenHipDysplasiaAdapter,
    # Gallbladder / GI
    "GIST514-DB":           GIST514DBAdapter,
    "RegensburgPedAppend":  RegensburgPediatricAppendicitisAdapter,
    "GBCU":                 GBCUAdapter,
    # Abdomen
    "AbdomenUS":        AbdomenUSAdapter,
    "AbdomenUS-liver":  AbdomenUSLiverAdapter,
    "cptac-pda":        CPTACPDAAdapter,
    # Kidney
    "KidneyUS":         KidneyUSAdapter,
    "Normal-Kidney-CV": NormalKidneyCVAdapter,
    # Thyroid
    "Segthy-Dataset":   SegthyAdapter,
    "DDTI":             DDTIAdapter,
    "TG3K":             TG3KAdapter,
    "TN5000":           TN5000Adapter,
    "TNSCUI":           TNSCUIAdapter,
    "Thyroid-Nodule-Pathology": ThyroidNodulePathologyAdapter,
    "MuSeg":                    MuSegAdapter,
    "Micro-Ultrasound-Prostate-Segmentation": MicroUltrasoundProstateSegAdapter,
    # Ovarian
    "MMOTU-3D":         MMOTU3DAdapter,
    "MMOTU-2D":         MMOTU2DAdapter,
    "Polycystic-Ovary-US-Telkom": PolycysticOvaryTelkomAdapter,
    "PCOSGen":          PCOSGenAdapter,
    # Prostate
    "ProstateSeg":      OpenProsAdapter,
    "Prostate-MRI-US-Biopsy": ProstateMRIUSBiopsyAdapter,
    "muregpro":         MuRegProAdapter,
    # Brain
    "braTioUS":         BraTioUSAdapter,
    # Nerve
    "optic-nerve-sheaths":  OpticNerveSheathsAdapter,
    "us-guided-anesthesia": USGuidedAnesthesiaAdapter,
}


def build_adapter(dataset_id: str, root: str, **kwargs) -> BaseAdapter:
    """Instantiate a registered adapter by dataset_id."""
    if dataset_id not in ADAPTER_REGISTRY:
        raise KeyError(
            f"No adapter registered for '{dataset_id}'. "
            f"Available: {sorted(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[dataset_id](root=root, **kwargs)


def build_manifest_for_dataset(
    dataset_id: str,
    root: Path,
    writer,
    split_override: Optional[str] = None,
) -> int:
    """
    Run the adapter for dataset_id at root, write all entries to writer.
    writer must have a .write(USManifestEntry) method (e.g. ManifestWriter).
    Returns the number of entries written.
    """
    adapter = build_adapter(dataset_id, str(root), split_override=split_override)
    n = 0
    for e in adapter.iter_entries():
        writer.write(e)
        n += 1
    return n


__all__ = [
    "BaseAdapter",
    # Cardiac
    "CAMUSAdapter", "EchoNetDynamicAdapter", "EchoNetPediatricAdapter",
    "EchoNetLVHAdapter", "MIMICEchoAdapter", "MIMICLVVolA4CAdapter",
    "TEDAdapter", "UnityAdapter", "CardiacUDCAdapter", "EchoCPAdapter",
    "EchocardiogramUCIAdapter",
    "MIMICEchoQAAdapter", "CACTUSAdapter", "MITEAAdapter",
    # Non-cardiac
    "BUSIAdapter", "TN3KAdapter", "CUBSAdapter",
    "CommonCarotidArteryImagesAdapter",
    "ThreeDUSNeuroimagesAdapter", "BITEAdapter", "REMINDBrainIUSAdapter",
    "RESECTAdapter", "ReMIND2RegAdapter", "STUHospitalAdapter",
    "AnnotatedHeterogeneousUSDBAdapter", "USAnotAIAdapter", "ERDESAdapter",
    "DermatologicSkinLesionsAdapter",
    # Breast
    "BrEaSTAdapter", "BUIDAdapter", "BUSBAdapter", "BUSBRAAdapter",
    "BUSUCAdapter", "BUSUCLMAdapter", "BUSVAdapter",
    "ChineseUSReportBreastAdapter", "GDPHSYSUCCAdapter", "S1Adapter",
    "BUSIWHUAdapter", "MidiBAdapter", "STAnfordBUSAdapter",
    # Lung
    "BeninLUSAdapter", "COVIDxUSAdapter", "LUSMulticenterAdapter",
    "OpenPOCUSAdapter", "RSALUSAdapter",
    "LUSSPhantomAdapter", "LungDatabaseAdapter", "PocusCovidAdapter",
    "BeninVideosAdapter", "COVIDBLUESAdapter", "POCUSLUSAdapter",
    "ULTRASOUNDLUSAdapter",
    # Liver
    "AULAdapter", "US105Adapter", "FattyLiverBmodeAdapter",
    "LiverCVProjectAdapter", "LEPsetAdapter",
    "CTRUSAdapter", "BModeCEUSLiverAdapter", "BEHSOFAdapter",
    # Maternal / fetal
    "ACOUSLICAIAdapter", "FASSAdapter", "FetalPlanesDBAdapter",
    "FOCUSAdapter", "FPUS23Adapter", "FUGCAdapter", "FHPSAOPAdapter",
    "HC18Adapter", "IUGC2024Adapter", "JNUIFMAdapter",
    "LargeScaleFetalHeadBiometryAdapter",
    "MaternalFetalUSVideoIntrapartumAdapter", "PBFUS1Adapter", "PSFHSAdapter",
    "UltrasoundFetusAdapter", "OC4USAdapter",
    # Muscle / MSK
    "STMUSNDAAdapter", "FALLMUDAdapter", "LUMINOUSAdapter", "DeepMTJAdapter",
    "KneeUSJoCoHSAdapter", "TUSRECAdapter", "TUSRECValAdapter",
    "SpinalCordInjuryUSAdapter",
    "MSKHeckmattAdapter", "MSKNMDAdapter", "OpenHipDysplasiaAdapter",
    # Gallbladder / GI
    "GIST514DBAdapter", "RegensburgPediatricAppendicitisAdapter",
    "GBCUAdapter",
    # Abdomen
    "AbdomenUSAdapter", "CPTACPDAAdapter",
    # Kidney
    "NormalKidneyCVAdapter", "KidneyUSAdapter",
    # Thyroid
    "SegthyAdapter", "DDTIAdapter", "TG3KAdapter", "TN5000Adapter",
    "TNSCUIAdapter",
    # Ovarian
    "MMOTU3DAdapter", "MMOTU2DAdapter", "PolycysticOvaryTelkomAdapter",
    "PCOSGenAdapter",
    # Prostate
    "OpenProsAdapter",
    # Brain
    "BraTioUSAdapter",
    # Nerve
    "OpticNerveSheathsAdapter", "USGuidedAnesthesiaAdapter",
    # Generic
    "GenericMaskPairAdapter", "_make_generic",
    # Helpers
    "ADAPTER_REGISTRY", "build_adapter", "build_manifest_for_dataset",
]
