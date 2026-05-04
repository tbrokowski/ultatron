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

# ── Non-cardiac ───────────────────────────────────────────────────────────────
from .busi    import BUSIAdapter
from .tn3k    import TN3KAdapter
from .annotated_heterogeneous_us_db import AnnotatedHeterogeneousUSDBAdapter
from .bite import BITEAdapter
from .brain_3d_us_neuroimages import ThreeDUSNeuroimagesAdapter
from .common_carotid import CommonCarotidArteryImagesAdapter
from .cubs import CUBSAdapter
from .dermatologic_skin_lesions import DermatologicSkinLesionsAdapter
from .erdes import ERDESAdapter
from .remind2reg import ReMIND2RegAdapter
from .remind_brain_ius import REMINDBrainIUSAdapter
from .resect import RESECTAdapter
from .stu_hospital import STUHospitalAdapter

# ── Lung ──────────────────────────────────────────────────────────────────────
from .lung.benin_lus import BeninLUSAdapter
from .lung.rsa_lus   import RSALUSAdapter

# ── Liver ─────────────────────────────────────────────────────────────────────
from .liver.aul   import AULAdapter
from .liver.us105 import US105Adapter

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
    # Breast / thyroid
    "BUSI":                     BUSIAdapter,
    "TN3K":                     TN3KAdapter,
    # Vascular / carotid
    "CUBS":                     CUBSAdapter,
    "Common-Carotid-Artery-Ultrasound-Images": CommonCarotidArteryImagesAdapter,
    # Brain / multi-organ / ocular / skin
    "3D-US-Neuroimages-Dataset": ThreeDUSNeuroimagesAdapter,
    "BITE":                     BITEAdapter,
    "REMIND-Brain-iUS":         REMINDBrainIUSAdapter,
    "RESECT":                   RESECTAdapter,
    "ReMIND2Reg":               ReMIND2RegAdapter,
    "STU-Hospital-master":      STUHospitalAdapter,
    "annotated_heterogeneous_us_db": AnnotatedHeterogeneousUSDBAdapter,
    "ERDES":                    ERDESAdapter,
    "Dermatologic-US-Skin-Lesions": DermatologicSkinLesionsAdapter,
    # Lung
    "Benin-LUS":                BeninLUSAdapter,
    "RSA-LUS":                  RSALUSAdapter,
    # Liver
    "AUL":                      AULAdapter,
    "105US":                    US105Adapter,
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
    # Non-cardiac
    "BUSIAdapter", "TN3KAdapter", "CUBSAdapter",
    "CommonCarotidArteryImagesAdapter",
    "ThreeDUSNeuroimagesAdapter", "BITEAdapter", "REMINDBrainIUSAdapter",
    "RESECTAdapter", "ReMIND2RegAdapter", "STUHospitalAdapter",
    "AnnotatedHeterogeneousUSDBAdapter", "ERDESAdapter",
    "DermatologicSkinLesionsAdapter",
    # Lung
    "BeninLUSAdapter", "RSALUSAdapter",
    # Liver
    "AULAdapter", "US105Adapter",
    # Helpers
    "ADAPTER_REGISTRY", "build_adapter", "build_manifest_for_dataset",
]
