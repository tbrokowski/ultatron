"""
data/adapters/__init__.py
Adapter registry and convenience imports.
"""
from pathlib import Path
from typing import Optional

from .base import BaseAdapter
from .breast.bus_bra_adapter import BUSBRAAdapter
from .breast.bus_uclm_adapter import BUSUCLMAdapter
from .breast.breast_adapter import BrEaSTAdapter
from .breast.buid_adapter     import BUIDAdapter
from .breast.s1_adapter import S1Adapter
from .breast.busv_adapter import BUSVAdapter
from .breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
from .breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter







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
from .breast.bus_uc_adapter import BUSUCAdapter


# ── Lung ──────────────────────────────────────────────────────────────────────
from .lung.benin_lus import BeninLUSAdapter
from .lung.rsa_lus   import RSALUSAdapter

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
    "BUS-BRA": BUSBRAAdapter,
    "BUS-UC": BUSUCAdapter,
    "BUS-UCLM": BUSUCLMAdapter,
    "BrEaST": BrEaSTAdapter,
    "BUID":     BUIDAdapter,
    "S1": S1Adapter,
    "BUSV": BUSVAdapter,
    "GDPH-SYSUCC": GDPHSYSUCCAdapter,
    "Chinese-US-Report-Breast": ChineseUSReportBreastAdapter,






    "TN3K":                     TN3KAdapter,
    # Lung
    "Benin-LUS":                BeninLUSAdapter,
    "RSA-LUS":                  RSALUSAdapter,
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
    "BUSIAdapter", "TN3KAdapter", "BUSBRAAdapter","BUSUCAdapter","BUSUCLMAdapter","BrEaSTAdapter","BUIDAdapter","S1Adapter","BUSVAdapter","GDPHSYSUCCAdapter","ChineseUSReportBreastAdapter",
    # Lung
    "BeninLUSAdapter", "RSALUSAdapter",
    # Helpers
    "ADAPTER_REGISTRY", "build_adapter", "build_manifest_for_dataset",
]
