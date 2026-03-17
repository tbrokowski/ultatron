"""
data/adapters/__init__.py
Adapter registry and convenience imports.
"""
from pathlib import Path
from typing import Optional

from .base       import BaseAdapter
from .camus      import CAMUSAdapter
from .echonet    import EchoNetDynamicAdapter
from .mimic_echo import MIMICEchoAdapter
from .busi       import BUSIAdapter
from .tn3k       import TN3KAdapter
from .benin_lus  import BeninLUSAdapter
from .rsa_lus    import RSALUSAdapter

# Registry: dataset_id -> adapter class
ADAPTER_REGISTRY = {
    "CAMUS":            CAMUSAdapter,
    "EchoNet-Dynamic":  EchoNetDynamicAdapter,
    "MIMIC-IV-ECHO":    MIMICEchoAdapter,
    "BUSI":             BUSIAdapter,
    "TN3K":             TN3KAdapter,
    "Benin-LUS":        BeninLUSAdapter,
    "RSA-LUS":          RSALUSAdapter,
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
    "CAMUSAdapter", "EchoNetDynamicAdapter", "MIMICEchoAdapter",
    "BUSIAdapter", "TN3KAdapter",
    "BeninLUSAdapter", "RSALUSAdapter",
    "ADAPTER_REGISTRY", "build_adapter", "build_manifest_for_dataset",
]
