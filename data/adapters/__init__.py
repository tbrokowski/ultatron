"""
data/adapters/__init__.py
Adapter registry and convenience imports.
"""
from .base      import BaseAdapter
from .camus     import CAMUSAdapter
from .echonet   import EchoNetDynamicAdapter
from .mimic_echo import MIMICEchoAdapter
from .busi      import BUSIAdapter
from .tn3k      import TN3KAdapter

# Registry: dataset_id -> adapter class
ADAPTER_REGISTRY = {
    "CAMUS":            CAMUSAdapter,
    "EchoNet-Dynamic":  EchoNetDynamicAdapter,
    "MIMIC-IV-ECHO":    MIMICEchoAdapter,
    "BUSI":             BUSIAdapter,
    "TN3K":             TN3KAdapter,
}


def build_adapter(dataset_id: str, root: str, **kwargs) -> BaseAdapter:
    """Instantiate a registered adapter by dataset_id."""
    if dataset_id not in ADAPTER_REGISTRY:
        raise KeyError(
            f"No adapter registered for '{dataset_id}'. "
            f"Available: {sorted(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[dataset_id](root=root, **kwargs)


__all__ = [
    "BaseAdapter",
    "CAMUSAdapter", "EchoNetDynamicAdapter", "MIMICEchoAdapter",
    "BUSIAdapter", "TN3KAdapter",
    "ADAPTER_REGISTRY", "build_adapter",
]
