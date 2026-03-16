"""data.infra — storage routing (CSCS-specific, swappable)."""
from .cscs_paths import CSCSConfig
from .storage import StorageConfig

__all__ = ["CSCSConfig", "StorageConfig"]
