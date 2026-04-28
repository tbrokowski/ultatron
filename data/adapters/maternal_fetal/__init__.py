"""
data/adapters/maternal_fetal/__init__.py
Re-exports all maternal-fetal dataset adapters.
"""
from .acouslic import ACOUSLICAIAdapter

__all__ = [
    "ACOUSLICAIAdapter",
]
