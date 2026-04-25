"""
resect_explore.py  ·  Real-data exploration for the RESECT adapter
==================================================================

Usage:
    python -m tests.dataset_adapters.resect_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.resect import RESECTAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="RESECT",
        adapter_cls=RESECTAdapter,
        default_root=Path("/capstor/store/cscs/swissai/a127/ultrasound/raw/brain/RESECT"),
        default_out_dir=Path("dataset_exploration_outputs/resect"),
        root_env_var="US_RESECT_ROOT",
        out_dir_env_var="US_RESECT_OUT_DIR",
    )


if __name__ == "__main__":
    main()
