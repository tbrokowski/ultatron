"""
erdes_explore.py  ·  Real-data exploration for the ERDES adapter
================================================================

Usage:
    python -m tests.dataset_adapters.erdes_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.erdes import ERDESAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="ERDES",
        adapter_cls=ERDESAdapter,
        default_root=Path("/capstor/store/cscs/swissai/a127/ultrasound/raw/ocular/ERDES"),
        default_out_dir=Path("dataset_exploration_outputs/erdes"),
        root_env_var="US_ERDES_ROOT",
        out_dir_env_var="US_ERDES_OUT_DIR",
    )


if __name__ == "__main__":
    main()
