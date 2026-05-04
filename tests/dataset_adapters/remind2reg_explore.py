"""
remind2reg_explore.py  ·  Real-data exploration for the ReMIND2Reg adapter
============================================================================

Usage:
    python -m tests.dataset_adapters.remind2reg_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.remind2reg import ReMIND2RegAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="ReMIND2Reg",
        adapter_cls=ReMIND2RegAdapter,
        default_root=Path("/capstor/store/cscs/swissai/a127/ultrasound/raw/brain/ReMIND2Reg"),
        default_out_dir=Path("dataset_exploration_outputs/remind2reg"),
        root_env_var="US_REMIND2REG_ROOT",
        out_dir_env_var="US_REMIND2REG_OUT_DIR",
    )


if __name__ == "__main__":
    main()
