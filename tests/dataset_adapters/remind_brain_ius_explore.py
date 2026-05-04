"""
remind_brain_ius_explore.py  ·  Real-data exploration for ReMIND-Brain-iUS
============================================================================

Usage:
    python -m tests.dataset_adapters.remind_brain_ius_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.remind_brain_ius import REMINDBrainIUSAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="REMIND-Brain-iUS",
        adapter_cls=REMINDBrainIUSAdapter,
        default_root=Path(
            "/capstor/store/cscs/swissai/a127/ultrasound/raw/brain/REMIND-Brain-iUS"
        ),
        default_out_dir=Path("dataset_exploration_outputs/remind_brain_ius"),
        root_env_var="US_REMIND_BRAIN_IUS_ROOT",
        out_dir_env_var="US_REMIND_BRAIN_IUS_OUT_DIR",
    )


if __name__ == "__main__":
    main()
