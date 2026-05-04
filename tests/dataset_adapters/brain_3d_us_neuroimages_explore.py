"""
brain_3d_us_neuroimages_explore.py  ·  Real-data exploration for 3D-US-Neuroimages
====================================================================================

Usage:
    python -m tests.dataset_adapters.brain_3d_us_neuroimages_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.brain_3d_us_neuroimages import ThreeDUSNeuroimagesAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="3D-US-Neuroimages-Dataset",
        adapter_cls=ThreeDUSNeuroimagesAdapter,
        default_root=Path(
            "/capstor/store/cscs/swissai/a127/ultrasound/raw/brain/3D-US-Neuroimages-Dataset"
        ),
        default_out_dir=Path("dataset_exploration_outputs/brain_3d_us_neuroimages"),
        root_env_var="US_3D_US_NEUROIMAGES_ROOT",
        out_dir_env_var="US_3D_US_NEUROIMAGES_OUT_DIR",
    )


if __name__ == "__main__":
    main()
