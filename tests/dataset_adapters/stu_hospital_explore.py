"""
stu_hospital_explore.py  ·  Real-data exploration for the STU-Hospital adapter
================================================================================

Usage:
    python -m tests.dataset_adapters.stu_hospital_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.stu_hospital import STUHospitalAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="STU-Hospital-master",
        adapter_cls=STUHospitalAdapter,
        default_root=Path(
            "/capstor/store/cscs/swissai/a127/ultrasound/raw/multi_organ/STU-Hospital-master"
        ),
        default_out_dir=Path("dataset_exploration_outputs/stu_hospital"),
        root_env_var="US_STU_HOSPITAL_ROOT",
        out_dir_env_var="US_STU_HOSPITAL_OUT_DIR",
    )


if __name__ == "__main__":
    main()
