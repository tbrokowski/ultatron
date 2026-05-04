"""
annotated_heterogeneous_us_db_explore.py  ·  Exploration for annotated_heterogeneous_us_db
============================================================================================

Usage:
    python -m tests.dataset_adapters.annotated_heterogeneous_us_db_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.annotated_heterogeneous_us_db import AnnotatedHeterogeneousUSDBAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="annotated_heterogeneous_us_db",
        adapter_cls=AnnotatedHeterogeneousUSDBAdapter,
        default_root=Path(
            "/capstor/store/cscs/swissai/a127/ultrasound/raw/multi_organ/annotated_heterogeneous_us_db"
        ),
        default_out_dir=Path("dataset_exploration_outputs/annotated_heterogeneous_us_db"),
        root_env_var="US_ANNOTATED_HETEROGENEOUS_US_DB_ROOT",
        out_dir_env_var="US_ANNOTATED_HETEROGENEOUS_US_DB_OUT_DIR",
    )


if __name__ == "__main__":
    main()
