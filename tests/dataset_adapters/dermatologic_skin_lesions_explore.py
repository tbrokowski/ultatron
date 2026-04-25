"""
dermatologic_skin_lesions_explore.py  ·  Real-data exploration for skin lesions
=================================================================================

Usage:
    python -m tests.dataset_adapters.dermatologic_skin_lesions_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.dermatologic_skin_lesions import DermatologicSkinLesionsAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="Dermatologic-US-Skin-Lesions",
        adapter_cls=DermatologicSkinLesionsAdapter,
        default_root=Path(
            "/capstor/store/cscs/swissai/a127/ultrasound/raw/skin/Dermatologic-US-Skin-Lesions"
        ),
        default_out_dir=Path("dataset_exploration_outputs/dermatologic_skin_lesions"),
        root_env_var="US_DERMATOLOGIC_SKIN_LESIONS_ROOT",
        out_dir_env_var="US_DERMATOLOGIC_SKIN_LESIONS_OUT_DIR",
    )


if __name__ == "__main__":
    main()
