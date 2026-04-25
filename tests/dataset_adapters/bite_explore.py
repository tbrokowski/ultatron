"""
bite_explore.py  ·  Real-data exploration for the BITE adapter
==============================================================

Usage:
    python -m tests.dataset_adapters.bite_explore
"""
from __future__ import annotations

from pathlib import Path

from data.adapters.bite import BITEAdapter

from ._explore_utils import run_adapter_explore


def main() -> None:
    run_adapter_explore(
        dataset_name="BITE",
        adapter_cls=BITEAdapter,
        default_root=Path("/capstor/store/cscs/swissai/a127/ultrasound/raw/brain/BITE"),
        default_out_dir=Path("dataset_exploration_outputs/bite"),
        root_env_var="US_BITE_ROOT",
        out_dir_env_var="US_BITE_OUT_DIR",
    )


if __name__ == "__main__":
    main()
