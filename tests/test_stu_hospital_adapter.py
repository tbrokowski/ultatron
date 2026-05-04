from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.stu_hospital import STUHospitalAdapter


def test_stu_hospital_adapter_maps_png_mask_pairs(stu_hospital_root: Path):
    entries = list(STUHospitalAdapter(stu_hospital_root).iter_entries())

    assert len(entries) == 2

    for entry in entries:
        assert entry.dataset_id == "STU-Hospital-master"
        assert entry.anatomy_family == "multi"
        assert entry.task_type == "segmentation"
        assert entry.has_mask is True
        assert entry.is_promptable is True
        assert len(entry.instances) == 1
        assert entry.instances[0].mask_path is not None
        assert Path(entry.instances[0].mask_path).exists()

