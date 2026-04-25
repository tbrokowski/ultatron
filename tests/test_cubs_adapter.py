from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.cubs import CUBSAdapter


def test_cubs_adapter_maps_measurement_sample(cubs_root: Path):
    entries = list(CUBSAdapter(cubs_root).iter_entries())
    assert len(entries) == 1

    entry = entries[0]
    assert entry.task_type == "measurement"
    assert entry.dataset_id == "CUBS"
    assert entry.has_mask is False
    assert len(entry.instances) == 1
    assert entry.instances[0].measurement_mm == pytest.approx(0.63694, rel=1e-4)
    assert entry.instances[0].bbox_xyxy == [5.0, 5.0, 45.0, 25.0]
    assert entry.source_meta["clinical"]["age"] == 61
    assert entry.source_meta["cf_mm_per_px"] == pytest.approx(0.063694)
