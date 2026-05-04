from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.brain_3d_us_neuroimages import ThreeDUSNeuroimagesAdapter


def test_brain_3d_us_neuroimages_adapter_maps_unlabeled_brain_volumes(
    brain_3d_us_neuroimages_root: Path,
):
    entries = list(ThreeDUSNeuroimagesAdapter(brain_3d_us_neuroimages_root).iter_entries())

    assert len(entries) == 3
    assert {e.study_id for e in entries} == {"CaseA", "CaseB"}
    assert len({e.split for e in entries if e.study_id == "CaseA"}) == 1

    for entry in entries:
        assert entry.dataset_id == "3D-US-Neuroimages-Dataset"
        assert entry.anatomy_family == "brain"
        assert entry.modality_type == "volume"
        assert entry.is_3d is True
        assert entry.task_type == "ssl_only"
        assert entry.ssl_stream == "image"
        assert entry.is_promptable is False

