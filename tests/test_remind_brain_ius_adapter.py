from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.remind_brain_ius import REMINDBrainIUSAdapter


def test_remind_brain_ius_adapter_filters_to_us_dicom_series(remind_brain_ius_root: Path):
    entries = list(REMINDBrainIUSAdapter(remind_brain_ius_root).iter_entries())

    assert len(entries) == 2

    for entry in entries:
        assert entry.dataset_id == "REMIND-Brain-iUS"
        assert entry.anatomy_family == "brain"
        assert entry.modality_type == "volume"
        assert entry.is_3d is True
        assert entry.task_type == "ssl_only"
        assert entry.series_id.startswith("US_")
        assert "/US_" in entry.image_paths[0]

