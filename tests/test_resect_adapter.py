from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.resect import RESECTAdapter


def test_resect_adapter_emits_only_us_resection_stages(resect_root: Path):
    entries = list(RESECTAdapter(resect_root).iter_entries())

    assert len(entries) == 3
    assert {e.source_meta["resection_stage"] for e in entries} == {"before", "during", "after"}
    assert {e.series_id for e in entries} == {
        "Case1-US-before",
        "Case1-US-during",
        "Case1-US-after",
    }

    for entry in entries:
        assert entry.dataset_id == "RESECT"
        assert entry.anatomy_family == "brain"
        assert entry.modality_type == "volume"
        assert entry.task_type == "ssl_only"

