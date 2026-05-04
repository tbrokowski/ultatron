from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.remind2reg import ReMIND2RegAdapter


def test_remind2reg_adapter_keeps_only_ius_channel(remind2reg_root: Path):
    entries = list(ReMIND2RegAdapter(remind2reg_root).iter_entries())

    assert len(entries) == 1

    entry = entries[0]
    assert entry.dataset_id == "ReMIND2Reg"
    assert entry.anatomy_family == "brain"
    assert entry.modality_type == "volume"
    assert entry.series_id == "ReMIND2Reg_001_0000"
    assert entry.study_id == "ReMIND2Reg_001"
    assert entry.source_meta["channel"] == "us_post_resection"
    assert entry.task_type == "ssl_only"

