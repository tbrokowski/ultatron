from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.bite import BITEAdapter


def test_bite_adapter_keeps_only_ultrasound_content(bite_root: Path):
    entries = list(BITEAdapter(bite_root).iter_entries())

    assert len(entries) == 5

    image_entries = [e for e in entries if e.modality_type == "image"]
    volume_entries = [e for e in entries if e.modality_type == "volume"]

    assert len(image_entries) == 3
    assert len(volume_entries) == 2
    assert all(e.anatomy_family == "brain" for e in entries)
    assert all(e.task_type == "ssl_only" for e in entries)
    assert all("MRT1" not in e.image_paths[0] for e in entries)
    assert {e.study_id for e in entries} == {"group1:subject01", "group2:subject02"}

