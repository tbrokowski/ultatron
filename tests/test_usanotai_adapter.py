from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.multi_organ.usanotai import USAnotAIAdapter


def test_usanotai_adapter_maps_organ_classes(usanotai_root: Path):
    entries = list(USAnotAIAdapter(usanotai_root).iter_entries())

    assert len(entries) == 4
    organs = {e.instances[0].label_raw for e in entries}
    assert organs == {"bladder", "liver"}

    for entry in entries:
        assert entry.dataset_id == "USAnotAI-master"
        assert entry.anatomy_family == "multi"
        assert entry.task_type == "multiclass_cls"
        assert entry.has_mask is False
        assert "class_id" in entry.source_meta
        assert Path(entry.image_paths[0]).exists()

    train_entries = [e for e in entries if e.split == "train"]
    test_entries = [e for e in entries if e.split == "test"]
    assert len(train_entries) == 2
    assert len(test_entries) == 2
