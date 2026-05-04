from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.erdes import ERDESAdapter


def test_erdes_adapter_uses_official_split_files(erdes_root: Path):
    entries = list(ERDESAdapter(erdes_root).iter_entries())

    assert len(entries) == 3
    assert {e.split for e in entries} == {"train", "val", "test"}

    by_study = {e.study_id: e for e in entries}
    positive = by_study["101"]
    negative = by_study["202"]

    assert positive.instances[0].classification_label == 1
    assert positive.instances[0].label_ontology == "retinal_detachment"
    assert negative.instances[0].classification_label == 0
    assert negative.instances[0].label_ontology == "retina"

    for entry in entries:
        assert entry.dataset_id == "ERDES"
        assert entry.anatomy_family == "ocular"
        assert entry.modality_type == "video"
        assert entry.task_type == "classification"
        assert entry.has_temporal_order is True

