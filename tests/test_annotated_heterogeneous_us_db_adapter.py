from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.annotated_heterogeneous_us_db import AnnotatedHeterogeneousUSDBAdapter


def test_annotated_heterogeneous_us_db_adapter_preserves_weak_labels_and_noise_filter(
    annotated_heterogeneous_us_db_root: Path,
):
    entries = list(
        AnnotatedHeterogeneousUSDBAdapter(
            annotated_heterogeneous_us_db_root
        ).iter_entries()
    )

    assert len(entries) == 3

    by_case = {entry.study_id: entry for entry in entries}

    labeled_video = by_case["10"]
    assert labeled_video.modality_type == "video"
    assert labeled_video.task_type == "weak_label"
    assert len(labeled_video.instances) == 2
    assert labeled_video.source_meta["pathology_vector"][1] == 1.0
    assert labeled_video.source_meta["pathology_vector"][11] == 1.0

    photo_sequence = by_case["11"]
    assert photo_sequence.modality_type == "pseudo_video"
    assert photo_sequence.num_frames == 2
    assert photo_sequence.source_meta["kept_frames"] == 2
    assert [Path(p).name for p in photo_sequence.image_paths] == ["01.jpg", "03.jpg"]

    unlabeled_video = by_case["12"]
    assert unlabeled_video.task_type == "ssl_only"
    assert unlabeled_video.instances == []

