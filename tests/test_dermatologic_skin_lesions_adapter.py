from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.dermatologic_skin_lesions import DermatologicSkinLesionsAdapter


def test_dermatologic_skin_lesions_adapter_handles_paired_and_fallback_images(
    dermatologic_skin_lesions_root: Path,
):
    entries = list(
        DermatologicSkinLesionsAdapter(
            dermatologic_skin_lesions_root
        ).iter_entries()
    )

    assert len(entries) == 3
    by_case = {entry.study_id: entry for entry in entries}

    benign_pair = by_case["01"]
    assert benign_pair.modality_type == "pseudo_video"
    assert benign_pair.num_frames == 2
    assert benign_pair.instances[0].label_ontology == "skin_lesion_benign"

    typo_resolved = by_case["02"]
    assert typo_resolved.modality_type == "pseudo_video"
    assert typo_resolved.image_paths[1].endswith("02_doppler.jpg")
    assert typo_resolved.instances[0].classification_label == 1
    assert typo_resolved.source_meta["diagnosis"] == "leiomyoma"

    bw_only = by_case["03"]
    assert bw_only.modality_type == "image"
    assert bw_only.num_frames == 1
    assert bw_only.source_meta["has_doppler"] is False
    assert bw_only.source_meta["diagnosis"] == "venous_malformation"

