from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.common_carotid import CommonCarotidArteryImagesAdapter


def test_common_carotid_adapter_maps_segmentation_sample(common_carotid_root: Path):
    entries = list(CommonCarotidArteryImagesAdapter(common_carotid_root).iter_entries())
    assert len(entries) == 1

    entry = entries[0]
    assert entry.task_type == "segmentation"
    assert entry.dataset_id == "Common-Carotid-Artery-Ultrasound-Images"
    assert entry.has_mask is True
    assert entry.is_promptable is True
    assert len(entry.instances) == 1
    assert entry.instances[0].mask_path is not None
    assert Path(entry.instances[0].mask_path).exists()
