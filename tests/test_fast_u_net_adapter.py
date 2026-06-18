from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.maternal_fetal.fast_u_net import FastUNetAdapter


def test_fast_u_net_adapter_plane_and_test_splits(fast_u_net_root: Path):
    entries = list(FastUNetAdapter(fast_u_net_root).iter_entries())

    assert len(entries) == 4
    splits = {e.split for e in entries}
    assert splits == {"train", "val", "test"}

    for entry in entries:
        assert entry.dataset_id == "Fast-U-Net"
        assert entry.anatomy_family == "fetal_abdomen"
        assert entry.modality_type == "image"
        assert Path(entry.image_paths[0]).exists()

    seg_entries = [e for e in entries if e.task_type == "segmentation"]
    assert len(seg_entries) == 4
    assert all(e.has_mask for e in seg_entries)
    assert all(e.instances[0].mask_path for e in seg_entries)

    planes = {e.source_meta["plane"] for e in entries}
    assert planes == {"AC", "test"}
