from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.adapters.lung.lung_database import LungDatabaseAdapter


def test_lung_database_adapter_emits_frames_and_zone_clips(lung_database_root: Path):
    entries = list(LungDatabaseAdapter(lung_database_root).iter_entries())

    image_entries = [e for e in entries if e.modality_type == "image"]
    clip_entries = [e for e in entries if e.modality_type == "pseudo_video"]

    assert len(image_entries) == 10
    assert len(clip_entries) == 4

    by_series = {e.series_id: e for e in clip_entries}
    z01 = by_series["Pt01_z01"]
    assert z01.ssl_stream == "both"
    assert z01.num_frames == 2
    assert z01.is_cine
    assert z01.has_temporal_order
    assert z01.view_type == "z01"
    assert [Path(p).name for p in z01.image_paths] == [
        "image_001_Pt01_z01_frame_000000.jpg",
        "image_002_Pt01_z01_frame_000001.jpg",
    ]

    z02 = by_series["Pt01_z02"]
    assert z02.num_frames == 3

    ed_clip = by_series["ED1_z03"]
    assert ed_clip.study_id == "ED1"
    assert ed_clip.num_frames == 3

    for e in image_entries:
        assert e.ssl_stream == "image"
        assert e.anatomy_family == "lung"
        assert e.source_meta["video_source"] == "extracted_frames"
        assert "lung_zone" in e.source_meta


@pytest.mark.skipif(
    not Path("/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/Lung Database").exists(),
    reason="Lung Database not mounted",
)
def test_lung_database_real_data_has_pseudo_videos():
    root = "/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/Lung Database"
    entries = list(LungDatabaseAdapter(root).iter_entries())

    images = [e for e in entries if e.modality_type == "image"]
    clips = [e for e in entries if e.modality_type == "pseudo_video"]

    assert len(images) > 200_000
    assert len(clips) > 1_000
    assert all(e.ssl_stream == "both" for e in clips)
    assert all(e.num_frames >= 2 for e in clips)
