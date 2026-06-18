from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.brain.brain_3d_us_neuroimages import ThreeDUSNeuroimagesAdapter


def test_brain_3d_us_neuroimages_adapter_maps_unlabeled_brain_volumes(
    brain_3d_us_neuroimages_root: Path,
):
    entries = list(ThreeDUSNeuroimagesAdapter(brain_3d_us_neuroimages_root).iter_entries())

    assert len(entries) == 3
    assert {e.study_id for e in entries} == {"CaseA", "CaseB"}
    assert len({e.split for e in entries if e.study_id == "CaseA"}) == 1

    for entry in entries:
        assert entry.dataset_id == "3D-US-Neuroimages-Dataset"
        assert entry.anatomy_family == "brain"
        assert entry.modality_type == "volume"
        assert entry.is_3d is True
        assert entry.is_cine is True
        assert entry.has_temporal_order is True
        assert entry.task_type == "ssl_only"
        assert entry.ssl_stream == "both"
        assert entry.is_promptable is False
        assert entry.instances == []
        assert entry.series_id
        assert entry.source_meta["scan_id"] == entry.series_id


def test_brain_3d_us_neuroimages_adapter_parses_real_scan_names(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    for name in (
        "1_2013_11_08.nrrd",
        "1_2013_11_15.nrrd",
        "4_2013_06_26.nrrd",
    ):
        content = (
            "NRRD0003\n"
            "type: unsigned char\n"
            "dimension: 3\n"
            "sizes: 251 187 217\n"
            "encoding: raw\n"
            "\n"
        ).encode("latin-1")
        (root / name).write_bytes(content + b"\x00" * 16)

    entries = list(ThreeDUSNeuroimagesAdapter(root).iter_entries())

    assert len(entries) == 3
    assert {e.study_id for e in entries} == {"1", "4"}
    assert {e.series_id for e in entries} == {
        "1_2013_11_08",
        "1_2013_11_15",
        "4_2013_06_26",
    }
    assert all(e.num_frames == 251 for e in entries)
    assert all(e.ssl_stream == "both" for e in entries)
    assert entries[0].source_meta["scan_date"] == "2013-11-08"
    assert entries[0].source_meta["patient_id"] == "1"
    assert entries[0].source_meta["num_z_slices"] == 251
