"""
tests/dataset_adapters/test_midi_b.py  ·  MidiBAdapter contract tests
"""
from __future__ import annotations

import zipfile

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def midi_b_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("midi-b")
    dicoms = root / "dicoms"
    dicoms.mkdir()

    for series_uid in ("series-a", "series-b"):
        series_dir = dicoms / series_uid
        series_dir.mkdir()
        (series_dir / "00000001.dcm").write_bytes(b"\x00" * 128)

    (root / "series_uids.txt").write_text("series-a\nseries-b\n")
    return root


@pytest.fixture(scope="module")
def midi_b_zip_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("midi-b-zip")
    dicoms = root / "dicoms"
    dicoms.mkdir()

    series_uid = "series-zip"
    zip_path = dicoms / f"{series_uid}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("00000001.dcm", b"\x00" * 128)

    (root / "series_uids.txt").write_text(f"{series_uid}\n")
    return root


class TestMidiBAdapter:

    def test_import(self):
        from data.adapters.breast.midi_b import MidiBAdapter
        assert MidiBAdapter.DATASET_ID     == "midi-b"
        assert MidiBAdapter.ANATOMY_FAMILY == "breast"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "midi-b" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, midi_b_root):
        from data.adapters.breast.midi_b import MidiBAdapter
        entries = list(MidiBAdapter(root=midi_b_root).iter_entries())
        assert len(entries) == 2

    def test_entry_schema(self, midi_b_root):
        from data.adapters.breast.midi_b import MidiBAdapter
        from data.schema.manifest import USManifestEntry
        for e in MidiBAdapter(root=midi_b_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id == "midi-b"
            assert e.modality_type == "image"
            assert e.task_type == "ssl_only"
            assert e.image_paths[0].endswith(".dcm")

    def test_extracts_zip_on_demand(self, midi_b_zip_root):
        from data.adapters.breast.midi_b import MidiBAdapter
        entries = list(MidiBAdapter(root=midi_b_zip_root).iter_entries())
        assert len(entries) == 1
        assert Path(entries[0].image_paths[0]).exists()

    def test_build_manifest_for_dataset(self, midi_b_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "midi-b.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("midi-b", midi_b_root, writer)
        assert count == 2
        entries = load_manifest(out)
        assert all(e.dataset_id == "midi-b" for e in entries)


@pytest.mark.skipif(
    not Path(
        "/capstor/store/cscs/swissai/a127/ultrasound/raw/breast/midi-b/dicoms"
    ).is_dir(),
    reason="midi-b not mounted on capstor",
)
def test_midi_b_real_capstor_data():
    from data.adapters.breast.midi_b import MidiBAdapter
    root = "/capstor/store/cscs/swissai/a127/ultrasound/raw/breast/midi-b"
    entries = list(MidiBAdapter(root=root).iter_entries())
    assert len(entries) == 269
    assert len({e.series_id for e in entries}) == 61
    assert all(e.image_paths[0].endswith(".dcm") for e in entries)
    assert all(Path(e.image_paths[0]).exists() for e in entries)
