"""
tests/dataset_adapters/test_tus_rec_adapters.py  ·  TUSRECAdapter contract tests
==================================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_tus_rec_adapters.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def tus_rec_root(tmp_path_factory):
    """
    Synthetic TUS-REC layout:
      frames_transfs/
        subject_001/
          RH_rotating.h5
          LH_rotating.h5
        subject_002/
          RH_fanning.h5
          LH_fanning.h5
      landmarks/
        subject_001.h5
      calib_matrix.csv
    """
    root = tmp_path_factory.mktemp("TUS_REC")

    (root / "frames_transfs" / "subject_001").mkdir(parents=True)
    (root / "frames_transfs" / "subject_002").mkdir(parents=True)
    (root / "landmarks").mkdir()

    # Stub .h5 files (not real HDF5 — adapter handles missing h5py gracefully)
    for subj, scans in [
        ("subject_001", ["RH_rotating.h5", "LH_rotating.h5"]),
        ("subject_002", ["RH_fanning.h5",  "LH_fanning.h5"]),
    ]:
        for fname in scans:
            (root / "frames_transfs" / subj / fname).write_bytes(b"\x00" * 64)

    # Landmark file only for subject_001
    (root / "landmarks" / "subject_001.h5").write_bytes(b"\x00" * 64)

    # Calibration CSV
    (root / "calib_matrix.csv").write_text(
        "scaling_from_pixel_to_mm,spatial_calibration\n0.15,identity\n"
    )
    return root


@pytest.fixture(scope="module")
def tus_rec_root_flat(tmp_path_factory):
    """Flat layout: .h5 files directly at root."""
    root = tmp_path_factory.mktemp("TUS_REC_flat")
    for fname in ["RH_rotating.h5", "LH_rotating.h5"]:
        (root / fname).write_bytes(b"\x00" * 64)
    return root


@pytest.fixture(scope="module")
def tus_rec_root_zenodo(tmp_path_factory):
    """
    Zenodo download layout (matches capstor staging):
      calib_matrix.csv
      train_part1/
        000/
          LH_Par_C_DtP.h5
          RH_rotating.h5
        001/
          LH_fanning.h5
    """
    root = tmp_path_factory.mktemp("TUS_REC_zenodo")
    (root / "train_part1" / "000").mkdir(parents=True)
    (root / "train_part1" / "001").mkdir(parents=True)
    for subj, scans in [
        ("000", ["LH_Par_C_DtP.h5", "RH_rotating.h5"]),
        ("001", ["LH_fanning.h5"]),
    ]:
        for fname in scans:
            (root / "train_part1" / subj / fname).write_bytes(b"\x00" * 64)
    (root / "calib_matrix.csv").write_text(
        "scaling_from_pixel_to_mm,spatial_calibration\n0.15,identity\n"
    )
    return root


class TestTUSRECAdapter:

    def test_import(self):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        assert TUSRECAdapter.DATASET_ID     == "TUS-REC"
        assert TUSRECAdapter.ANATOMY_FAMILY == "muscle"
        assert TUSRECAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "TUS-REC" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, tus_rec_root):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        entries = list(TUSRECAdapter(root=tus_rec_root).iter_entries())
        assert len(entries) == 4  # 2 subjects × 2 scans

    def test_entry_schema(self, tus_rec_root):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in TUSRECAdapter(root=tus_rec_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "TUS-REC"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "video"
            assert e.ssl_stream      == "video"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.has_mask        is False
            assert e.task_type       == "reconstruction_3d"
            assert e.probe_type      == "curvilinear"
            assert e.has_temporal_order is True

    def test_h5_path_in_image_paths(self, tus_rec_root):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        for e in TUSRECAdapter(root=tus_rec_root).iter_entries():
            assert len(e.image_paths) == 1
            assert e.image_paths[0].endswith(".h5")
            assert Path(e.image_paths[0]).exists()

    def test_side_and_motion_parsed(self, tus_rec_root):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        entries = list(TUSRECAdapter(root=tus_rec_root).iter_entries())
        sides   = {e.source_meta["side"]   for e in entries}
        motions = {e.source_meta["motion"] for e in entries}
        assert "right"    in sides
        assert "left"     in sides
        assert "rotating" in motions or "fanning" in motions

    def test_subject_id_in_meta(self, tus_rec_root):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        entries = list(TUSRECAdapter(root=tus_rec_root).iter_entries())
        subjects = {e.source_meta["subject_id"] for e in entries}
        assert "subject_001" in subjects
        assert "subject_002" in subjects

    def test_landmark_flag(self, tus_rec_root):
        """subject_001 has landmark file, subject_002 does not."""
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        entries = list(TUSRECAdapter(root=tus_rec_root).iter_entries())
        by_subj = {e.source_meta["subject_id"]: e for e in entries}
        assert by_subj["subject_001"].source_meta["has_landmarks"] is True
        assert by_subj["subject_002"].source_meta["has_landmarks"] is False

    def test_split_override(self, tus_rec_root):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        for e in TUSRECAdapter(root=tus_rec_root, split_override="train").iter_entries():
            assert e.split == "train"

    def test_sample_ids_unique(self, tus_rec_root):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        ids = [e.sample_id for e in TUSRECAdapter(root=tus_rec_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_flat_layout(self, tus_rec_root_flat):
        """Flat layout with h5 at root should still yield entries."""
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        entries = list(TUSRECAdapter(root=tus_rec_root_flat).iter_entries())
        assert len(entries) == 2

    def test_zenodo_train_part1_layout(self, tus_rec_root_zenodo):
        """Zenodo/capstor layout: train_part1/ under dataset root."""
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        entries = list(TUSRECAdapter(root=tus_rec_root_zenodo).iter_entries())
        assert len(entries) == 3
        assert all("train_part1" in e.image_paths[0] for e in entries)
        assert entries[0].source_meta["calib_csv"].endswith("calib_matrix.csv")
        assert "train_part1" not in entries[0].source_meta["calib_csv"]

    def test_zenodo_train_part1_and_part2_layout(self, tmp_path):
        """Both train_part1/ and train_part2/ are scanned for full 2024 data."""
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        root = tmp_path / "TUS_REC_parts"
        for part, subj, scans in [
            ("train_part1", "000", ["LH_Par_C_DtP.h5", "RH_Per_L_DtP.h5"]),
            ("train_part2", "025", ["LH_Par_S_PtD.h5"]),
        ]:
            d = root / part / subj
            d.mkdir(parents=True)
            for fname in scans:
                (d / fname).write_bytes(b"\x00" * 64)
        (root / "calib_matrix.csv").write_text("scaling_from_pixel_to_mm\n0.15\n")
        entries = list(TUSRECAdapter(root=root).iter_entries())
        assert len(entries) == 3
        subjects = {e.source_meta["subject_id"] for e in entries}
        assert subjects == {"000", "025"}

    def test_zenodo_filename_parsing(self, tus_rec_root_zenodo):
        from data.adapters.muscle.tus_rec import TUSRECAdapter
        entries = list(TUSRECAdapter(root=tus_rec_root_zenodo).iter_entries())
        zenodo = next(e for e in entries if "LH_Par_C_DtP" in e.image_paths[0])
        assert zenodo.source_meta["side"] == "left"
        assert zenodo.source_meta["motion"] == "Par_C_DtP"
        assert zenodo.source_meta["scan_name"] == "LH_Par_C_DtP"

    def test_build_manifest_for_dataset(self, tus_rec_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "tus_rec.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("TUS-REC", tus_rec_root, writer)
        assert count == 4
        entries = load_manifest(out)
        assert all(e.dataset_id == "TUS-REC" for e in entries)
