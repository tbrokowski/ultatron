"""
tests/dataset_adapters/test_tus_rec_val_adapters.py
====================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_tus_rec_val_adapters.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def tus_rec_val_root(tmp_path_factory):
    """
    Synthetic TUS-REC-Val layout (separate frames/ and transfs/):
      frames/
        subject_001/  RH_rotating.h5, LH_rotating.h5
        subject_002/  RH_rotating.h5
      transfs/
        subject_001/  RH_rotating.h5, LH_rotating.h5
        subject_002/  RH_rotating.h5
      landmarks/
        subject_001.h5
      calib_matrix.csv
    """
    root = tmp_path_factory.mktemp("TUS_REC_Val")

    for subj, scans in [
        ("subject_001", ["RH_rotating.h5", "LH_rotating.h5"]),
        ("subject_002", ["RH_rotating.h5"]),
    ]:
        (root / "frames"  / subj).mkdir(parents=True)
        (root / "transfs" / subj).mkdir(parents=True)
        for fname in scans:
            (root / "frames"  / subj / fname).write_bytes(b"\x00" * 64)
            (root / "transfs" / subj / fname).write_bytes(b"\x00" * 64)

    (root / "landmarks").mkdir()
    (root / "landmarks" / "subject_001.h5").write_bytes(b"\x00" * 64)
    (root / "calib_matrix.csv").write_text("scaling,spatial\n0.15,identity\n")
    return root


@pytest.fixture(scope="module")
def tus_rec_val_root_zenodo(tmp_path_factory):
    """Zenodo validation layout with landmark/ (singular) dir."""
    root = tmp_path_factory.mktemp("TUS_REC_Val_zenodo")
    (root / "frames" / "050").mkdir(parents=True)
    (root / "transfs" / "050").mkdir(parents=True)
    (root / "landmark").mkdir()
    scan = "LH_Par_C_DtP.h5"
    (root / "frames" / "050" / scan).write_bytes(b"\x00" * 64)
    (root / "transfs" / "050" / scan).write_bytes(b"\x00" * 64)
    (root / "landmark" / "landmark_050.h5").write_bytes(b"\x00" * 64)
    (root / "calib_matrix.csv").write_text("scaling,spatial\n0.15,identity\n")
    return root


class TestTUSRECValAdapter:

    def test_import(self):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        assert TUSRECValAdapter.DATASET_ID     == "TUS-REC-Val"
        assert TUSRECValAdapter.ANATOMY_FAMILY == "muscle"
        assert TUSRECValAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "TUS-REC-Val" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, tus_rec_val_root):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        entries = list(TUSRECValAdapter(root=tus_rec_val_root).iter_entries())
        assert len(entries) == 3  # 2 + 1

    def test_entry_schema(self, tus_rec_val_root):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in TUSRECValAdapter(root=tus_rec_val_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "TUS-REC-Val"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "video"
            assert e.ssl_stream      == "video"
            assert e.task_type       == "reconstruction_3d"
            assert e.probe_type      == "curvilinear"
            assert e.has_mask        is False
            assert e.has_temporal_order is True
            assert e.num_frames      >= 1

    def test_default_split_is_val(self, tus_rec_val_root):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        for e in TUSRECValAdapter(root=tus_rec_val_root).iter_entries():
            assert e.split == "val"

    def test_split_override(self, tus_rec_val_root):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        for e in TUSRECValAdapter(root=tus_rec_val_root, split_override="test").iter_entries():
            assert e.split == "test"

    def test_tforms_path_in_meta(self, tus_rec_val_root):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        for e in TUSRECValAdapter(root=tus_rec_val_root).iter_entries():
            assert "tforms_path"  in e.source_meta
            assert "has_tforms"   in e.source_meta
            assert e.source_meta["has_tforms"] is True
            assert Path(e.source_meta["tforms_path"]).exists()

    def test_landmark_flag(self, tus_rec_val_root):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        entries = {
            e.source_meta["subject_id"]: e
            for e in TUSRECValAdapter(root=tus_rec_val_root).iter_entries()
        }
        assert entries["subject_001"].source_meta["has_landmarks"] is True
        assert entries["subject_002"].source_meta["has_landmarks"] is False

    def test_side_and_motion_parsed(self, tus_rec_val_root):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        entries = list(TUSRECValAdapter(root=tus_rec_val_root).iter_entries())
        sides   = {e.source_meta["side"]   for e in entries}
        motions = {e.source_meta["motion"] for e in entries}
        assert "right"    in sides
        assert "left"     in sides
        assert "rotating" in motions

    def test_sample_ids_unique(self, tus_rec_val_root):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        ids = [e.sample_id for e in TUSRECValAdapter(root=tus_rec_val_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_zenodo_landmark_dir(self, tus_rec_val_root_zenodo):
        from data.adapters.muscle.tus_rec_val import TUSRECValAdapter
        entries = list(TUSRECValAdapter(root=tus_rec_val_root_zenodo).iter_entries())
        assert len(entries) == 1
        e = entries[0]
        assert e.source_meta["has_landmarks"] is True
        assert e.source_meta["landmark_path"].endswith("landmark_050.h5")
        assert e.source_meta["side"] == "left"
        assert e.source_meta["motion"] == "Par_C_DtP"

    def test_build_manifest_for_dataset(self, tus_rec_val_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "tus_rec_val.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("TUS-REC-Val", tus_rec_val_root, writer)
        assert count == 3
        entries = load_manifest(out)
        assert all(e.dataset_id == "TUS-REC-Val" for e in entries)
