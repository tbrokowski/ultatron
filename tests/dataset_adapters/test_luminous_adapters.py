"""
tests/dataset_adapters/test_luminous_adapters.py  ·  LUMINOUSAdapter contract tests
=====================================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_luminous_adapters.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def luminous_root(tmp_path_factory):
    """
    Synthetic LUMINOUS layout:
      B-mode/
        1_prone_left.tif      ← structured filename (subject 1, prone, left)
        2_prone_right.tif
        3_standing_left.tif
        42.tif                ← flat numeric filename (no position/side)
      Masks/
        1_prone_left.tif
        2_prone_right.tif
        3_standing_left.tif
        42.tif
    """
    root = tmp_path_factory.mktemp("LUMINOUS")
    (root / "B-mode").mkdir()
    (root / "Masks").mkdir()
    stems = ["1_prone_left", "2_prone_right", "3_standing_left", "42"]
    for s in stems:
        (root / "B-mode" / f"{s}.tif").write_bytes(b"\x89PNG")
        (root / "Masks"  / f"{s}.tif").write_bytes(b"\x89PNG")
    return root


@pytest.fixture(scope="module")
def luminous_root_no_mask(tmp_path_factory):
    """LUMINOUS without masks — ssl_only mode."""
    root = tmp_path_factory.mktemp("LUMINOUS_nomask")
    (root / "B-mode").mkdir()
    for i in range(3):
        (root / "B-mode" / f"{i}.tif").write_bytes(b"\x89PNG")
    return root


class TestLUMINOUSAdapter:

    def test_import(self):
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        assert LUMINOUSAdapter.DATASET_ID     == "LUMINOUS"
        assert LUMINOUSAdapter.ANATOMY_FAMILY == "muscle"
        assert LUMINOUSAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "LUMINOUS" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, luminous_root):
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        entries = list(LUMINOUSAdapter(root=luminous_root).iter_entries())
        assert len(entries) == 4

    def test_entry_schema(self, luminous_root):
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in LUMINOUSAdapter(root=luminous_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "LUMINOUS"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.task_type       == "segmentation"
            assert e.probe_type      == "curvilinear"

    def test_all_entries_have_masks(self, luminous_root):
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        for e in LUMINOUSAdapter(root=luminous_root).iter_entries():
            assert e.has_mask is True
            assert len(e.instances) == 1
            assert e.instances[0].mask_path is not None
            assert Path(e.instances[0].mask_path).exists()
            assert e.instances[0].is_promptable is True

    def test_label_raw_and_ontology(self, luminous_root):
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        for e in LUMINOUSAdapter(root=luminous_root).iter_entries():
            assert e.instances[0].label_raw      == "lumbar_multifidus"
            assert e.instances[0].label_ontology == "muscle"

    def test_structured_filename_parsing(self, luminous_root):
        """Position and side must be parsed from structured filenames."""
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        entries = {
            Path(e.image_paths[0]).stem: e
            for e in LUMINOUSAdapter(root=luminous_root).iter_entries()
        }
        e1 = entries["1_prone_left"]
        assert e1.source_meta["position"] == "prone"
        assert e1.source_meta["side"]     == "left"
        assert e1.source_meta["subject_id"] == "1"

        e2 = entries["2_prone_right"]
        assert e2.source_meta["side"] == "right"

        e3 = entries["3_standing_left"]
        assert e3.source_meta["position"] == "standing"

    def test_flat_filename_parsing(self, luminous_root):
        """Flat numeric filenames should still parse subject_id."""
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        entries = {
            Path(e.image_paths[0]).stem: e
            for e in LUMINOUSAdapter(root=luminous_root).iter_entries()
        }
        e42 = entries["42"]
        assert e42.source_meta["subject_id"] == "42"
        assert e42.source_meta["position"]   is None
        assert e42.source_meta["side"]       is None

    def test_ssl_only_without_masks(self, luminous_root_no_mask):
        """Without a Masks/ dir, entries should be ssl_only."""
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        entries = list(LUMINOUSAdapter(root=luminous_root_no_mask).iter_entries())
        assert len(entries) == 3
        for e in entries:
            assert e.has_mask  is False
            assert e.task_type == "ssl_only"

    def test_split_override(self, luminous_root):
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        for e in LUMINOUSAdapter(root=luminous_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, luminous_root):
        from data.adapters.muscle.luminous import LUMINOUSAdapter
        ids = [e.sample_id for e in LUMINOUSAdapter(root=luminous_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, luminous_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "luminous.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("LUMINOUS", luminous_root, writer)
        assert count == 4
        entries = load_manifest(out)
        assert all(e.dataset_id == "LUMINOUS" for e in entries)
