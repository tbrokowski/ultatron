"""
tests/dataset_adapters/test_breast.py  ·  BrEaSTAdapter contract tests
=======================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_breast.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def breast_root(tmp_path_factory):
    """
    Synthetic BrEaST layout:
      case001.png + case001_tumor.png
      case002.png + case002_tumor.png  (2 masks)
                  + case002_tumor2.png
      case003.png  (no mask → ssl_only)
    """
    root = tmp_path_factory.mktemp("BrEaST")
    # case001 — 1 mask
    (root / "case001.png").write_bytes(b"\x89PNG")
    (root / "case001_tumor.png").write_bytes(b"\x89PNG")
    # case002 — 2 masks
    (root / "case002.png").write_bytes(b"\x89PNG")
    (root / "case002_tumor.png").write_bytes(b"\x89PNG")
    (root / "case002_tumor2.png").write_bytes(b"\x89PNG")
    # case003 — no mask
    (root / "case003.png").write_bytes(b"\x89PNG")
    return root


class TestBrEaSTAdapter:

    def test_import(self):
        from data.adapters.breast.breast_adapter import BrEaSTAdapter
        assert BrEaSTAdapter.DATASET_ID     == "BrEaST"
        assert BrEaSTAdapter.ANATOMY_FAMILY == "breast"
        assert BrEaSTAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "BrEaST" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, breast_root):
        from data.adapters.breast.breast_adapter import BrEaSTAdapter
        entries = list(BrEaSTAdapter(root=breast_root).iter_entries())
        assert len(entries) == 3  # one entry per image

    def test_entry_schema(self, breast_root):
        from data.adapters.breast.breast_adapter import BrEaSTAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in BrEaSTAdapter(root=breast_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "BrEaST"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}

    def test_case001_has_one_mask(self, breast_root):
        from data.adapters.breast.breast_adapter import BrEaSTAdapter
        entries = {
            e.source_meta["case_id"]: e
            for e in BrEaSTAdapter(root=breast_root).iter_entries()
        }
        e = entries["case001"]
        assert e.has_mask
        assert e.task_type == "segmentation"
        assert len(e.instances) == 1
        assert e.source_meta["num_masks"] == 1

    def test_case002_has_two_masks(self, breast_root):
        from data.adapters.breast.breast_adapter import BrEaSTAdapter
        entries = {
            e.source_meta["case_id"]: e
            for e in BrEaSTAdapter(root=breast_root).iter_entries()
        }
        e = entries["case002"]
        assert e.has_mask
        assert len(e.instances) == 2
        assert e.source_meta["num_masks"] == 2

    def test_case003_no_mask(self, breast_root):
        from data.adapters.breast.breast_adapter import BrEaSTAdapter
        entries = {
            e.source_meta["case_id"]: e
            for e in BrEaSTAdapter(root=breast_root).iter_entries()
        }
        e = entries["case003"]
        assert not e.has_mask
        assert e.task_type == "ssl_only"

    def test_split_override(self, breast_root):
        from data.adapters.breast.breast_adapter import BrEaSTAdapter
        for e in BrEaSTAdapter(root=breast_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, breast_root):
        from data.adapters.breast.breast_adapter import BrEaSTAdapter
        ids = [e.sample_id for e in BrEaSTAdapter(root=breast_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, breast_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "breast.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("BrEaST", breast_root, writer)
        assert count == 3
        entries = load_manifest(out)
        assert all(e.dataset_id == "BrEaST" for e in entries)
