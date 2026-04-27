"""
tests/dataset_adapters/test_gdph_sysucc.py  ·  GDPHSYSUCCAdapter contract tests
=================================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_gdph_sysucc.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def gdph_sysucc_root(tmp_path_factory):
    """
    Synthetic layout:
      GDPH/   benign(0..3).png  malignant(0..1).png
      SYSUCC/ benign(0..2).png  malignant(0).png
    """
    root = tmp_path_factory.mktemp("GDPH_SYSUCC")

    for sub, n_benign, n_malignant in [("GDPH", 4, 2), ("SYSUCC", 3, 1)]:
        d = root / sub
        d.mkdir()
        for i in range(n_benign):
            (d / f"benign({i}).png").write_bytes(b"\x89PNG")
        for i in range(n_malignant):
            (d / f"malignant({i}).png").write_bytes(b"\x89PNG")

    return root


class TestGDPHSYSUCCAdapter:

    def test_import(self):
        from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
        assert GDPHSYSUCCAdapter.DATASET_ID     == "GDPH-SYSUCC"
        assert GDPHSYSUCCAdapter.ANATOMY_FAMILY == "breast"
        assert GDPHSYSUCCAdapter.SONODQS        == "bronze"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "GDPH-SYSUCC" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, gdph_sysucc_root):
        from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
        entries = list(GDPHSYSUCCAdapter(root=gdph_sysucc_root).iter_entries())
        assert len(entries) == 10  # 4+2 GDPH + 3+1 SYSUCC

    def test_entry_schema(self, gdph_sysucc_root):
        from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in GDPHSYSUCCAdapter(root=gdph_sysucc_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "GDPH-SYSUCC"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}

    def test_label_ontology(self, gdph_sysucc_root):
        from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
        ontologies = {
            e.instances[0].label_ontology
            for e in GDPHSYSUCCAdapter(root=gdph_sysucc_root).iter_entries()
        }
        assert "breast_lesion_benign"    in ontologies
        assert "breast_lesion_malignant" in ontologies

    def test_no_mask(self, gdph_sysucc_root):
        from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
        for e in GDPHSYSUCCAdapter(root=gdph_sysucc_root).iter_entries():
            assert not e.has_mask
            assert e.task_type == "classification"

    def test_sub_dataset_in_source_meta(self, gdph_sysucc_root):
        from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
        subs = {
            e.source_meta["sub_dataset"]
            for e in GDPHSYSUCCAdapter(root=gdph_sysucc_root).iter_entries()
        }
        assert "GDPH"   in subs
        assert "SYSUCC" in subs

    def test_sample_ids_unique(self, gdph_sysucc_root):
        from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
        ids = [e.sample_id for e in GDPHSYSUCCAdapter(root=gdph_sysucc_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_split_override(self, gdph_sysucc_root):
        from data.adapters.breast.gdph_sysucc_adapter import GDPHSYSUCCAdapter
        for e in GDPHSYSUCCAdapter(root=gdph_sysucc_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_build_manifest_for_dataset(self, gdph_sysucc_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "gdph_sysucc.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("GDPH-SYSUCC", gdph_sysucc_root, writer)
        assert count == 10
        entries = load_manifest(out)
        assert all(e.dataset_id == "GDPH-SYSUCC" for e in entries)
