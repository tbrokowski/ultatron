"""
tests/dataset_adapters/test_buid.py  ·  BUIDAdapter contract tests
===================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_buid.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def buid_root(tmp_path_factory):
    """
    Synthetic BUID layout:
      Benign/    4 cases  (Image.bmp + Mask.tif + Lesion.bmp)
      Malignant/ 3 cases
    """
    root = tmp_path_factory.mktemp("BUID")

    for cls in ("Benign", "Malignant"):
        d = root / cls
        d.mkdir()
        n = 4 if cls == "Benign" else 3
        for i in range(1, n + 1):
            (d / f"{i} {cls} Image.bmp").write_bytes(b"BM")
            (d / f"{i} {cls} Lesion.bmp").write_bytes(b"BM")
            (d / f"{i} {cls} Mask.tif").write_bytes(b"II")  # minimal TIFF header

    return root


class TestBUIDAdapter:

    def test_import(self):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        assert BUIDAdapter.DATASET_ID     == "BUID"
        assert BUIDAdapter.ANATOMY_FAMILY == "breast"
        assert BUIDAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "BUID" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, buid_root):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        entries = list(BUIDAdapter(root=buid_root).iter_entries())
        assert len(entries) == 7  # 4 benign + 3 malignant

    def test_lesion_bmp_ignored(self, buid_root):
        """Lesion.bmp files must not appear as image_paths."""
        from data.adapters.breast.buid_adapter import BUIDAdapter
        for e in BUIDAdapter(root=buid_root).iter_entries():
            assert "Lesion" not in e.image_paths[0]

    def test_entry_schema(self, buid_root):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in BUIDAdapter(root=buid_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "BUID"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}

    def test_has_mask(self, buid_root):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        entries = list(BUIDAdapter(root=buid_root).iter_entries())
        assert all(e.has_mask for e in entries)
        assert all(e.instances[0].mask_path is not None for e in entries)

    def test_mask_is_tif(self, buid_root):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        for e in BUIDAdapter(root=buid_root).iter_entries():
            assert e.instances[0].mask_path.endswith(".tif")

    def test_label_ontology(self, buid_root):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        ontologies = {
            e.instances[0].label_ontology
            for e in BUIDAdapter(root=buid_root).iter_entries()
        }
        assert "breast_lesion_benign"    in ontologies
        assert "breast_lesion_malignant" in ontologies

    def test_task_type(self, buid_root):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        for e in BUIDAdapter(root=buid_root).iter_entries():
            assert e.task_type == "segmentation"

    def test_split_override(self, buid_root):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        for e in BUIDAdapter(root=buid_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, buid_root):
        from data.adapters.breast.buid_adapter import BUIDAdapter
        ids = [e.sample_id for e in BUIDAdapter(root=buid_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, buid_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "buid.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("BUID", buid_root, writer)
        assert count == 7
        entries = load_manifest(out)
        assert all(e.dataset_id == "BUID" for e in entries)
