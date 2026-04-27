"""
tests/dataset_adapters/test_bus_bra.py  ·  BUSBRAAdapter contract tests
========================================================================

Uses the bus_bra_root fixture from tests/conftest.py.

Run with:
    pytest tests/dataset_adapters/test_bus_bra.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


class TestBUSBRAAdapter:

    def test_import(self):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        assert BUSBRAAdapter.DATASET_ID     == "BUS-BRA"
        assert BUSBRAAdapter.ANATOMY_FAMILY == "breast"
        assert BUSBRAAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "BUS-BRA" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, bus_bra_root):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        entries = list(BUSBRAAdapter(root=bus_bra_root).iter_entries())
        assert len(entries) == 8

    def test_entry_schema(self, bus_bra_root):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in BUSBRAAdapter(root=bus_bra_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "BUS-BRA"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert len(e.image_paths) == 1
            assert e.curriculum_tier  in {1, 2, 3}

    def test_has_mask(self, bus_bra_root):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        entries = list(BUSBRAAdapter(root=bus_bra_root).iter_entries())
        assert all(e.has_mask for e in entries)
        assert all(e.instances[0].mask_path is not None for e in entries)

    def test_label_ontology(self, bus_bra_root):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        ontologies = {
            e.instances[0].label_ontology
            for e in BUSBRAAdapter(root=bus_bra_root).iter_entries()
        }
        assert "breast_lesion_benign"    in ontologies
        assert "breast_lesion_malignant" in ontologies

    def test_task_type(self, bus_bra_root):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        for e in BUSBRAAdapter(root=bus_bra_root).iter_entries():
            assert e.task_type == "segmentation"

    def test_split_override(self, bus_bra_root):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        for e in BUSBRAAdapter(root=bus_bra_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, bus_bra_root):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        ids = [e.sample_id for e in BUSBRAAdapter(root=bus_bra_root).iter_entries()]
        assert len(ids) == len(set(ids)), "sample_ids must be unique"

    def test_build_manifest(self, bus_bra_root):
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        assert len(BUSBRAAdapter(root=bus_bra_root).build_manifest()) == 8

    def test_missing_csv_graceful(self, tmp_path):
        """Adapter must not crash if annotations.csv is absent (ssl_only mode)."""
        from data.adapters.breast.bus_bra_adapter import BUSBRAAdapter
        (tmp_path / "images").mkdir()
        (tmp_path / "masks").mkdir()
        (tmp_path / "images" / "img_9999.png").write_bytes(b"\x89PNG")
        entries = list(BUSBRAAdapter(root=tmp_path).iter_entries())
        assert len(entries) == 1
        assert entries[0].task_type == "ssl_only"

    def test_build_manifest_for_dataset(self, bus_bra_root, tmp_path):
        """build_manifest_for_dataset() helper should work end-to-end."""
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "bus_bra.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("BUS-BRA", bus_bra_root, writer)
        assert count == 8
        entries = load_manifest(out)
        assert len(entries) == count
        assert all(e.dataset_id == "BUS-BRA" for e in entries)
