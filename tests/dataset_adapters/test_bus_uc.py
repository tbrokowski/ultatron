"""
tests/dataset_adapters/test_bus_uc.py  ·  BUSUCAdapter contract tests
======================================================================
Self-contained: builds its own synthetic fixture, no conftest.py changes needed.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_bus_uc.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


# ── Synthetic fixture ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bus_uc_root(tmp_path_factory):
    """
    Build a minimal BUS-UC directory structure:
        root/Benign/images/01.png ... 04.png
        root/Benign/masks/01.png  ... 04.png
        root/Malignant/images/01.png ... 04.png
        root/Malignant/masks/01.png  ... 04.png
    """
    root = tmp_path_factory.mktemp("BUS-UC")
    for cls in ("Benign", "Malignant"):
        (root / cls / "images").mkdir(parents=True)
        (root / cls / "masks").mkdir(parents=True)
        for i in range(1, 5):
            fname = f"{i:02d}.png"
            (root / cls / "images" / fname).write_bytes(b"\x89PNG")
            (root / cls / "masks"  / fname).write_bytes(b"\x89PNG")
    return root


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBUSUCAdapter:

    def test_import(self):
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        assert BUSUCAdapter.DATASET_ID     == "BUS-UC"
        assert BUSUCAdapter.ANATOMY_FAMILY == "breast"
        assert BUSUCAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "BUS-UC" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, bus_uc_root):
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        entries = list(BUSUCAdapter(root=bus_uc_root).iter_entries())
        assert len(entries) == 8  # 4 benign + 4 malignant

    def test_entry_schema(self, bus_uc_root):
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in BUSUCAdapter(root=bus_uc_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "BUS-UC"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert len(e.image_paths) == 1
            assert e.curriculum_tier  in {1, 2, 3}

    def test_has_mask(self, bus_uc_root):
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        entries = list(BUSUCAdapter(root=bus_uc_root).iter_entries())
        assert all(e.has_mask for e in entries)
        assert all(e.instances[0].mask_path is not None for e in entries)

    def test_label_ontology(self, bus_uc_root):
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        ontologies = {
            e.instances[0].label_ontology
            for e in BUSUCAdapter(root=bus_uc_root).iter_entries()
        }
        assert "breast_lesion_benign"    in ontologies
        assert "breast_lesion_malignant" in ontologies

    def test_task_type(self, bus_uc_root):
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        for e in BUSUCAdapter(root=bus_uc_root).iter_entries():
            assert e.task_type == "segmentation"

    def test_split_override(self, bus_uc_root):
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        for e in BUSUCAdapter(root=bus_uc_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, bus_uc_root):
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        ids = [e.sample_id for e in BUSUCAdapter(root=bus_uc_root).iter_entries()]
        assert len(ids) == len(set(ids)), "sample_ids must be unique"

    def test_no_all_folder_duplicates(self, bus_uc_root):
        """All/ folder must be ignored — no duplicate entries."""
        from data.adapters.breast.bus_uc_adapter import BUSUCAdapter
        entries = list(BUSUCAdapter(root=bus_uc_root).iter_entries())
        paths = [e.image_paths[0] for e in entries]
        assert len(paths) == len(set(paths)), "Duplicate paths — All/ folder leaked in"

    def test_build_manifest_for_dataset(self, bus_uc_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "bus_uc.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("BUS-UC", bus_uc_root, writer)
        assert count == 8
        entries = load_manifest(out)
        assert all(e.dataset_id == "BUS-UC" for e in entries)
