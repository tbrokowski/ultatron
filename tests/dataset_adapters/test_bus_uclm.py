"""
tests/dataset_adapters/test_bus_uclm.py  ·  BUSUCLMAdapter contract tests
==========================================================================
Self-contained: builds its own synthetic RGB mask fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_bus_uclm.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


# ── Helpers to create synthetic RGB mask PNGs ─────────────────────────────────

def _write_rgb_png(path: Path, r: int, g: int, b: int, size: int = 64):
    """Write a solid-color PNG using PIL if available, else raw bytes."""
    try:
        from PIL import Image
        import numpy as np
        arr = np.full((size, size, 3), [r, g, b], dtype="uint8")
        Image.fromarray(arr, "RGB").save(path)
    except ImportError:
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([r, g, b] * 4))


@pytest.fixture(scope="module")
def bus_uclm_root(tmp_path_factory):
    """
    Synthetic BUS-UCLM with:
      4 benign  images (green mask)
      3 malignant images (red mask)
      3 normal  images (black mask)
    """
    root = tmp_path_factory.mktemp("BUS-UCLM")
    (root / "images").mkdir()
    (root / "masks").mkdir()

    samples = (
        [("BEN", i, 0, 180, 0)   for i in range(4)] +   # green → benign
        [("MAL", i, 180, 0, 0)   for i in range(3)] +   # red   → malignant
        [("NOR", i, 0,   0, 0)   for i in range(3)]     # black → normal
    )

    for prefix, idx, r, g, b in samples:
        fname = f"{prefix}_{idx:03d}.png"
        (root / "images" / fname).write_bytes(b"\x89PNG")
        _write_rgb_png(root / "masks" / fname, r, g, b)

    return root


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBUSUCLMAdapter:

    def test_import(self):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        assert BUSUCLMAdapter.DATASET_ID     == "BUS-UCLM"
        assert BUSUCLMAdapter.ANATOMY_FAMILY == "breast"
        assert BUSUCLMAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "BUS-UCLM" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, bus_uclm_root):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        entries = list(BUSUCLMAdapter(root=bus_uclm_root).iter_entries())
        assert len(entries) == 10  # 4 + 3 + 3

    def test_entry_schema(self, bus_uclm_root):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in BUSUCLMAdapter(root=bus_uclm_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id     == "BUS-UCLM"
            assert e.anatomy_family in ANATOMY_FAMILIES
            assert e.modality_type  == "image"
            assert e.ssl_stream     == "image"
            assert e.split          in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}

    def test_label_ontology(self, bus_uclm_root):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        ontologies = {
            e.instances[0].label_ontology
            for e in BUSUCLMAdapter(root=bus_uclm_root).iter_entries()
        }
        assert "breast_lesion_benign"    in ontologies
        assert "breast_lesion_malignant" in ontologies
        assert "breast_normal"           in ontologies

    def test_normal_has_no_mask(self, bus_uclm_root):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        for e in BUSUCLMAdapter(root=bus_uclm_root).iter_entries():
            if e.instances[0].label_ontology == "breast_normal":
                assert not e.has_mask
                assert e.task_type == "classification"

    def test_lesion_has_mask(self, bus_uclm_root):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        for e in BUSUCLMAdapter(root=bus_uclm_root).iter_entries():
            if e.instances[0].label_ontology in (
                "breast_lesion_benign", "breast_lesion_malignant"
            ):
                assert e.has_mask
                assert e.task_type == "segmentation"

    def test_split_override(self, bus_uclm_root):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        for e in BUSUCLMAdapter(root=bus_uclm_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, bus_uclm_root):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        ids = [e.sample_id for e in BUSUCLMAdapter(root=bus_uclm_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_patient_code_in_source_meta(self, bus_uclm_root):
        from data.adapters.breast.bus_uclm_adapter import BUSUCLMAdapter
        for e in BUSUCLMAdapter(root=bus_uclm_root).iter_entries():
            assert "patient_code" in e.source_meta
            assert e.source_meta["patient_code"] in ("BEN", "MAL", "NOR")

    def test_build_manifest_for_dataset(self, bus_uclm_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "bus_uclm.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("BUS-UCLM", bus_uclm_root, writer)
        assert count == 10
        entries = load_manifest(out)
        assert all(e.dataset_id == "BUS-UCLM" for e in entries)
