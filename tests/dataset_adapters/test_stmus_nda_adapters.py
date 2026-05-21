"""
tests/dataset_adapters/test_stmus_nda_adapters.py  ·  STMUSNDAAdapter contract tests
======================================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_stmus_nda_adapters.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def stmus_nda_root(tmp_path_factory):
    """
    Synthetic STMUS-NDA layout (Variant A — per-muscle dirs, co-located masks):
      BB/BB_0000.png + BB/BB_0000_mask.png
      BB/BB_0001.png + BB/BB_0001_mask.png
      TA/TA_0000.png + TA/TA_0000_mask.png
      TA/TA_0001.png + TA/TA_0001_mask.png
      GM/GM_0000.png + GM/GM_0000_mask.png
      GM/GM_0001.png + GM/GM_0001_mask.png
    """
    root = tmp_path_factory.mktemp("STMUS_NDA")
    for muscle in ("BB", "TA", "GM"):
        d = root / muscle
        d.mkdir()
        for i in range(2):
            (d / f"{muscle}_{i:04d}.png").write_bytes(b"\x89PNG")
            (d / f"{muscle}_{i:04d}_mask.png").write_bytes(b"\x89PNG")
    return root


class TestSTMUSNDAAdapter:

    def test_import(self):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        assert STMUSNDAAdapter.DATASET_ID     == "STMUS-NDA"
        assert STMUSNDAAdapter.ANATOMY_FAMILY == "muscle"
        assert STMUSNDAAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "STMUS-NDA" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, stmus_nda_root):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        entries = list(STMUSNDAAdapter(root=stmus_nda_root).iter_entries())
        assert len(entries) == 6  # 3 muscles × 2 images

    def test_entry_schema(self, stmus_nda_root):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in STMUSNDAAdapter(root=stmus_nda_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "STMUS-NDA"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.task_type       == "segmentation"
            assert e.probe_type      == "linear"

    def test_all_entries_have_masks(self, stmus_nda_root):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        for e in STMUSNDAAdapter(root=stmus_nda_root).iter_entries():
            assert e.has_mask is True
            assert len(e.instances) == 1
            assert e.instances[0].mask_path is not None
            assert e.instances[0].is_promptable is True

    def test_muscle_labels(self, stmus_nda_root):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        entries = list(STMUSNDAAdapter(root=stmus_nda_root).iter_entries())
        labels  = {e.instances[0].label_raw for e in entries}
        assert labels == {"biceps_brachii", "tibialis_anterior", "gastrocnemius_medialis"}

    def test_label_ontology(self, stmus_nda_root):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        for e in STMUSNDAAdapter(root=stmus_nda_root).iter_entries():
            assert e.instances[0].label_ontology == "muscle"

    def test_split_override(self, stmus_nda_root):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        for e in STMUSNDAAdapter(root=stmus_nda_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, stmus_nda_root):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        ids = [e.sample_id for e in STMUSNDAAdapter(root=stmus_nda_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_source_meta(self, stmus_nda_root):
        from data.adapters.muscle.stmus_nda import STMUSNDAAdapter
        for e in STMUSNDAAdapter(root=stmus_nda_root).iter_entries():
            assert "muscle"    in e.source_meta
            assert "doi"       in e.source_meta
            assert e.source_meta["doi"] == "https://doi.org/10.17632/3jykz7wz8d.1"

    def test_build_manifest_for_dataset(self, stmus_nda_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "stmus_nda.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("STMUS-NDA", stmus_nda_root, writer)
        assert count == 6
        entries = load_manifest(out)
        assert all(e.dataset_id == "STMUS-NDA" for e in entries)
