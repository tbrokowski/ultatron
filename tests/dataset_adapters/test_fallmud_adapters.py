"""
tests/dataset_adapters/test_fallmud_adapters.py  ·  FALLMUDAdapter contract tests
==================================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_fallmud_adapters.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def fallmud_root(tmp_path_factory):
    """
    Synthetic FALLMUD layout:
      NeilCronin/
        images/            img_00001.tif, img_00002.tif
        fascicle_masks/    img_00001.tif, img_00002.tif
        aponeurosis_masks/ img_00001.jpg, img_00002.jpg
      RyanCunningham/
        images/            0.jpg, 1.jpg
        fascicle_masks/    0.jpg, 1.jpg
        aponeurosis_masks/ 0.jpg, 1.jpg
    """
    root = tmp_path_factory.mktemp("FALLMUD")
    for sub, stems, img_ext, mask_ext in [
        ("NeilCronin",     ["img_00001", "img_00002"], ".tif", ".jpg"),
        ("RyanCunningham", ["0", "1"],                 ".jpg", ".jpg"),
    ]:
        (root / sub / "images").mkdir(parents=True)
        (root / sub / "fascicle_masks").mkdir(parents=True)
        (root / sub / "aponeurosis_masks").mkdir(parents=True)
        for s in stems:
            (root / sub / "images"            / (s + img_ext)).write_bytes(b"\x89PNG")
            (root / sub / "fascicle_masks"    / (s + mask_ext)).write_bytes(b"\x89PNG")
            (root / sub / "aponeurosis_masks" / (s + mask_ext)).write_bytes(b"\x89PNG")
    return root


class TestFALLMUDAdapter:

    def test_import(self):
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        assert FALLMUDAdapter.DATASET_ID     == "FALLMUD"
        assert FALLMUDAdapter.ANATOMY_FAMILY == "muscle"
        assert FALLMUDAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "FALLMUD" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, fallmud_root):
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        entries = list(FALLMUDAdapter(root=fallmud_root).iter_entries())
        assert len(entries) == 4  # 2 NeilCronin + 2 RyanCunningham

    def test_entry_schema(self, fallmud_root):
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in FALLMUDAdapter(root=fallmud_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "FALLMUD"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.task_type       == "segmentation"
            assert e.probe_type      == "linear"

    def test_two_instances_per_entry(self, fallmud_root):
        """Each image must have exactly 2 instances: fascicle + aponeurosis."""
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        for e in FALLMUDAdapter(root=fallmud_root).iter_entries():
            assert len(e.instances) == 2
            labels = {inst.label_raw for inst in e.instances}
            assert labels == {"muscle_fascicle", "muscle_aponeurosis"}

    def test_mask_paths_exist(self, fallmud_root):
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        for e in FALLMUDAdapter(root=fallmud_root).iter_entries():
            for inst in e.instances:
                assert inst.mask_path is not None
                assert Path(inst.mask_path).exists()
                assert inst.is_promptable is True

    def test_label_ontology(self, fallmud_root):
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        for e in FALLMUDAdapter(root=fallmud_root).iter_entries():
            for inst in e.instances:
                assert inst.label_ontology == "muscle"

    def test_sub_datasets_in_meta(self, fallmud_root):
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        entries = list(FALLMUDAdapter(root=fallmud_root).iter_entries())
        subs = {e.source_meta["sub_dataset"] for e in entries}
        assert "NeilCronin"     in subs
        assert "RyanCunningham" in subs

    def test_split_override(self, fallmud_root):
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        for e in FALLMUDAdapter(root=fallmud_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, fallmud_root):
        from data.adapters.muscle.fallmud import FALLMUDAdapter
        ids = [e.sample_id for e in FALLMUDAdapter(root=fallmud_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, fallmud_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "fallmud.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("FALLMUD", fallmud_root, writer)
        assert count == 4
        entries = load_manifest(out)
        assert all(e.dataset_id == "FALLMUD" for e in entries)
