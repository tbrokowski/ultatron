"""
tests/dataset_adapters/test_s1.py  ·  S1Adapter contract tests
===============================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_s1.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def s1_root(tmp_path_factory):
    """
    Synthetic S1 layout:
      TrainingDataSet/BreastTumourImages/       0.jpg ... 4.jpg
      TrainingDataSet/General-1-channel-Labels/ 0.png ... 4.png
      TestingDataSet/Test-General-1-BreastTumourImages/ 100.jpg 101.jpg
      TestingDataSet/Test-General-1-channel-Labels/     100.png 101.png
    """
    root = tmp_path_factory.mktemp("S1")

    # Training
    train_imgs  = root / "TrainingDataSet" / "BreastTumourImages"
    train_masks = root / "TrainingDataSet" / "General-1-channel-Labels"
    train_imgs.mkdir(parents=True)
    train_masks.mkdir(parents=True)
    for i in range(5):
        (train_imgs  / f"{i}.jpg").write_bytes(b"\xff\xd8")
        (train_masks / f"{i}.png").write_bytes(b"\x89PNG")

    # Testing
    test_imgs  = root / "TestingDataSet" / "Test-General-1-BreastTumourImages"
    test_masks = root / "TestingDataSet" / "Test-General-1-channel-Labels"
    test_imgs.mkdir(parents=True)
    test_masks.mkdir(parents=True)
    for i in [100, 101]:
        (test_imgs  / f"{i}.jpg").write_bytes(b"\xff\xd8")
        (test_masks / f"{i}.png").write_bytes(b"\x89PNG")

    return root


class TestS1Adapter:

    def test_import(self):
        from data.adapters.breast.s1_adapter import S1Adapter
        assert S1Adapter.DATASET_ID     == "S1"
        assert S1Adapter.ANATOMY_FAMILY == "breast"
        assert S1Adapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "S1" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, s1_root):
        from data.adapters.breast.s1_adapter import S1Adapter
        entries = list(S1Adapter(root=s1_root).iter_entries())
        assert len(entries) == 7  # 5 train + 2 test

    def test_split_assignment(self, s1_root):
        from data.adapters.breast.s1_adapter import S1Adapter
        entries = list(S1Adapter(root=s1_root).iter_entries())
        train_entries = [e for e in entries if e.split == "train"]
        test_entries  = [e for e in entries if e.split == "test"]
        assert len(train_entries) == 5
        assert len(test_entries)  == 2

    def test_entry_schema(self, s1_root):
        from data.adapters.breast.s1_adapter import S1Adapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in S1Adapter(root=s1_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "S1"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.curriculum_tier in {1, 2, 3}

    def test_has_mask(self, s1_root):
        from data.adapters.breast.s1_adapter import S1Adapter
        entries = list(S1Adapter(root=s1_root).iter_entries())
        assert all(e.has_mask for e in entries)
        assert all(e.task_type == "segmentation" for e in entries)

    def test_label_ontology(self, s1_root):
        from data.adapters.breast.s1_adapter import S1Adapter
        for e in S1Adapter(root=s1_root).iter_entries():
            assert e.instances[0].label_ontology == "breast_lesion"

    def test_split_override(self, s1_root):
        from data.adapters.breast.s1_adapter import S1Adapter
        for e in S1Adapter(root=s1_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, s1_root):
        from data.adapters.breast.s1_adapter import S1Adapter
        ids = [e.sample_id for e in S1Adapter(root=s1_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, s1_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "s1.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("S1", s1_root, writer)
        assert count == 7
        entries = load_manifest(out)
        assert all(e.dataset_id == "S1" for e in entries)
