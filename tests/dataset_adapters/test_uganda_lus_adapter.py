"""
tests/dataset_adapters/test_uganda_lus_adapter.py  ·  UgandaLUSAdapter tests
=============================================================================
Self-contained synthetic fixture — no real data needed.

Run with:
    PYTHONPATH=/path/to/ultatron pytest tests/dataset_adapters/test_uganda_lus_adapter.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uganda_lus_root(tmp_path_factory):
    """
    Synthetic Uganda LUS layout:

        dataset/
          train/
            covid/    img_c1.png  img_c2.png
            healthy/  img_h1.png
            other/    img_o1.png
          test/
            covid/    img_tc1.png
            healthy/  img_th1.png
            other/    (empty)
          validation/
            covid/    img_vc1.png
            healthy/  (empty)
            other/    img_vo1.png
    """
    root = tmp_path_factory.mktemp("UgandaLUS")
    dataset = root / "dataset"

    layout = {
        "train/covid":      ["img_c1.png", "img_c2.png"],
        "train/healthy":    ["img_h1.png"],
        "train/other":      ["img_o1.png"],
        "test/covid":       ["img_tc1.png"],
        "test/healthy":     ["img_th1.png"],
        "test/other":       [],
        "validation/covid": ["img_vc1.png"],
        "validation/healthy": [],
        "validation/other": ["img_vo1.png"],
    }
    for rel, files in layout.items():
        d = dataset / rel
        d.mkdir(parents=True, exist_ok=True)
        for fname in files:
            (d / fname).write_bytes(b"\x00")

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUgandaLUSAdapter:

    def test_import(self):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        assert UgandaLUSAdapter.DATASET_ID     == "uganda-lus"
        assert UgandaLUSAdapter.ANATOMY_FAMILY == "lung"
        assert UgandaLUSAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "uganda-lus" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        entries = list(UgandaLUSAdapter(root=uganda_lus_root).iter_entries())
        # train: 2+1+1=4, test: 1+1+0=2, val: 1+0+1=2 → total 8
        assert len(entries) == 8

    def test_entry_schema(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in UgandaLUSAdapter(root=uganda_lus_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "uganda-lus"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.task_type       == "multiclass_cls"
            assert e.has_mask        is False
            assert e.has_temporal_order is False
            assert e.num_frames      == 1
            assert e.curriculum_tier in {1, 2, 3}

    def test_split_mapping(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        entries = list(UgandaLUSAdapter(root=uganda_lus_root).iter_entries())
        splits = {e.split for e in entries}
        assert splits == {"train", "val", "test"}

        # validation/ folder must map to "val", never "validation"
        val_entries = [e for e in entries if e.split == "val"]
        assert len(val_entries) == 2  # img_vc1 + img_vo1

    def test_label_raw_values(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        entries = list(UgandaLUSAdapter(root=uganda_lus_root).iter_entries())
        label_raws = {e.instances[0].label_raw for e in entries if e.instances}
        assert label_raws == {"covid", "healthy", "other"}

    def test_one_instance_per_entry(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        for e in UgandaLUSAdapter(root=uganda_lus_root).iter_entries():
            assert len(e.instances) == 1

    def test_study_id_is_filename_stem(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        for e in UgandaLUSAdapter(root=uganda_lus_root).iter_entries():
            img_stem = Path(e.image_paths[0]).stem
            assert e.study_id == img_stem

    def test_instance_id_matches_stem(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        for e in UgandaLUSAdapter(root=uganda_lus_root).iter_entries():
            inst = e.instances[0]
            assert inst.instance_id == Path(e.image_paths[0]).stem

    def test_sample_ids_unique(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        ids = [e.sample_id for e in UgandaLUSAdapter(root=uganda_lus_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_split_override(self, uganda_lus_root):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        for e in UgandaLUSAdapter(root=uganda_lus_root, split_override="test").iter_entries():
            assert e.split == "test"

    def test_missing_dataset_subdir_raises(self, tmp_path):
        from data.adapters.lung.uganda_lus import UgandaLUSAdapter
        # root exists but has no dataset/ subfolder
        with pytest.raises(FileNotFoundError, match="dataset/"):
            list(UgandaLUSAdapter(root=tmp_path).iter_entries())

    def test_build_manifest_for_dataset(self, uganda_lus_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "uganda_lus.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("uganda-lus", uganda_lus_root, writer)
        assert count == 8
        entries = load_manifest(out)
        assert all(e.dataset_id == "uganda-lus" for e in entries)
