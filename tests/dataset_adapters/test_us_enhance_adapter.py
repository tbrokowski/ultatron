"""
tests/dataset_adapters/test_us_enhance_adapter.py  ·  USEnhanceAdapter tests
=============================================================================
Self-contained synthetic fixture — no real data needed.

Run with:
    PYTHONPATH=/path/to/ultatron pytest tests/dataset_adapters/test_us_enhance_adapter.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def us_enhance_root(tmp_path_factory):
    """
    Synthetic layout:

        Training set/
          train_datasets/
            thyroid/  high_quality/0001.png  low_quality/0001.png 0002.png
            liver/    high_quality/0001.png  low_quality/0001.png
            breast/   (low_quality only — no high_quality dir)
                      low_quality/0001.png
            carotid/  high_quality/0001.png  low_quality/0001.png
            kidney/   high_quality/0001.png  low_quality/0001.png
        Testing set/
          low_quality_images/  0005.png  0006.png
    """
    root = tmp_path_factory.mktemp("USEnhance")

    # Training — paired images (high+low) for thyroid, liver, carotid, kidney
    paired_organs = ("thyroid", "liver", "carotid", "kidney")
    for organ in paired_organs:
        for quality in ("high_quality", "low_quality"):
            d = root / "Training set" / "train_datasets" / organ / quality
            d.mkdir(parents=True)
            (d / "0001.png").write_bytes(b"\x00")
    # thyroid gets a second low_quality image (no matching high_quality)
    (root / "Training set" / "train_datasets" / "thyroid" / "low_quality" / "0002.png").write_bytes(b"\x00")

    # breast — low_quality only (no high_quality dir)
    breast_lq = root / "Training set" / "train_datasets" / "breast" / "low_quality"
    breast_lq.mkdir(parents=True)
    (breast_lq / "0001.png").write_bytes(b"\x00")

    # Testing
    test_dir = root / "Testing set" / "low_quality_images"
    test_dir.mkdir(parents=True)
    (test_dir / "0005.png").write_bytes(b"\x00")
    (test_dir / "0006.png").write_bytes(b"\x00")

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUSEnhanceAdapter:

    def test_import(self):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        assert USEnhanceAdapter.DATASET_ID     == "us-enhance-2023"
        assert USEnhanceAdapter.ANATOMY_FAMILY == "multi"
        assert USEnhanceAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "us-enhance-2023" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        entries = list(USEnhanceAdapter(root=us_enhance_root).iter_entries())
        # thyroid: 2 (0001+0002), liver: 1, breast: 1, carotid: 1, kidney: 1 → 6 train
        # test: 2
        assert len(entries) == 8

    def test_entry_schema(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in USEnhanceAdapter(root=us_enhance_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id         == "us-enhance-2023"
            assert e.anatomy_family     in ANATOMY_FAMILIES
            assert e.modality_type      == "image"
            assert e.ssl_stream         == "image"
            assert e.split              in {"train", "test"}
            assert e.task_type          == "weak_label"
            assert e.has_mask           is False
            assert e.has_temporal_order is False
            assert e.num_frames         == 1
            assert e.label_raw          is None
            assert e.instances          == []
            assert e.curriculum_tier    in {1, 2, 3}

    def test_train_split(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        entries = list(USEnhanceAdapter(root=us_enhance_root).iter_entries())
        train = [e for e in entries if e.split == "train"]
        assert len(train) == 6

    def test_test_split(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        entries = list(USEnhanceAdapter(root=us_enhance_root).iter_entries())
        test = [e for e in entries if e.split == "test"]
        assert len(test) == 2

    def test_anatomy_family_per_organ(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        entries = [e for e in USEnhanceAdapter(root=us_enhance_root).iter_entries()
                   if e.split == "train"]
        by_organ = {}
        for e in entries:
            organ = e.source_meta.get("organ")
            by_organ[organ] = e.anatomy_family
        assert by_organ["thyroid"] == "thyroid"
        assert by_organ["liver"]   == "liver"
        assert by_organ["breast"]  == "breast"
        assert by_organ["carotid"] == "carotid"
        assert by_organ["kidney"]  == "kidney"

    def test_test_anatomy_family_is_other(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        for e in USEnhanceAdapter(root=us_enhance_root).iter_entries():
            if e.split == "test":
                assert e.anatomy_family == "other"

    def test_high_quality_path_in_source_meta(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        for e in USEnhanceAdapter(root=us_enhance_root).iter_entries():
            if e.split != "train":
                continue
            assert "high_quality_path" in e.source_meta

    def test_paired_entry_has_valid_hq_path(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        entries = [e for e in USEnhanceAdapter(root=us_enhance_root).iter_entries()
                   if e.split == "train" and e.source_meta.get("organ") == "thyroid"
                   and e.study_id == "0001"]
        assert len(entries) == 1
        hq = entries[0].source_meta["high_quality_path"]
        assert hq is not None
        assert Path(hq).exists()

    def test_unpaired_entry_has_no_hq_path(self, us_enhance_root):
        # thyroid/0002.png has no matching high_quality file
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        entries = [e for e in USEnhanceAdapter(root=us_enhance_root).iter_entries()
                   if e.split == "train" and e.source_meta.get("organ") == "thyroid"
                   and e.study_id == "0002"]
        assert len(entries) == 1
        assert entries[0].source_meta["high_quality_path"] is None

    def test_breast_no_hq_dir(self, us_enhance_root):
        # breast has no high_quality dir at all
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        entries = [e for e in USEnhanceAdapter(root=us_enhance_root).iter_entries()
                   if e.source_meta.get("organ") == "breast"]
        assert len(entries) == 1
        assert entries[0].source_meta["high_quality_path"] is None

    def test_organ_in_source_meta(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        for e in USEnhanceAdapter(root=us_enhance_root).iter_entries():
            if e.split == "train":
                assert "organ" in e.source_meta
                assert e.source_meta["organ"] in ("thyroid", "breast", "carotid", "liver", "kidney")

    def test_study_id_is_stem(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        for e in USEnhanceAdapter(root=us_enhance_root).iter_entries():
            assert e.study_id == Path(e.image_paths[0]).stem

    def test_sample_ids_unique(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        ids = [e.sample_id for e in USEnhanceAdapter(root=us_enhance_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_split_override(self, us_enhance_root):
        from data.adapters.thyroid.us_enhance import USEnhanceAdapter
        for e in USEnhanceAdapter(root=us_enhance_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_build_manifest_for_dataset(self, us_enhance_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "us_enhance.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("us-enhance-2023", us_enhance_root, writer)
        assert count == 8
        entries = load_manifest(out)
        assert all(e.dataset_id == "us-enhance-2023" for e in entries)
