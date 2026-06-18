"""
tests/adapters/liver/test_ctrus.py
====================================

Unit tests for CTRUSAdapter.

Run with:
    pytest tests/adapters/liver/test_ctrus.py -v
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

# 5 images spanning all 5 folds (0-4) and 2 quality levels
_CSV_ROWS = [
    {"file": "img_f0.jpg", "quality": "0", "quality_name": "high",   "patient": "1",  "testitem_in_fold": "0"},
    {"file": "img_f1.jpg", "quality": "1", "quality_name": "medium", "patient": "2",  "testitem_in_fold": "1"},
    {"file": "img_f2.jpg", "quality": "2", "quality_name": "low",    "patient": "3",  "testitem_in_fold": "2"},
    {"file": "img_f3.jpg", "quality": "0", "quality_name": "high",   "patient": "4",  "testitem_in_fold": "3"},
    {"file": "img_f4.jpg", "quality": "1", "quality_name": "medium", "patient": "5",  "testitem_in_fold": "4"},
]


def _build_ctrus(root: Path) -> Path:
    """Build synthetic c-trus-main/ layout under root. Returns the data root."""
    base = root / "c-trus-main"
    (base / "original").mkdir(parents=True)
    (base / "labels").mkdir(parents=True)

    for row in _CSV_ROWS:
        (base / "original" / row["file"]).write_bytes(b"\x00")
        (base / "labels"   / row["file"]).write_bytes(b"\x00")

    csv_path = base / "c-trus.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_CSV_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(_CSV_ROWS)

    return root


@pytest.fixture(scope="module")
def ctrus_root(tmp_path_factory):
    return _build_ctrus(tmp_path_factory.mktemp("CTRUS"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries(root, **kwargs):
    from data.adapters.liver.ctrus import CTRUSAdapter
    return list(CTRUSAdapter(root, **kwargs).iter_entries())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCTRUSMeta:

    def test_class_attributes(self):
        from data.adapters.liver.ctrus import CTRUSAdapter
        assert CTRUSAdapter.DATASET_ID     == "C-TRUS"
        assert CTRUSAdapter.ANATOMY_FAMILY == "liver"
        assert CTRUSAdapter.SONODQS        == "silver"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "C-TRUS" in ADAPTER_REGISTRY

    def test_nested_root_resolution(self, tmp_path):
        """Adapter works when pointed at the parent of c-trus-main/."""
        _build_ctrus(tmp_path)
        entries = _entries(tmp_path)
        assert len(entries) == len(_CSV_ROWS)


class TestCTRUSEntryCount:

    def test_total_entries(self, ctrus_root):
        assert len(_entries(ctrus_root)) == len(_CSV_ROWS)

    def test_missing_mask_skipped(self, tmp_path):
        """Images without a matching mask file are silently skipped."""
        base = tmp_path / "c-trus-main"
        (base / "original").mkdir(parents=True)
        (base / "labels").mkdir(parents=True)
        (base / "original" / "orphan.jpg").write_bytes(b"\x00")
        # no matching labels/orphan.jpg
        csv_path = base / "c-trus.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "quality", "quality_name", "patient", "testitem_in_fold"])
            w.writeheader()
            w.writerow({"file": "orphan.jpg", "quality": "0", "quality_name": "high",
                        "patient": "1", "testitem_in_fold": "2"})
        assert _entries(tmp_path) == []


class TestCTRUSSchema:

    def test_entry_fields(self, ctrus_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _entries(ctrus_root):
            assert e.dataset_id         == "C-TRUS"
            assert e.anatomy_family     in ANATOMY_FAMILIES
            assert e.modality_type      == "image"
            assert e.ssl_stream         == "image"
            assert e.task_type          == "segmentation"
            assert e.has_mask           is True
            assert e.has_temporal_order is False
            assert e.num_frames         == 1
            assert e.is_promptable      is True
            assert e.label_raw          == ["colon_wall"]
            assert e.curriculum_tier    in {1, 2, 3}
            assert len(e.image_paths)   == 1

    def test_one_instance_per_entry(self, ctrus_root):
        for e in _entries(ctrus_root):
            assert len(e.instances) == 1

    def test_instance_fields(self, ctrus_root):
        for e in _entries(ctrus_root):
            inst = e.instances[0]
            assert inst.label_raw      == "colon_wall"
            assert inst.label_ontology == "colon"
            assert inst.is_promptable  is True
            assert inst.mask_path      is not None


class TestCTRUSStudyId:

    def test_study_id_format(self, ctrus_root):
        for e in _entries(ctrus_root):
            patient = e.source_meta["patient"]
            assert e.study_id == f"patient_{patient}"

    def test_study_id_values(self, ctrus_root):
        study_ids = {e.study_id for e in _entries(ctrus_root)}
        expected  = {f"patient_{r['patient']}" for r in _CSV_ROWS}
        assert study_ids == expected


class TestCTRUSSplit:

    def test_fold0_is_test(self, ctrus_root):
        entries = {Path(e.image_paths[0]).stem: e for e in _entries(ctrus_root)}
        assert entries["img_f0"].split == "test"

    def test_fold1_is_val(self, ctrus_root):
        entries = {Path(e.image_paths[0]).stem: e for e in _entries(ctrus_root)}
        assert entries["img_f1"].split == "val"

    def test_folds_2_3_4_are_train(self, ctrus_root):
        entries = {Path(e.image_paths[0]).stem: e for e in _entries(ctrus_root)}
        for stem in ("img_f2", "img_f3", "img_f4"):
            assert entries[stem].split == "train"

    def test_all_splits_valid(self, ctrus_root):
        for e in _entries(ctrus_root):
            assert e.split in {"train", "val", "test"}

    def test_split_override(self, ctrus_root):
        for e in _entries(ctrus_root, split_override="val"):
            assert e.split == "val"


class TestCTRUSSourceMeta:

    def test_source_meta_keys(self, ctrus_root):
        for e in _entries(ctrus_root):
            assert "patient"      in e.source_meta
            assert "quality_name" in e.source_meta
            assert "fold"         in e.source_meta

    def test_quality_name_values(self, ctrus_root):
        quality_names = {e.source_meta["quality_name"] for e in _entries(ctrus_root)}
        assert quality_names <= {"high", "medium", "low"}

    def test_fold_values(self, ctrus_root):
        folds = {e.source_meta["fold"] for e in _entries(ctrus_root)}
        assert folds == {0, 1, 2, 3, 4}

    def test_mask_path_exists(self, ctrus_root):
        for e in _entries(ctrus_root):
            assert Path(e.instances[0].mask_path).exists()


class TestCTRUSManifest:

    def test_sample_ids_unique(self, ctrus_root):
        ids = [e.sample_id for e in _entries(ctrus_root)]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, ctrus_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "ctrus.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("C-TRUS", ctrus_root, writer)
        assert count == len(_CSV_ROWS)
        entries = load_manifest(out)
        assert all(e.dataset_id == "C-TRUS" for e in entries)
