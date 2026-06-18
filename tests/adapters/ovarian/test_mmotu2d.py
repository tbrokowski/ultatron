"""
tests/adapters/ovarian/test_mmotu2d.py
=======================================

Unit tests for MMOTU2DAdapter.

Run with:
    pytest tests/adapters/ovarian/test_mmotu2d.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
#
# Synthetic layout:
#   OTU_2d/
#     images/        1000.JPG  1001.JPG  1002.JPG
#     annotations/   1000.PNG  1000_binary.PNG
#                    1001.PNG  1001_binary.PNG
#                    1002.PNG            ← no binary mask
#     train.txt      1000  1002
#     val.txt        1001
#     train_cls.txt  1000.JPG  5    1002.JPG  1
#     val_cls.txt    1001.JPG  7
#
# Expected entries: 3 (one per image that has a .PNG mask)

_TRAIN_IDS  = ["1000", "1002"]
_VAL_IDS    = ["1001"]
_TRAIN_CLS  = [("1000.JPG", "5"), ("1002.JPG", "1")]
_VAL_CLS    = [("1001.JPG", "7")]


def _build_mmotu2d(root: Path) -> Path:
    base = root / "OTU_2d"
    (base / "images").mkdir(parents=True)
    (base / "annotations").mkdir(parents=True)

    all_ids = _TRAIN_IDS + _VAL_IDS
    for stem in all_ids:
        (base / "images"      / f"{stem}.JPG").write_bytes(b"\x00")
        (base / "annotations" / f"{stem}.PNG").write_bytes(b"\x00")

    # binary masks for 1000 and 1001 only (1002 intentionally missing)
    for stem in ["1000", "1001"]:
        (base / "annotations" / f"{stem}_binary.PNG").write_bytes(b"\x00")

    (base / "train.txt").write_text("\n".join(_TRAIN_IDS) + "\n")
    (base / "val.txt").write_text("\n".join(_VAL_IDS) + "\n")
    (base / "train_cls.txt").write_text(
        "\n".join(f"{f}  {c}" for f, c in _TRAIN_CLS) + "\n"
    )
    (base / "val_cls.txt").write_text(
        "\n".join(f"{f}  {c}" for f, c in _VAL_CLS) + "\n"
    )
    return root


@pytest.fixture(scope="module")
def mmotu2d_root(tmp_path_factory):
    return _build_mmotu2d(tmp_path_factory.mktemp("MMOTU2D"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries(root, **kwargs):
    from data.adapters.ovarian.mmotu2d import MMOTU2DAdapter
    return list(MMOTU2DAdapter(root, **kwargs).iter_entries())


def _by_stem(root, **kwargs):
    return {Path(e.image_paths[0]).stem: e for e in _entries(root, **kwargs)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMMOTU2DMeta:

    def test_class_attributes(self):
        from data.adapters.ovarian.mmotu2d import MMOTU2DAdapter
        assert MMOTU2DAdapter.DATASET_ID     == "MMOTU-2D"
        assert MMOTU2DAdapter.ANATOMY_FAMILY == "ovarian"
        assert MMOTU2DAdapter.SONODQS        == "silver"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "MMOTU-2D" in ADAPTER_REGISTRY

    def test_nested_root_resolution(self, tmp_path):
        _build_mmotu2d(tmp_path)
        assert len(_entries(tmp_path)) == 3


class TestMMOTU2DEntryCount:

    def test_total_entries(self, mmotu2d_root):
        assert len(_entries(mmotu2d_root)) == 3

    def test_image_without_mask_skipped(self, tmp_path):
        base = tmp_path / "OTU_2d"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        (base / "images" / "999.JPG").write_bytes(b"\x00")
        # no 999.PNG → should be skipped
        (base / "train.txt").write_text("999\n")
        (base / "val.txt").write_text("")
        (base / "train_cls.txt").write_text("999.JPG  3\n")
        (base / "val_cls.txt").write_text("")
        assert _entries(tmp_path) == []


class TestMMOTU2DSchema:

    def test_entry_fields(self, mmotu2d_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _entries(mmotu2d_root):
            assert e.dataset_id         == "MMOTU-2D"
            assert e.anatomy_family     in ANATOMY_FAMILIES
            assert e.modality_type      == "image"
            assert e.ssl_stream         == "image"
            assert e.task_type          == "segmentation"
            assert e.has_mask           is True
            assert e.has_box            is False
            assert e.has_temporal_order is False
            assert e.num_frames         == 1
            assert e.is_promptable      is True
            assert e.curriculum_tier    in {1, 2, 3}
            assert len(e.image_paths)   == 1

    def test_one_instance_per_entry(self, mmotu2d_root):
        for e in _entries(mmotu2d_root):
            assert len(e.instances) == 1

    def test_instance_ontology(self, mmotu2d_root):
        for e in _entries(mmotu2d_root):
            assert e.instances[0].label_ontology == "ovarian_tumor"
            assert e.instances[0].is_promptable  is True
            assert e.instances[0].mask_path       is not None


class TestMMOTU2DLabels:

    def test_label_raw_is_class_name(self, mmotu2d_root):
        entries = _by_stem(mmotu2d_root)
        assert entries["1000"].label_raw == ["simple_cyst"]            # class 5
        assert entries["1001"].label_raw == ["high_grade_serous_carcinoma"]  # class 7
        assert entries["1002"].label_raw == ["chocolate_cyst"]         # class 1

    def test_instance_label_raw_matches_entry(self, mmotu2d_root):
        for e in _entries(mmotu2d_root):
            assert e.instances[0].label_raw == e.label_raw[0]

    def test_unknown_class_falls_back_to_ovarian_tumor(self, tmp_path):
        base = tmp_path / "OTU_2d"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        (base / "images" / "500.JPG").write_bytes(b"\x00")
        (base / "annotations" / "500.PNG").write_bytes(b"\x00")
        (base / "train.txt").write_text("500\n")
        (base / "val.txt").write_text("")
        (base / "train_cls.txt").write_text("")   # no class info
        (base / "val_cls.txt").write_text("")
        entries = _entries(tmp_path)
        assert len(entries) == 1
        assert entries[0].label_raw == ["ovarian_tumor"]


class TestMMOTU2DSplit:

    def test_split_values(self, mmotu2d_root):
        entries = _by_stem(mmotu2d_root)
        assert entries["1000"].split == "train"
        assert entries["1002"].split == "train"
        assert entries["1001"].split == "val"

    def test_no_test_split(self, mmotu2d_root):
        splits = {e.split for e in _entries(mmotu2d_root)}
        assert "test" not in splits

    def test_split_override(self, mmotu2d_root):
        for e in _entries(mmotu2d_root, split_override="test"):
            assert e.split == "test"


class TestMMOTU2DStudyId:

    def test_study_id_is_stem(self, mmotu2d_root):
        for e in _entries(mmotu2d_root):
            assert e.study_id == Path(e.image_paths[0]).stem

    def test_study_id_values(self, mmotu2d_root):
        ids = {e.study_id for e in _entries(mmotu2d_root)}
        assert ids == {"1000", "1001", "1002"}


class TestMMOTU2DSourceMeta:

    def test_class_id_stored(self, mmotu2d_root):
        entries = _by_stem(mmotu2d_root)
        assert entries["1000"].source_meta["class_id"] == 5
        assert entries["1001"].source_meta["class_id"] == 7
        assert entries["1002"].source_meta["class_id"] == 1

    def test_binary_mask_path_present_when_exists(self, mmotu2d_root):
        entries = _by_stem(mmotu2d_root)
        bm = entries["1000"].source_meta["binary_mask_path"]
        assert bm is not None
        assert Path(bm).exists()
        assert bm.endswith("_binary.PNG")

    def test_binary_mask_path_none_when_missing(self, mmotu2d_root):
        entries = _by_stem(mmotu2d_root)
        # 1002 has no binary mask
        assert entries["1002"].source_meta["binary_mask_path"] is None

    def test_primary_mask_path_exists(self, mmotu2d_root):
        for e in _entries(mmotu2d_root):
            assert Path(e.instances[0].mask_path).exists()
            assert e.instances[0].mask_path.endswith(".PNG")
            assert "_binary" not in e.instances[0].mask_path


class TestMMOTU2DManifest:

    def test_sample_ids_unique(self, mmotu2d_root):
        ids = [e.sample_id for e in _entries(mmotu2d_root)]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, mmotu2d_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "mmotu2d.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("MMOTU-2D", mmotu2d_root, writer)
        assert count == 3
        entries = load_manifest(out)
        assert all(e.dataset_id == "MMOTU-2D" for e in entries)
