"""
tests/adapters/ovarian/test_mmotu3d.py
=======================================

Unit tests for MMOTU3DAdapter.

Run with:
    pytest tests/adapters/ovarian/test_mmotu3d.py -v
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
#   OTU_3d/
#     images/        100.JPG  101.JPG  102.JPG
#     annotations/   100.PNG  100_binary.PNG
#                    101.PNG  101_binary.PNG
#                    102.PNG             ← no binary mask
#     train.txt      100  102
#     val.txt        101
#     train_cls.txt  100.JPG  4    102.JPG  0
#     val_cls.txt    101.JPG  6

_TRAIN_IDS = ["100", "102"]
_VAL_IDS   = ["101"]
_TRAIN_CLS = [("100.JPG", "4"), ("102.JPG", "0")]
_VAL_CLS   = [("101.JPG", "6")]


def _build_mmotu3d(root: Path) -> Path:
    base = root / "OTU_3d"
    (base / "images").mkdir(parents=True)
    (base / "annotations").mkdir(parents=True)

    for stem in _TRAIN_IDS + _VAL_IDS:
        (base / "images"      / f"{stem}.JPG").write_bytes(b"\x00")
        (base / "annotations" / f"{stem}.PNG").write_bytes(b"\x00")

    # binary masks for 100 and 101 only (102 intentionally missing)
    for stem in ["100", "101"]:
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
def mmotu3d_root(tmp_path_factory):
    return _build_mmotu3d(tmp_path_factory.mktemp("MMOTU3D"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries(root, **kwargs):
    from data.adapters.ovarian.mmotu3d import MMOTU3DAdapter
    return list(MMOTU3DAdapter(root, **kwargs).iter_entries())


def _by_stem(root, **kwargs):
    return {Path(e.image_paths[0]).stem: e for e in _entries(root, **kwargs)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMMOTU3DMeta:

    def test_class_attributes(self):
        from data.adapters.ovarian.mmotu3d import MMOTU3DAdapter
        assert MMOTU3DAdapter.DATASET_ID     == "MMOTU-3D"
        assert MMOTU3DAdapter.ANATOMY_FAMILY == "ovarian"
        assert MMOTU3DAdapter.SONODQS        == "silver"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "MMOTU-3D" in ADAPTER_REGISTRY

    def test_nested_root_resolution(self, tmp_path):
        _build_mmotu3d(tmp_path)
        assert len(_entries(tmp_path)) == 3


class TestMMOTU3DEntryCount:

    def test_total_entries(self, mmotu3d_root):
        assert len(_entries(mmotu3d_root)) == 3

    def test_image_without_mask_skipped(self, tmp_path):
        base = tmp_path / "OTU_3d"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        (base / "images" / "999.JPG").write_bytes(b"\x00")
        # no 999.PNG → skipped
        (base / "train.txt").write_text("999\n")
        (base / "val.txt").write_text("")
        (base / "train_cls.txt").write_text("999.JPG  2\n")
        (base / "val_cls.txt").write_text("")
        assert _entries(tmp_path) == []


class TestMMOTU3DSchema:

    def test_entry_fields(self, mmotu3d_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _entries(mmotu3d_root):
            assert e.dataset_id         == "MMOTU-3D"
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

    def test_one_instance_per_entry(self, mmotu3d_root):
        for e in _entries(mmotu3d_root):
            assert len(e.instances) == 1

    def test_instance_ontology(self, mmotu3d_root):
        for e in _entries(mmotu3d_root):
            assert e.instances[0].label_ontology == "ovarian_tumor"
            assert e.instances[0].is_promptable  is True
            assert e.instances[0].mask_path       is not None


class TestMMOTU3DLabels:

    def test_label_raw_is_class_name(self, mmotu3d_root):
        entries = _by_stem(mmotu3d_root)
        assert entries["100"].label_raw == ["simple_cyst"]                   # class 4
        assert entries["101"].label_raw == ["high_grade_serous_carcinoma"]   # class 6
        assert entries["102"].label_raw == ["chocolate_cyst"]                # class 0

    def test_instance_label_raw_matches_entry(self, mmotu3d_root):
        for e in _entries(mmotu3d_root):
            assert e.instances[0].label_raw == e.label_raw[0]

    def test_seven_classes_no_normal_ovary(self):
        from data.adapters.ovarian.mmotu3d import _CLASS_NAMES
        assert len(_CLASS_NAMES) == 7
        assert "normal_ovary" not in _CLASS_NAMES.values()
        assert 0 in _CLASS_NAMES   # 0-indexed
        assert 6 in _CLASS_NAMES
        assert 7 not in _CLASS_NAMES

    def test_unknown_class_falls_back(self, tmp_path):
        base = tmp_path / "OTU_3d"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        (base / "images" / "500.JPG").write_bytes(b"\x00")
        (base / "annotations" / "500.PNG").write_bytes(b"\x00")
        (base / "train.txt").write_text("500\n")
        (base / "val.txt").write_text("")
        (base / "train_cls.txt").write_text("")
        (base / "val_cls.txt").write_text("")
        entries = _entries(tmp_path)
        assert len(entries) == 1
        assert entries[0].label_raw == ["ovarian_tumor"]


class TestMMOTU3DSplit:

    def test_split_values(self, mmotu3d_root):
        entries = _by_stem(mmotu3d_root)
        assert entries["100"].split == "train"
        assert entries["102"].split == "train"
        assert entries["101"].split == "val"

    def test_no_test_split(self, mmotu3d_root):
        assert "test" not in {e.split for e in _entries(mmotu3d_root)}

    def test_split_override(self, mmotu3d_root):
        for e in _entries(mmotu3d_root, split_override="test"):
            assert e.split == "test"


class TestMMOTU3DStudyId:

    def test_study_id_is_stem(self, mmotu3d_root):
        for e in _entries(mmotu3d_root):
            assert e.study_id == Path(e.image_paths[0]).stem

    def test_study_id_values(self, mmotu3d_root):
        assert {e.study_id for e in _entries(mmotu3d_root)} == {"100", "101", "102"}


class TestMMOTU3DSourceMeta:

    def test_class_id_stored(self, mmotu3d_root):
        entries = _by_stem(mmotu3d_root)
        assert entries["100"].source_meta["class_id"] == 4
        assert entries["101"].source_meta["class_id"] == 6
        assert entries["102"].source_meta["class_id"] == 0

    def test_binary_mask_present_when_exists(self, mmotu3d_root):
        entries = _by_stem(mmotu3d_root)
        bm = entries["100"].source_meta["binary_mask_path"]
        assert bm is not None
        assert Path(bm).exists()
        assert bm.endswith("_binary.PNG")

    def test_binary_mask_none_when_missing(self, mmotu3d_root):
        entries = _by_stem(mmotu3d_root)
        assert entries["102"].source_meta["binary_mask_path"] is None

    def test_primary_mask_is_not_binary(self, mmotu3d_root):
        for e in _entries(mmotu3d_root):
            mp = e.instances[0].mask_path
            assert Path(mp).exists()
            assert "_binary" not in mp


class TestMMOTU3DManifest:

    def test_sample_ids_unique(self, mmotu3d_root):
        ids = [e.sample_id for e in _entries(mmotu3d_root)]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, mmotu3d_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "mmotu3d.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("MMOTU-3D", mmotu3d_root, writer)
        assert count == 3
        entries = load_manifest(out)
        assert all(e.dataset_id == "MMOTU-3D" for e in entries)
