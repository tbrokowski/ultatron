"""
tests/adapters/maternal_fetal/test_ultrasound_fetus.py
=======================================================

Unit tests for UltrasoundFetusAdapter.

Run with:
    pytest tests/adapters/maternal_fetal/test_ultrasound_fetus.py -v
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
# Synthetic layout (mirrors the real Data/Data/ structure):
#
#   train/
#     benign/     img_b1.png  img_b2.png
#     malignant/  img_m1.png  img_m1_Annotation.png   ← mask exists
#                 img_m2.png                           ← no mask
#     normal/     img_n1.png
#   test/
#     benign/     img_tb1.png
#     malignant/  img_tm1.png  img_tm1_Annotation.png
#     normal/     (empty dir)
#   validation/
#     benign/     img_vb1.png
#     malignant/  (empty dir)
#     normal/     img_vn1.png
#
# Expected entries:
#   train: 2 benign + 2 malignant + 1 normal = 5
#   test:  1 benign + 1 malignant + 0 normal = 2
#   val:   1 benign + 0 malignant + 1 normal = 2
#   Total: 9


def _build_fetus(root: Path) -> Path:
    layout = {
        "train/benign":    ["img_b1.png", "img_b2.png"],
        "train/malignant": ["img_m1.png", "img_m1_Annotation.png", "img_m2.png"],
        "train/normal":    ["img_n1.png"],
        "test/benign":     ["img_tb1.png"],
        "test/malignant":  ["img_tm1.png", "img_tm1_Annotation.png"],
        "test/normal":     [],
        "validation/benign":    ["img_vb1.png"],
        "validation/malignant": [],
        "validation/normal":    ["img_vn1.png"],
    }
    for rel, files in layout.items():
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_bytes(b"\x00")
    return root


@pytest.fixture(scope="module")
def fetus_root(tmp_path_factory):
    return _build_fetus(tmp_path_factory.mktemp("UltrasoundFetus"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries(root, **kwargs):
    from data.adapters.maternal_fetal.ultrasound_fetus import UltrasoundFetusAdapter
    return list(UltrasoundFetusAdapter(root, **kwargs).iter_entries())


def _by_stem(root, **kwargs):
    return {Path(e.image_paths[0]).stem: e for e in _entries(root, **kwargs)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUltrasoundFetusMeta:

    def test_class_attributes(self):
        from data.adapters.maternal_fetal.ultrasound_fetus import UltrasoundFetusAdapter
        assert UltrasoundFetusAdapter.DATASET_ID     == "ultrasound-fetus-dataset"
        assert UltrasoundFetusAdapter.ANATOMY_FAMILY == "fetal_head"
        assert UltrasoundFetusAdapter.SONODQS        == "silver"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "ultrasound-fetus-dataset" in ADAPTER_REGISTRY


class TestUltrasoundFetusEntryCount:

    def test_total_entries(self, fetus_root):
        assert len(_entries(fetus_root)) == 9

    def test_annotation_files_skipped(self, fetus_root):
        # _Annotation.png files must never appear as entries
        stems = {Path(e.image_paths[0]).stem for e in _entries(fetus_root)}
        assert not any("_Annotation" in s for s in stems)

    def test_train_count(self, fetus_root):
        assert len([e for e in _entries(fetus_root) if e.split == "train"]) == 5

    def test_test_count(self, fetus_root):
        assert len([e for e in _entries(fetus_root) if e.split == "test"]) == 2

    def test_val_count(self, fetus_root):
        assert len([e for e in _entries(fetus_root) if e.split == "val"]) == 2


class TestUltrasoundFetusSchema:

    def test_entry_fields(self, fetus_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _entries(fetus_root):
            assert e.dataset_id         == "ultrasound-fetus-dataset"
            assert e.anatomy_family     in ANATOMY_FAMILIES
            assert e.modality_type      == "image"
            assert e.ssl_stream         == "image"
            assert e.task_type          == "classification"
            assert e.has_box            is False
            assert e.has_temporal_order is False
            assert e.num_frames         == 1
            assert e.curriculum_tier    in {1, 2, 3}
            assert len(e.image_paths)   == 1

    def test_one_instance_per_entry(self, fetus_root):
        for e in _entries(fetus_root):
            assert len(e.instances) == 1

    def test_instance_ontology(self, fetus_root):
        for e in _entries(fetus_root):
            assert e.instances[0].label_ontology == "fetal_health"


class TestUltrasoundFetusLabels:

    def test_label_raw_values(self, fetus_root):
        label_raws = {e.label_raw[0] for e in _entries(fetus_root)}
        assert label_raws == {"benign", "malignant", "normal"}

    def test_instance_label_raw_matches_entry(self, fetus_root):
        for e in _entries(fetus_root):
            assert e.instances[0].label_raw == e.label_raw[0]

    def test_benign_labels(self, fetus_root):
        entries = _by_stem(fetus_root)
        assert entries["img_b1"].label_raw == ["benign"]
        assert entries["img_b2"].label_raw == ["benign"]

    def test_malignant_labels(self, fetus_root):
        entries = _by_stem(fetus_root)
        assert entries["img_m1"].label_raw == ["malignant"]
        assert entries["img_m2"].label_raw == ["malignant"]

    def test_normal_labels(self, fetus_root):
        entries = _by_stem(fetus_root)
        assert entries["img_n1"].label_raw == ["normal"]


class TestUltrasoundFetusMask:

    def test_malignant_with_annotation_has_mask(self, fetus_root):
        entries = _by_stem(fetus_root)
        # img_m1 has _Annotation.png
        assert entries["img_m1"].has_mask is True
        assert entries["img_m1"].instances[0].mask_path is not None
        assert Path(entries["img_m1"].instances[0].mask_path).exists()

    def test_malignant_without_annotation_no_mask(self, fetus_root):
        entries = _by_stem(fetus_root)
        # img_m2 has no _Annotation.png
        assert entries["img_m2"].has_mask is False
        assert entries["img_m2"].instances[0].mask_path is None

    def test_benign_no_mask(self, fetus_root):
        entries = _by_stem(fetus_root)
        assert entries["img_b1"].has_mask is False
        assert entries["img_b1"].instances[0].mask_path is None

    def test_normal_no_mask(self, fetus_root):
        entries = _by_stem(fetus_root)
        assert entries["img_n1"].has_mask is False

    def test_mask_path_ends_with_annotation(self, fetus_root):
        for e in _entries(fetus_root):
            if e.has_mask:
                mp = e.instances[0].mask_path
                assert mp.endswith("_Annotation.png")


class TestUltrasoundFetusStudyId:

    def test_study_id_is_stem(self, fetus_root):
        for e in _entries(fetus_root):
            assert e.study_id == Path(e.image_paths[0]).stem


class TestUltrasoundFetusSplit:

    def test_split_values(self, fetus_root):
        splits = {e.split for e in _entries(fetus_root)}
        assert splits == {"train", "test", "val"}

    def test_validation_maps_to_val(self, fetus_root):
        entries = _by_stem(fetus_root)
        assert entries["img_vb1"].split == "val"
        assert entries["img_vn1"].split == "val"

    def test_split_override(self, fetus_root):
        for e in _entries(fetus_root, split_override="test"):
            assert e.split == "test"


class TestUltrasoundFetusManifest:

    def test_sample_ids_unique(self, fetus_root):
        ids = [e.sample_id for e in _entries(fetus_root)]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, fetus_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "fetus.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset(
                "ultrasound-fetus-dataset", fetus_root, writer
            )
        assert count == 9
        entries = load_manifest(out)
        assert all(e.dataset_id == "ultrasound-fetus-dataset" for e in entries)
