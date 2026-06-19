"""
tests/adapters/kidney/test_normal_kidney_cv.py
===============================================

Unit tests for NormalKidneyCVAdapter.

Run with:
    pytest tests/adapters/kidney/test_normal_kidney_cv.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_CATEGORIES = [
    {"id": 0, "name": "Normal-Kidney"},
    {"id": 1, "name": "Kidney"},
    {"id": 2, "name": "Liver"},
    {"id": 3, "name": "Spleen"},
]

# 3 images:
#   img_a → 2 annotations (Kidney + Liver)
#   img_b → 1 annotation  (Kidney)
#   img_c → 1 annotation  (Spleen)
_IMAGES = [
    {"id": 1, "file_name": "img_a.jpg", "height": 480, "width": 640,
     "extra": {"name": "13080.jpg"}},
    {"id": 2, "file_name": "img_b.jpg", "height": 512, "width": 512,
     "extra": {"name": "13081.jpg"}},
    {"id": 3, "file_name": "img_c.jpg", "height": 480, "width": 640,
     "extra": {"name": "13082.jpg"}},
]
_ANNOTATIONS = [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [10.0, 20.0, 100.0, 80.0], "area": 8000,
     "segmentation": [[10.0, 20.0, 110.0, 20.0, 110.0, 100.0, 10.0, 100.0]],
     "iscrowd": 0},
    {"id": 2, "image_id": 1, "category_id": 2,
     "bbox": [200.0, 50.0, 60.0, 40.0], "area": 2400,
     "segmentation": [[200.0, 50.0, 260.0, 50.0, 260.0, 90.0, 200.0, 90.0]],
     "iscrowd": 0},
    {"id": 3, "image_id": 2, "category_id": 1,
     "bbox": [5.0, 5.0, 50.0, 50.0], "area": 2500,
     "segmentation": [[5.0, 5.0, 55.0, 5.0, 55.0, 55.0, 5.0, 55.0]],
     "iscrowd": 0},
    {"id": 4, "image_id": 3, "category_id": 3,
     "bbox": [0.0, 0.0, 30.0, 30.0], "area": 900,
     "segmentation": [[0.0, 0.0, 30.0, 0.0, 30.0, 30.0, 0.0, 30.0]],
     "iscrowd": 0},
]
_COCO = {
    "categories":   _CATEGORIES,
    "images":       _IMAGES,
    "annotations":  _ANNOTATIONS,
}


def _build_normal_kidney(root: Path) -> Path:
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    for img in _IMAGES:
        (train_dir / img["file_name"]).write_bytes(b"\x00")
    (train_dir / "_annotations.coco.json").write_text(
        json.dumps(_COCO), encoding="utf-8"
    )
    return root


@pytest.fixture(scope="module")
def kidney_root(tmp_path_factory):
    return _build_normal_kidney(tmp_path_factory.mktemp("NormalKidneyCV"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries(root, **kwargs):
    from data.adapters.kidney.normal_kidney_cv import NormalKidneyCVAdapter
    return list(NormalKidneyCVAdapter(root, **kwargs).iter_entries())


def _by_study(root, **kwargs):
    return {e.study_id: e for e in _entries(root, **kwargs)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNormalKidneyCVMeta:

    def test_class_attributes(self):
        from data.adapters.kidney.normal_kidney_cv import NormalKidneyCVAdapter
        assert NormalKidneyCVAdapter.DATASET_ID     == "Normal-Kidney-CV"
        assert NormalKidneyCVAdapter.ANATOMY_FAMILY == "kidney"
        assert NormalKidneyCVAdapter.SONODQS        == "silver"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "Normal-Kidney-CV" in ADAPTER_REGISTRY

    def test_nested_root_resolution(self, tmp_path):
        # adapter root can be the parent of train/
        _build_normal_kidney(tmp_path)
        assert len(_entries(tmp_path)) == 3


class TestNormalKidneyCVEntryCount:

    def test_total_entries(self, kidney_root):
        assert len(_entries(kidney_root)) == 3


class TestNormalKidneyCVSchema:

    def test_entry_fields(self, kidney_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _entries(kidney_root):
            assert e.dataset_id         == "Normal-Kidney-CV"
            assert e.anatomy_family     in ANATOMY_FAMILIES
            assert e.modality_type      == "image"
            assert e.ssl_stream         == "image"
            assert e.task_type          == "segmentation"
            assert e.has_mask           is True
            assert e.has_box            is True
            assert e.has_temporal_order is False
            assert e.num_frames         == 1
            assert e.curriculum_tier    in {1, 2, 3}
            assert len(e.image_paths)   == 1

    def test_height_width_populated(self, kidney_root):
        entries = _by_study(kidney_root)
        assert entries["13080"].height == 480
        assert entries["13080"].width  == 640
        assert entries["13081"].height == 512
        assert entries["13081"].width  == 512


class TestNormalKidneyCVStudyId:

    def test_study_id_from_extra_name(self, kidney_root):
        study_ids = {e.study_id for e in _entries(kidney_root)}
        assert study_ids == {"13080", "13081", "13082"}

    def test_study_id_is_stem(self, kidney_root):
        # "13080.jpg" → stem "13080"
        entries = _by_study(kidney_root)
        assert "." not in entries["13080"].study_id


class TestNormalKidneyCVLabelRaw:

    def test_multi_category_image(self, kidney_root):
        entries = _by_study(kidney_root)
        # img_a has Kidney + Liver
        assert set(entries["13080"].label_raw) == {"Kidney", "Liver"}

    def test_single_category_image(self, kidney_root):
        entries = _by_study(kidney_root)
        assert entries["13081"].label_raw == ["Kidney"]
        assert entries["13082"].label_raw == ["Spleen"]

    def test_instance_label_raw_is_category_name(self, kidney_root):
        valid = {c["name"] for c in _CATEGORIES}
        for e in _entries(kidney_root):
            for inst in e.instances:
                assert inst.label_raw in valid


class TestNormalKidneyCVInstances:

    def test_instance_count_matches_annotations(self, kidney_root):
        entries = _by_study(kidney_root)
        assert len(entries["13080"].instances) == 2  # Kidney + Liver
        assert len(entries["13081"].instances) == 1
        assert len(entries["13082"].instances) == 1

    def test_bbox_xyxy_on_instance(self, kidney_root):
        entries = _by_study(kidney_root)
        inst = entries["13080"].instances[0]
        assert inst.bbox_xyxy is not None
        assert len(inst.bbox_xyxy) == 4
        # [x, y, x+w, y+h] = [10, 20, 110, 100]
        assert inst.bbox_xyxy == pytest.approx([10.0, 20.0, 110.0, 100.0])

    def test_polygon_on_instance(self, kidney_root):
        for e in _entries(kidney_root):
            for inst in e.instances:
                assert inst.polygon is not None


class TestNormalKidneyCVSplit:

    def test_all_train(self, kidney_root):
        for e in _entries(kidney_root):
            assert e.split == "train"

    def test_split_override(self, kidney_root):
        for e in _entries(kidney_root, split_override="val"):
            assert e.split == "val"


class TestNormalKidneyCVSourceMeta:

    def test_required_keys(self, kidney_root):
        for e in _entries(kidney_root):
            assert "coco_image_id"    in e.source_meta
            assert "coco_annotations" in e.source_meta
            assert "coco_segmentation" in e.source_meta

    def test_coco_annotations_count(self, kidney_root):
        entries = _by_study(kidney_root)
        assert len(entries["13080"].source_meta["coco_annotations"]) == 2
        assert len(entries["13081"].source_meta["coco_annotations"]) == 1

    def test_coco_segmentation_not_empty(self, kidney_root):
        for e in _entries(kidney_root):
            segs = e.source_meta["coco_segmentation"]
            assert len(segs) == len(e.instances)
            for seg in segs:
                assert seg is not None and len(seg) > 0


class TestNormalKidneyCVManifest:

    def test_sample_ids_unique(self, kidney_root):
        ids = [e.sample_id for e in _entries(kidney_root)]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, kidney_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "kidney.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("Normal-Kidney-CV", kidney_root, writer)
        assert count == 3
        entries = load_manifest(out)
        assert all(e.dataset_id == "Normal-Kidney-CV" for e in entries)
