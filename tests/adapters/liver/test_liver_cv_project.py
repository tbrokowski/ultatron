"""
tests/adapters/liver/test_liver_cv_project.py
=============================================

Unit tests for LiverCVProjectAdapter.

Run with:
    pytest tests/adapters/liver/test_liver_cv_project.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ── Fixture ───────────────────────────────────────────────────────────────────

# Synthetic dataset layout
#   train/: img_001 (2 anns: HCC + HV), img_002 (1 ann: LVR), img_003 (no anns)
#   valid/: img_004 (1 ann: PV)
#   test/:  img_005 (1 ann: NO HCC)

_CATEGORIES = [
    {"id": 1, "name": "HCC"},
    {"id": 2, "name": "HV"},
    {"id": 3, "name": "IVC"},
    {"id": 4, "name": "K"},
    {"id": 5, "name": "K-M"},
    {"id": 6, "name": "LVR"},
    {"id": 7, "name": "NO HCC"},
    {"id": 8, "name": "PV"},
    {"id": 9, "name": "SAG"},
    {"id": 10, "name": "TRV"},
]

_SPLITS_SPEC = {
    "train": {
        "images": [
            {"id": 1, "file_name": "img_001.jpg", "width": 299, "height": 299},
            {"id": 2, "file_name": "img_002.jpg", "width": 299, "height": 299},
            {"id": 3, "file_name": "img_003.jpg", "width": 299, "height": 299},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10.0, 20.0, 50.0, 60.0], "segmentation": []},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100.0, 80.0, 40.0, 30.0], "segmentation": []},
            {"id": 3, "image_id": 2, "category_id": 6, "bbox": [5.0, 5.0, 290.0, 290.0], "segmentation": []},
            # image_id 3 intentionally has no annotations
        ],
    },
    "valid": {
        "images": [
            {"id": 10, "file_name": "img_004.jpg", "width": 299, "height": 299},
        ],
        "annotations": [
            {"id": 10, "image_id": 10, "category_id": 8, "bbox": [30.0, 40.0, 60.0, 70.0], "segmentation": []},
        ],
    },
    "test": {
        "images": [
            {"id": 20, "file_name": "img_005.jpg", "width": 299, "height": 299},
        ],
        "annotations": [
            {"id": 20, "image_id": 20, "category_id": 7, "bbox": [0.0, 0.0, 299.0, 299.0], "segmentation": []},
        ],
    },
}


def _save_jpg(path: Path, h: int = 8, w: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    if PIL_OK:
        PILImage.fromarray(arr).save(str(path), format="JPEG")
    else:
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)


def _build_liver_cv(root: Path) -> None:
    for dir_name, spec in _SPLITS_SPEC.items():
        split_dir = root / dir_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for img_info in spec["images"]:
            _save_jpg(split_dir / img_info["file_name"])

        coco = {
            "images":      spec["images"],
            "annotations": spec["annotations"],
            "categories":  _CATEGORIES,
        }
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(coco), encoding="utf-8"
        )


@pytest.fixture(scope="module")
def liver_cv_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("liver_cv")
    _build_liver_cv(root)
    return root


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entries(root, **kwargs):
    from data.adapters.liver.liver_cv_project import LiverCVProjectAdapter
    return list(LiverCVProjectAdapter(root, **kwargs).iter_entries())


def _by_file(root, **kwargs):
    return {Path(e.image_paths[0]).name: e for e in _entries(root, **kwargs)}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLiverCVProjectMeta:

    def test_class_attributes(self):
        from data.adapters.liver.liver_cv_project import LiverCVProjectAdapter
        assert LiverCVProjectAdapter.DATASET_ID     == "liver-CV-project"
        assert LiverCVProjectAdapter.ANATOMY_FAMILY == "liver"
        assert LiverCVProjectAdapter.SONODQS        == "silver"
        assert LiverCVProjectAdapter.DOI            == "https://public.roboflow.ai/object-detection/undefined"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "liver-CV-project" in ADAPTER_REGISTRY

    def test_missing_splits_raises(self, tmp_path):
        from data.adapters.liver.liver_cv_project import LiverCVProjectAdapter
        with pytest.raises(FileNotFoundError, match="liver-CV-project"):
            LiverCVProjectAdapter(tmp_path)

    def test_resolve_nested_root(self, tmp_path):
        """Adapter must work when passed the parent of liver_ultrasound.v11i.coco/."""
        from data.adapters.liver.liver_cv_project import LiverCVProjectAdapter
        inner = tmp_path / "liver_ultrasound.v11i.coco"
        _build_liver_cv(inner)
        entries = list(LiverCVProjectAdapter(tmp_path).iter_entries())
        assert len(entries) == 5


class TestLiverCVProjectEntryCount:

    def test_total_entries(self, liver_cv_root):
        assert len(_entries(liver_cv_root)) == 5   # 3 train + 1 val + 1 test

    def test_entries_per_split(self, liver_cv_root):
        entries = _entries(liver_cv_root)
        assert sum(1 for e in entries if e.split == "train") == 3
        assert sum(1 for e in entries if e.split == "val")   == 1
        assert sum(1 for e in entries if e.split == "test")  == 1


class TestLiverCVProjectSchema:

    def test_entry_fields(self, liver_cv_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _entries(liver_cv_root):
            assert e.dataset_id     == "liver-CV-project"
            assert e.anatomy_family == "liver"
            assert e.anatomy_family in ANATOMY_FAMILIES
            assert e.modality_type  == "image"
            assert e.view_type      == "liver_bmode"
            assert e.ssl_stream     == "image"
            assert e.has_mask       is False
            assert e.curriculum_tier in (1, 2, 3)
            assert e.study_id       == e.series_id
            assert len(e.image_paths) == 1

    def test_image_dimensions_set(self, liver_cv_root):
        for e in _entries(liver_cv_root):
            assert e.width  == 299
            assert e.height == 299

    def test_split_values(self, liver_cv_root):
        for e in _entries(liver_cv_root):
            assert e.split in ("train", "val", "test")

    def test_valid_folder_maps_to_val(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        assert by_file["img_004.jpg"].split == "val"

    def test_missing_image_file_skipped(self, tmp_path):
        """An image listed in the COCO JSON but absent on disk must be skipped."""
        from data.adapters.liver.liver_cv_project import LiverCVProjectAdapter
        _build_liver_cv(tmp_path)
        # Remove one image file
        (tmp_path / "train" / "img_001.jpg").unlink()
        entries = list(LiverCVProjectAdapter(tmp_path).iter_entries())
        names = {Path(e.image_paths[0]).name for e in entries}
        assert "img_001.jpg" not in names


class TestLiverCVProjectDetection:

    def test_annotated_image_is_detection(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        for name in ("img_001.jpg", "img_002.jpg", "img_004.jpg", "img_005.jpg"):
            e = by_file[name]
            assert e.task_type     == "detection"
            assert e.has_box       is True
            assert e.is_promptable is True

    def test_unannotated_image_is_ssl_only(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        e = by_file["img_003.jpg"]
        assert e.task_type     == "ssl_only"
        assert e.has_box       is False
        assert e.is_promptable is False
        assert e.instances     == []

    def test_instance_count_per_image(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        assert len(by_file["img_001.jpg"].instances) == 2
        assert len(by_file["img_002.jpg"].instances) == 1
        assert len(by_file["img_004.jpg"].instances) == 1


class TestLiverCVProjectBbox:

    def test_bbox_xyxy_conversion(self, liver_cv_root):
        """COCO [x,y,w,h] must be converted to [x1,y1,x2,y2]."""
        by_file = _by_file(liver_cv_root)
        # img_001 ann id=1: bbox=[10, 20, 50, 60] → xyxy=[10, 20, 60, 80]
        inst = next(
            i for i in by_file["img_001.jpg"].instances
            if i.label_raw == "HCC"
        )
        assert inst.bbox_xyxy == [10.0, 20.0, 60.0, 80.0]

    def test_bbox_xyxy_second_annotation(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        # img_001 ann id=2: bbox=[100, 80, 40, 30] → xyxy=[100, 80, 140, 110]
        inst = next(
            i for i in by_file["img_001.jpg"].instances
            if i.label_raw == "HV"
        )
        assert inst.bbox_xyxy == [100.0, 80.0, 140.0, 110.0]

    def test_all_instances_have_bbox(self, liver_cv_root):
        for e in _entries(liver_cv_root):
            for inst in e.instances:
                assert inst.bbox_xyxy is not None
                assert len(inst.bbox_xyxy) == 4
                x1, y1, x2, y2 = inst.bbox_xyxy
                assert x2 > x1 and y2 > y1

    def test_no_mask_path_on_instances(self, liver_cv_root):
        for e in _entries(liver_cv_root):
            for inst in e.instances:
                assert inst.mask_path is None


class TestLiverCVProjectOntology:

    def test_hcc_maps_to_liver_lesion(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        inst = next(i for i in by_file["img_001.jpg"].instances if i.label_raw == "HCC")
        assert inst.label_ontology == "liver_lesion"

    def test_hv_maps_to_hepatic_vein(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        inst = next(i for i in by_file["img_001.jpg"].instances if i.label_raw == "HV")
        assert inst.label_ontology == "hepatic_vein"

    def test_lvr_maps_to_liver_parenchyma(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        inst = by_file["img_002.jpg"].instances[0]
        assert inst.label_raw      == "LVR"
        assert inst.label_ontology == "liver_parenchyma"

    def test_pv_maps_to_portal_vein(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        inst = by_file["img_004.jpg"].instances[0]
        assert inst.label_raw      == "PV"
        assert inst.label_ontology == "portal_vein"

    def test_no_hcc_maps_to_liver_parenchyma(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        inst = by_file["img_005.jpg"].instances[0]
        assert inst.label_raw      == "NO HCC"
        assert inst.label_ontology == "liver_parenchyma"

    def test_all_instances_are_promptable(self, liver_cv_root):
        for e in _entries(liver_cv_root):
            for inst in e.instances:
                assert inst.is_promptable is True


class TestLiverCVProjectSourceMeta:

    def test_required_keys(self, liver_cv_root):
        for e in _entries(liver_cv_root):
            assert "coco_image_id"  in e.source_meta
            assert "file_name"      in e.source_meta
            assert "n_annotations"  in e.source_meta

    def test_n_annotations_matches_instances(self, liver_cv_root):
        for e in _entries(liver_cv_root):
            assert e.source_meta["n_annotations"] == len(e.instances)

    def test_file_name_matches_image_path(self, liver_cv_root):
        for e in _entries(liver_cv_root):
            assert e.source_meta["file_name"] == Path(e.image_paths[0]).name


class TestLiverCVProjectSplit:

    def test_split_override(self, liver_cv_root):
        entries = _entries(liver_cv_root, split_override="val")
        assert all(e.split == "val" for e in entries)

    def test_no_override_uses_folder_splits(self, liver_cv_root):
        by_file = _by_file(liver_cv_root)
        assert by_file["img_001.jpg"].split == "train"
        assert by_file["img_002.jpg"].split == "train"
        assert by_file["img_003.jpg"].split == "train"
        assert by_file["img_004.jpg"].split == "val"
        assert by_file["img_005.jpg"].split == "test"
