"""
tests/dataset_adapters/test_gist514_db_adapters.py
===================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_gist514_db_adapters.py -v
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path


def _coco_json(images, annotations, categories):
    return json.dumps({
        "info":        {"description": "GIST514-DB synthetic"},
        "images":      images,
        "annotations": annotations,
        "categories":  categories,
    })


@pytest.fixture(scope="module")
def gist514_root(tmp_path_factory):
    """
    Synthetic GIST514-DB layout (COCO format):
      images/
        train/  img_0001.jpg, img_0002.jpg
        val/    img_0003.jpg
        test/   img_0004.jpg
      annotations/
        instances_train.json
        instances_val.json
        instances_test.json
    """
    root = tmp_path_factory.mktemp("GIST514DB")

    cats = [
        {"id": 1, "name": "gist",      "supercategory": "lesion"},
        {"id": 2, "name": "leiomyoma", "supercategory": "lesion"},
    ]

    splits = {
        "train": {
            "images": [
                {"id": 1, "file_name": "img_0001.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "img_0002.jpg", "width": 640, "height": 480,
                 "anatomical_location": "gastric_body"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 100, 80], "area": 8000},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [50, 60, 120, 90], "area": 10800},
            ],
        },
        "val": {
            "images": [{"id": 3, "file_name": "img_0003.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 3, "image_id": 3, "category_id": 1, "bbox": [30, 40, 80, 60], "area": 4800},
            ],
        },
        "test": {
            "images": [{"id": 4, "file_name": "img_0004.jpg", "width": 640, "height": 480}],
            "annotations": [
                {"id": 4, "image_id": 4, "category_id": 1, "bbox": [20, 30, 90, 70], "area": 6300},
            ],
        },
    }

    (root / "annotations").mkdir()
    for split_name, data in splits.items():
        # Create image files
        img_dir = root / "images" / split_name
        img_dir.mkdir(parents=True)
        for img in data["images"]:
            (img_dir / img["file_name"]).write_bytes(b"\x89PNG")
        # Create annotation JSON
        ann_path = root / "annotations" / f"instances_{split_name}.json"
        ann_path.write_text(_coco_json(data["images"], data["annotations"], cats))

    return root


class TestGIST514DBAdapter:

    def test_import(self):
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        assert GIST514DBAdapter.DATASET_ID     == "GIST514-DB"
        assert GIST514DBAdapter.ANATOMY_FAMILY == "gallbladder"
        assert GIST514DBAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "GIST514-DB" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, gist514_root):
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        entries = list(GIST514DBAdapter(root=gist514_root).iter_entries())
        assert len(entries) == 4  # 2 train + 1 val + 1 test

    def test_entry_schema(self, gist514_root):
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in GIST514DBAdapter(root=gist514_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "GIST514-DB"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.probe_type      == "radial"

    def test_detection_task(self, gist514_root):
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        for e in GIST514DBAdapter(root=gist514_root).iter_entries():
            assert e.task_type   == "detection"
            assert e.has_box     is True
            assert len(e.instances) >= 1

    def test_bbox_xyxy_format(self, gist514_root):
        """COCO [x,y,w,h] must be converted to [xmin,ymin,xmax,ymax]."""
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        for e in GIST514DBAdapter(root=gist514_root).iter_entries():
            for inst in e.instances:
                assert inst.bbox_xyxy is not None
                xmin, ymin, xmax, ymax = inst.bbox_xyxy
                assert xmax > xmin
                assert ymax > ymin

    def test_label_raw_and_ontology(self, gist514_root):
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        entries = list(GIST514DBAdapter(root=gist514_root).iter_entries())
        label_raws = {inst.label_raw for e in entries for inst in e.instances}
        assert "gist"      in label_raws
        assert "leiomyoma" in label_raws
        for e in entries:
            for inst in e.instances:
                assert inst.label_ontology in {"gist", "leiomyoma", "gi_lesion"}

    def test_anatomical_location_in_meta(self, gist514_root):
        """Image with anatomical_location set should carry it in source_meta."""
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        entries = list(GIST514DBAdapter(root=gist514_root, split_override="train").iter_entries())
        locs = [e.source_meta.get("anatomy_loc") for e in entries]
        assert any(loc is not None for loc in locs)

    def test_splits_respected(self, gist514_root):
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        entries = list(GIST514DBAdapter(root=gist514_root).iter_entries())
        splits  = {e.split for e in entries}
        assert splits == {"train", "val", "test"}

    def test_split_override(self, gist514_root):
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        for e in GIST514DBAdapter(root=gist514_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, gist514_root):
        from data.adapters.gallbladder.gist514_db import GIST514DBAdapter
        ids = [e.sample_id for e in GIST514DBAdapter(root=gist514_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, gist514_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "gist514.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("GIST514-DB", gist514_root, writer)
        assert count == 4
        entries = load_manifest(out)
        assert all(e.dataset_id == "GIST514-DB" for e in entries)
