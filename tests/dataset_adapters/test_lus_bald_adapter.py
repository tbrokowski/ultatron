"""
tests/dataset_adapters/test_lus_bald_adapter.py  ·  LUSBALDAdapter tests
=========================================================================
Self-contained synthetic fixture — no real data needed.

Run with:
    PYTHONPATH=/path/to/ultatron pytest tests/dataset_adapters/test_lus_bald_adapter.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LABEL_A = "0 0.358025 0.246575 0.550617 0.227006 0.456790 1.000000 0.079012 0.943249\n"
_LABEL_B = (
    "0 0.1 0.2 0.3 0.2 0.3 0.8 0.1 0.8\n"
    "0 0.5 0.1 0.9 0.1 0.9 0.9 0.5 0.9\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lus_bald_root(tmp_path_factory):
    """
    Synthetic LUS-BALD layout:

        train/
          images/  img1.png  img2.jpeg
          labels/  img1.txt  img2.txt   (img2 has 2 annotations)
        val/
          images/  img_v1.png
          labels/  img_v1.txt           (1 annotation)
        test/
          images/  img_t1.png
          labels/  (no .txt → ssl_only)
    """
    root = tmp_path_factory.mktemp("LUS_BALD")

    layout = {
        "train/images": ["img1.png", "img2.jpeg"],
        "train/labels": [],
        "val/images":   ["img_v1.png"],
        "val/labels":   [],
        "test/images":  ["img_t1.png"],
        "test/labels":  [],
    }
    for rel in layout:
        (root / rel).mkdir(parents=True, exist_ok=True)

    for fname in ["img1.png", "img2.jpeg"]:
        (root / "train" / "images" / fname).write_bytes(b"\x00")
    (root / "train" / "labels" / "img1.txt").write_text(_LABEL_A)
    (root / "train" / "labels" / "img2.txt").write_text(_LABEL_B)

    (root / "val" / "images" / "img_v1.png").write_bytes(b"\x00")
    (root / "val" / "labels" / "img_v1.txt").write_text(_LABEL_A)

    (root / "test" / "images" / "img_t1.png").write_bytes(b"\x00")
    # no label file for img_t1 → ssl_only

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLUSBALDAdapter:

    def test_import(self):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        assert LUSBALDAdapter.DATASET_ID     == "LUS-BALD"
        assert LUSBALDAdapter.ANATOMY_FAMILY == "lung"
        assert LUSBALDAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "LUS-BALD" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        entries = list(LUSBALDAdapter(root=lus_bald_root).iter_entries())
        # train: img1 + img2, val: img_v1, test: img_t1 → 4 total
        assert len(entries) == 4

    def test_entry_schema(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in LUSBALDAdapter(root=lus_bald_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "LUS-BALD"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.has_mask        is False
            assert e.has_temporal_order is False
            assert e.num_frames      == 1
            assert e.curriculum_tier in {1, 2, 3}

    def test_detection_entries_have_box(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        entries = {Path(e.image_paths[0]).stem: e
                   for e in LUSBALDAdapter(root=lus_bald_root).iter_entries()}
        for stem in ("img1", "img2", "img_v1"):
            assert entries[stem].has_box     is True
            assert entries[stem].task_type   == "detection"
            assert entries[stem].label_raw   == ["b_line"]

    def test_ssl_only_when_no_label_file(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        entries = {Path(e.image_paths[0]).stem: e
                   for e in LUSBALDAdapter(root=lus_bald_root).iter_entries()}
        e = entries["img_t1"]
        assert e.task_type  == "ssl_only"
        assert e.has_box    is False
        assert e.instances  == []
        assert e.label_raw  is None

    def test_instance_count(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        entries = {Path(e.image_paths[0]).stem: e
                   for e in LUSBALDAdapter(root=lus_bald_root).iter_entries()}
        assert len(entries["img1"].instances)  == 1
        assert len(entries["img2"].instances)  == 2
        assert len(entries["img_v1"].instances) == 1

    def test_instance_label_raw(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        for e in LUSBALDAdapter(root=lus_bald_root).iter_entries():
            for inst in e.instances:
                assert inst.label_raw      == "b_line"
                assert inst.label_ontology == "b_line"

    def test_polygon_stored_on_instance(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        entries = {Path(e.image_paths[0]).stem: e
                   for e in LUSBALDAdapter(root=lus_bald_root).iter_entries()}
        inst = entries["img1"].instances[0]
        assert inst.polygon is not None
        assert len(inst.polygon) == 4
        for pt in inst.polygon:
            assert len(pt) == 2

    def test_bbox_xyxy_derived_from_polygon(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        entries = {Path(e.image_paths[0]).stem: e
                   for e in LUSBALDAdapter(root=lus_bald_root).iter_entries()}
        inst = entries["img1"].instances[0]
        # polygon from _LABEL_A: (0.358,0.247),(0.551,0.227),(0.457,1.0),(0.079,0.943)
        xs = [p[0] for p in inst.polygon]
        ys = [p[1] for p in inst.polygon]
        assert inst.bbox_xyxy == pytest.approx(
            [min(xs), min(ys), max(xs), max(ys)], rel=1e-5
        )

    def test_study_id_is_filename_stem(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        for e in LUSBALDAdapter(root=lus_bald_root).iter_entries():
            assert e.study_id == Path(e.image_paths[0]).stem

    def test_label_path_in_source_meta(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        for e in LUSBALDAdapter(root=lus_bald_root).iter_entries():
            assert "label_path" in e.source_meta

    def test_split_values(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        entries = list(LUSBALDAdapter(root=lus_bald_root).iter_entries())
        splits = {e.split for e in entries}
        assert splits == {"train", "val", "test"}

    def test_split_override(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        for e in LUSBALDAdapter(root=lus_bald_root, split_override="test").iter_entries():
            assert e.split == "test"

    def test_sample_ids_unique(self, lus_bald_root):
        from data.adapters.lung.lus_bald import LUSBALDAdapter
        ids = [e.sample_id for e in LUSBALDAdapter(root=lus_bald_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, lus_bald_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "lus_bald.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("LUS-BALD", lus_bald_root, writer)
        assert count == 4
        entries = load_manifest(out)
        assert all(e.dataset_id == "LUS-BALD" for e in entries)
