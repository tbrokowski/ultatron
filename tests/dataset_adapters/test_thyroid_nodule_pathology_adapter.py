"""
tests/dataset_adapters/test_thyroid_nodule_pathology_adapter.py
===============================================================
Self-contained synthetic fixture — no real data needed.

Run with:
    PYTHONPATH=/path/to/ultatron pytest tests/dataset_adapters/test_thyroid_nodule_pathology_adapter.py -v
"""
from __future__ import annotations

import csv
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def tnp_root(tmp_path_factory):
    """
    Synthetic layout mirroring 26067475/:

        dataset/
          patient_a_001.Jpg   (benign,  patient_name=patient_a)
          patient_b_001.Jpg   (malignant, patient_name=patient_b)
          patient_c_001.Jpg   (no label)
        batch1_image.csv
        batch1_image_label.csv
        batch2_image/
          thyroid_3_10_month/
            701/  img_701_001.Jpg   (benign)
            702/  img_702_001.Jpg   (malignant)
            703/  img_703_001.Jpg   (no label)
          batch2_image.csv
          batch2_image_label.csv
    """
    root = tmp_path_factory.mktemp("TNP") / "26067475"

    # ── Batch 1 ──
    img_dir = root / "dataset"
    img_dir.mkdir(parents=True)
    b1_files = {
        "patient_a": "patient_a_001.Jpg",
        "patient_b": "patient_b_001.Jpg",
        "patient_c": "patient_c_001.Jpg",
    }
    for fname in b1_files.values():
        (img_dir / fname).write_bytes(b"\x00")

    _write_csv(root / "batch1_image.csv", [
        {"patient_name": "patient_a", "path": "patient_a_001.Jpg"},
        {"patient_name": "patient_b", "path": "patient_b_001.Jpg"},
        {"patient_name": "patient_c", "path": "patient_c_001.Jpg"},
    ])
    _write_csv(root / "batch1_image_label.csv", [
        {"patient_index": "1", "patient_name": "patient_a", "histo_label": "0"},
        {"patient_index": "2", "patient_name": "patient_b", "histo_label": "1"},
        # patient_c intentionally omitted → ssl_only
    ])

    # ── Batch 2 ──
    b2_dir = root / "batch2_image"
    sub    = b2_dir / "thyroid_3_10_month"
    for pid in ("701", "702", "703"):
        d = sub / pid
        d.mkdir(parents=True)
        (d / f"img_{pid}_001.Jpg").write_bytes(b"\x00")

    _write_csv(b2_dir / "batch2_image.csv", [
        {"path": f"thyroid_3_10_month/{pid}/img_{pid}_001.Jpg",
         "patient_name": f"pat_{pid}"}
        for pid in ("701", "702", "703")
    ])
    _write_csv(b2_dir / "batch2_image_label.csv", [
        {"patient_index": "701", "patient_name": "pat_701", "histo_label": "0"},
        {"patient_index": "702", "patient_name": "pat_702", "histo_label": "1"},
        # 703 omitted → ssl_only
    ])

    # Adapter root is the *parent* of 26067475/
    return root.parent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestThyroidNodulePathologyAdapter:

    def test_import(self):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        assert ThyroidNodulePathologyAdapter.DATASET_ID     == "thyroid-nodule-pathology"
        assert ThyroidNodulePathologyAdapter.ANATOMY_FAMILY == "thyroid"
        assert ThyroidNodulePathologyAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "thyroid-nodule-pathology" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        entries = list(ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries())
        # batch1: 3 images, batch2: 3 images → 6 total
        assert len(entries) == 6

    def test_entry_schema(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id         == "thyroid-nodule-pathology"
            assert e.anatomy_family     in ANATOMY_FAMILIES
            assert e.modality_type      == "image"
            assert e.ssl_stream         == "image"
            assert e.split              == "train"
            assert e.has_mask           is False
            assert e.has_box            is False
            assert e.has_temporal_order is False
            assert e.num_frames         == 1
            assert e.curriculum_tier    in {1, 2, 3}

    def test_labelled_entries_have_classification_task(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        entries = list(ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries())
        labelled = [e for e in entries if e.task_type == "classification"]
        # patient_a + patient_b (batch1) + 701 + 702 (batch2) = 4
        assert len(labelled) == 4

    def test_unlabelled_entries_are_ssl_only(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        entries = list(ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries())
        ssl = [e for e in entries if e.task_type == "ssl_only"]
        # patient_c (batch1) + 703 (batch2) = 2
        assert len(ssl) == 2
        for e in ssl:
            assert e.instances == []
            assert e.label_raw is None

    def test_label_raw_benign_malignant(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        entries = {e.study_id: e
                   for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries()
                   if e.task_type == "classification"}
        # batch1
        assert entries["patient_a_001"].label_raw == ["benign"]
        assert entries["patient_b_001"].label_raw == ["malignant"]
        # batch2
        assert entries["701"].label_raw == ["benign"]
        assert entries["702"].label_raw == ["malignant"]

    def test_instance_label_ontology(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries():
            for inst in e.instances:
                assert inst.label_ontology == "thyroid_nodule_class"
                assert inst.label_raw in ("benign", "malignant")
                assert inst.classification_label in (0, 1)

    def test_batch1_study_id_is_stem(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        entries = [e for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries()
                   if e.source_meta.get("batch") == "batch1"]
        for e in entries:
            assert e.study_id == Path(e.image_paths[0]).stem

    def test_batch2_study_id_is_patient_index(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        entries = [e for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries()
                   if e.source_meta.get("batch") == "batch2"]
        for e in entries:
            assert e.study_id == e.source_meta["patient_index"]

    def test_source_meta_fields_batch1(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries():
            if e.source_meta.get("batch") == "batch1":
                assert "patient_name" in e.source_meta
                assert "histo_label"  in e.source_meta

    def test_source_meta_fields_batch2(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries():
            if e.source_meta.get("batch") == "batch2":
                assert "patient_index" in e.source_meta
                assert "rel_path"      in e.source_meta
                assert "histo_label"   in e.source_meta

    def test_all_splits_are_train(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries():
            assert e.split == "train"

    def test_split_override(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        for e in ThyroidNodulePathologyAdapter(root=tnp_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, tnp_root):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        ids = [e.sample_id for e in ThyroidNodulePathologyAdapter(root=tnp_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_missing_image_skipped(self, tmp_path):
        from data.adapters.thyroid.thyroid_nodule_pathology import ThyroidNodulePathologyAdapter
        root = tmp_path / "26067475"
        img_dir = root / "dataset"
        img_dir.mkdir(parents=True)
        _write_csv(root / "batch1_image.csv", [
            {"patient_name": "ghost", "path": "ghost_001.Jpg"},  # file absent
        ])
        _write_csv(root / "batch1_image_label.csv", [
            {"patient_index": "99", "patient_name": "ghost", "histo_label": "0"},
        ])
        entries = list(ThyroidNodulePathologyAdapter(root=tmp_path).iter_entries())
        assert entries == []

    def test_build_manifest_for_dataset(self, tnp_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "tnp.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("thyroid-nodule-pathology", tnp_root, writer)
        assert count == 6
        entries = load_manifest(out)
        assert all(e.dataset_id == "thyroid-nodule-pathology" for e in entries)
