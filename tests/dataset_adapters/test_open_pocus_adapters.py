"""
tests/dataset_adapters/test_open_pocus_adapters.py  ·  OpenPOCUSAdapter tests
==============================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_open_pocus_adapters.py -v
"""
from __future__ import annotations

import csv
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def open_pocus_root(tmp_path_factory):
    """
    Synthetic OpenPOCUS layout:
      clips/
        clip_001.mp4   (normal)
        clip_002.mp4   (b_lines)
        clip_003.mp4   (consolidation)
        clip_004.mp4   (indeterminate)
      metadata.csv
    """
    root = tmp_path_factory.mktemp("OpenPOCUS")
    (root / "clips").mkdir()

    rows = [
        {"filename": "clip_001.mp4", "patient_id": "P001", "lung_zone": "R1",
         "finding": "normal",        "covid_positive": "0", "device": "Butterfly"},
        {"filename": "clip_002.mp4", "patient_id": "P002", "lung_zone": "L2",
         "finding": "b_lines",       "covid_positive": "1", "device": "Philips"},
        {"filename": "clip_003.mp4", "patient_id": "P003", "lung_zone": "R3",
         "finding": "consolidation", "covid_positive": "0", "device": "GE"},
        {"filename": "clip_004.mp4", "patient_id": "P004", "lung_zone": "L4",
         "finding": "indeterminate", "covid_positive": "0", "device": "Butterfly"},
    ]
    for row in rows:
        (root / "clips" / row["filename"]).write_bytes(b"\x00" * 64)

    with open(root / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return root


@pytest.fixture(scope="module")
def open_pocus_root_no_csv(tmp_path_factory):
    """OpenPOCUS without metadata.csv — ssl_only fallback."""
    root = tmp_path_factory.mktemp("OpenPOCUS_nocsv")
    (root / "clips").mkdir()
    for i in range(1, 4):
        (root / "clips" / f"clip_{i:03d}.mp4").write_bytes(b"\x00" * 64)
    return root


class TestOpenPOCUSAdapter:

    def test_import(self):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        assert OpenPOCUSAdapter.DATASET_ID     == "OpenPOCUS"
        assert OpenPOCUSAdapter.ANATOMY_FAMILY == "lung"
        assert OpenPOCUSAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "OpenPOCUS" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, open_pocus_root):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        entries = list(OpenPOCUSAdapter(root=open_pocus_root).iter_entries())
        assert len(entries) == 4

    def test_entry_schema(self, open_pocus_root):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in OpenPOCUSAdapter(root=open_pocus_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "OpenPOCUS"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "video"
            assert e.ssl_stream      == "video"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.task_type       == "classification"
            assert e.has_mask        is False
            assert e.probe_type      == "phased_array"

    def test_finding_labels(self, open_pocus_root):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        entries    = list(OpenPOCUSAdapter(root=open_pocus_root).iter_entries())
        label_raws = {e.instances[0].label_raw for e in entries if e.instances}
        assert "normal"        in label_raws
        assert "b_lines"       in label_raws
        assert "consolidation" in label_raws

    def test_label_ontology(self, open_pocus_root):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        for e in OpenPOCUSAdapter(root=open_pocus_root).iter_entries():
            for inst in e.instances:
                assert inst.label_ontology in {
                    "lung_normal", "lung_b_lines",
                    "lung_consolidation", "lung_other",
                }

    def test_source_meta_fields(self, open_pocus_root):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        for e in OpenPOCUSAdapter(root=open_pocus_root).iter_entries():
            assert "patient_id"  in e.source_meta
            assert "lung_zone"   in e.source_meta
            assert "covid_pos"   in e.source_meta
            assert "raw_finding" in e.source_meta
            assert e.source_meta["doi"] == "https://doi.org/10.1101/2025.05.09.25327337"

    def test_ssl_only_without_csv(self, open_pocus_root_no_csv):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        entries = list(OpenPOCUSAdapter(root=open_pocus_root_no_csv).iter_entries())
        assert len(entries) == 3
        for e in entries:
            assert e.task_type == "ssl_only"
            assert e.instances == []

    def test_split_override(self, open_pocus_root):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        for e in OpenPOCUSAdapter(root=open_pocus_root, split_override="test").iter_entries():
            assert e.split == "test"

    def test_sample_ids_unique(self, open_pocus_root):
        from data.adapters.lung.open_pocus import OpenPOCUSAdapter
        ids = [e.sample_id for e in OpenPOCUSAdapter(root=open_pocus_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, open_pocus_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "open_pocus.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("OpenPOCUS", open_pocus_root, writer)
        assert count == 4
        entries = load_manifest(out)
        assert all(e.dataset_id == "OpenPOCUS" for e in entries)
