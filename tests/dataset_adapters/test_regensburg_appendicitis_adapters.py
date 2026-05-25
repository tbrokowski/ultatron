"""
tests/dataset_adapters/test_regensburg_appendicitis_adapters.py
===============================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_regensburg_appendicitis_adapters.py -v
"""
from __future__ import annotations

import csv
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def regensburg_root(tmp_path_factory):
    """
    Synthetic layout:
      US_Pictures/
        1.1.bmp   1.2.bmp        (subject 1, 2 views)
        2.1.bmp                   (subject 2, 1 view)
        3.1.bmp   3.2.bmp        (subject 3, 2 views → test set)
      app_data.csv
      test_set_codes.csv
    """
    root   = tmp_path_factory.mktemp("RegensburgAppend")
    us_dir = root / "US_Pictures"
    us_dir.mkdir()

    for stem in ["1.1", "1.2", "2.1", "3.1", "3.2"]:
        (us_dir / f"{stem}.bmp").write_bytes(b"\x00" * 64)

    rows = [
        {"subject_id": "1", "diagnosis": "appendicitis",    "management": "surgical",     "severity": "complicated"},
        {"subject_id": "2", "diagnosis": "no_appendicitis", "management": "conservative", "severity": "no_appendicitis"},
        {"subject_id": "3", "diagnosis": "appendicitis",    "management": "surgical",     "severity": "uncomplicated"},
    ]
    with open(root / "app_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "diagnosis", "management", "severity"])
        w.writeheader()
        w.writerows(rows)

    (root / "test_set_codes.csv").write_text("3\n")
    return root


class TestRegensburgPediatricAppendicitisAdapter:

    def test_import(self):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        assert RegensburgPediatricAppendicitisAdapter.DATASET_ID     == "RegensburgPedAppend"
        assert RegensburgPediatricAppendicitisAdapter.ANATOMY_FAMILY == "abdomen"
        assert RegensburgPediatricAppendicitisAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "RegensburgPedAppend" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, regensburg_root):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        entries = list(RegensburgPediatricAppendicitisAdapter(root=regensburg_root).iter_entries())
        assert len(entries) == 5

    def test_entry_schema(self, regensburg_root):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in RegensburgPediatricAppendicitisAdapter(root=regensburg_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "RegensburgPedAppend"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.task_type       == "classification"
            assert e.has_mask        is False
            assert e.probe_type      == "curvilinear"

    def test_test_split_from_codes_csv(self, regensburg_root):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        entries  = list(RegensburgPediatricAppendicitisAdapter(root=regensburg_root).iter_entries())
        subject3 = [e for e in entries if e.source_meta["subject_id"] == "3"]
        assert len(subject3) == 2
        assert all(e.split == "test" for e in subject3)

    def test_diagnosis_labels(self, regensburg_root):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        entries    = list(RegensburgPediatricAppendicitisAdapter(root=regensburg_root).iter_entries())
        label_raws = {e.instances[0].label_raw for e in entries}
        assert "appendicitis"    in label_raws
        assert "no_appendicitis" in label_raws

    def test_one_instance_per_entry(self, regensburg_root):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        for e in RegensburgPediatricAppendicitisAdapter(root=regensburg_root).iter_entries():
            assert len(e.instances) == 1

    def test_source_meta_fields(self, regensburg_root):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        for e in RegensburgPediatricAppendicitisAdapter(root=regensburg_root).iter_entries():
            assert "subject_id"  in e.source_meta
            assert "view_idx"    in e.source_meta
            assert "diagnosis"   in e.source_meta
            assert "management"  in e.source_meta
            assert "severity"    in e.source_meta
            assert e.source_meta["doi"] == "https://doi.org/10.5281/zenodo.7711412"

    def test_split_override(self, regensburg_root):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        for e in RegensburgPediatricAppendicitisAdapter(
            root=regensburg_root, split_override="val"
        ).iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, regensburg_root):
        from data.adapters.gallbladder.regensburg_pediatric_appendicitis import (
            RegensburgPediatricAppendicitisAdapter,
        )
        ids = [
            e.sample_id
            for e in RegensburgPediatricAppendicitisAdapter(root=regensburg_root).iter_entries()
        ]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, regensburg_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "regensburg.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("RegensburgPedAppend", regensburg_root, writer)
        assert count == 5
        entries = load_manifest(out)
        assert all(e.dataset_id == "RegensburgPedAppend" for e in entries)
