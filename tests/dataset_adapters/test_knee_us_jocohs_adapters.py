"""
tests/dataset_adapters/test_knee_us_jocohs_adapters.py
=======================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_knee_us_jocohs_adapters.py -v
"""
from __future__ import annotations

import csv
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def knee_us_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("KneeUS")
    (root / "images").mkdir()
    rows = []
    for i in range(1, 7):
        stem = f"img_{i:06d}"
        (root / "images" / f"{stem}.png").write_bytes(b"\x89PNG")
        rows.append({
            "image_id":             stem,
            "participant_id":       f"P{i:04d}",
            "knee_side":            "left" if i % 2 == 0 else "right",
            "view":                 "suprapatellar",
            "effusion":             str(i % 3),
            "synovial_hypertrophy": str((i + 1) % 3),
            "osteophyte_medial":    str(i % 2),
            "cartilage_damage":     "0",
            "meniscal_extrusion":   str(i % 2),
            "popliteal_cyst":       "0",
            "split":                ["train","train","train","train","val","test"][i - 1],
        })
    with open(root / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return root


@pytest.fixture(scope="module")
def knee_us_root_no_csv(tmp_path_factory):
    root = tmp_path_factory.mktemp("KneeUS_nocsv")
    (root / "images").mkdir()
    for i in range(1, 4):
        (root / "images" / f"img_{i:06d}.png").write_bytes(b"\x89PNG")
    return root


class TestKneeUSJoCoHSAdapter:

    def test_import(self):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        assert KneeUSJoCoHSAdapter.DATASET_ID     == "KneeUSJoCoHS"
        assert KneeUSJoCoHSAdapter.ANATOMY_FAMILY == "joint"
        assert KneeUSJoCoHSAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "KneeUSJoCoHS" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        entries = list(KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries())
        assert len(entries) == 6

    def test_entry_schema(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "KneeUSJoCoHS"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.has_mask        is False
            assert e.probe_type      == "linear"

    def test_task_type_weak_label(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        for e in KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries():
            assert e.task_type == "weak_label"

    def test_oa_feature_instances(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        entries = list(KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries())
        entries_with_instances = [e for e in entries if len(e.instances) > 0]
        assert len(entries_with_instances) > 0
        for e in entries_with_instances:
            for inst in e.instances:
                assert inst.label_ontology == "knee_oa_feature"
                assert inst.label_raw.startswith("knee_")
                assert inst.mask_path      is None
                assert inst.is_promptable  is False

    def test_split_from_csv(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        entries = list(KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries())
        splits = {e.split for e in entries}
        assert "train" in splits
        assert "val"   in splits
        assert "test"  in splits

    def test_split_override(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        for e in KneeUSJoCoHSAdapter(root=knee_us_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_source_meta(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        for e in KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries():
            assert "participant_id" in e.source_meta
            assert "knee_side"      in e.source_meta
            assert e.source_meta["doi"] == "https://doi.org/10.7910/DVN/SKP9IB"

    def test_ssl_only_without_csv(self, knee_us_root_no_csv):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        entries = list(KneeUSJoCoHSAdapter(root=knee_us_root_no_csv).iter_entries())
        assert len(entries) == 3
        for e in entries:
            assert e.task_type == "ssl_only"
            assert e.instances == []

    def test_sample_ids_unique(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohS import KneeUSJoCoHSAdapter
        ids = [e.sample_id for e in KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, knee_us_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "knee_us.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("KneeUSJoCoHS", knee_us_root, writer)
        assert count == 6
        entries = load_manifest(out)
        assert all(e.dataset_id == "KneeUSJoCoHS" for e in entries)
