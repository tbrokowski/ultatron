"""
tests/dataset_adapters/test_knee_us_jocohs_adapters.py
=======================================================
Self-contained synthetic fixture matching real JoCoHS layout.

Run with:
    PYTHONPATH=. pytest tests/dataset_adapters/test_knee_us_jocohs_adapters.py -v
"""
from __future__ import annotations

import csv
import pytest
from pathlib import Path


def _write_jocohs_fixture(root: Path, n_subjects: int = 6) -> None:
    """Build data/reference + data/image/ultrasound/imageArchive.* layout."""
    ref_dir = root / "data" / "reference"
    ref_dir.mkdir(parents=True)

    subjects = []
    image_rows = []
    splits = ["train", "train", "train", "train", "val", "test"]

    for i in range(1, n_subjects + 1):
        sid = f"subj-{i:04d}"
        archive = str(10 + i)  # imageArchive.11, .12, ...
        filename = f"{sid}_{i}.png"
        img_dir = root / "data" / "image" / "ultrasound" / f"imageArchive.{archive}"
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / filename).write_bytes(b"\x89PNG")

        image_rows.append({
            "E03SUBJECTID": sid,
            "E03USIMGT": archive,
            "E03USIMGF": filename,
            "E03USIMGZ": "1000",
            "E03USIMGD": "2",
        })
        subjects.append({
            "E03SUBJECTID": sid,
            "E03GENDER": "M",
            "E03PASKR": str(i % 6),
            "E03PASKL": str((i + 1) % 6),
            "E03AGE": "50",
        })

    with open(ref_dir / "dataTable.IMAGE_REF.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(image_rows)

    with open(ref_dir / "dataTable.SUBJECT.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(subjects[0].keys()), quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(subjects)


@pytest.fixture(scope="module")
def knee_us_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("KneeUS")
    _write_jocohs_fixture(root, n_subjects=6)
    return root


@pytest.fixture(scope="module")
def knee_us_root_no_csv(tmp_path_factory):
    root = tmp_path_factory.mktemp("KneeUS_nocsv")
    img_dir = root / "data" / "image" / "ultrasound" / "imageArchive.11"
    img_dir.mkdir(parents=True)
    for i in range(1, 4):
        (img_dir / f"orphan_{i}.png").write_bytes(b"\x89PNG")
    return root


class TestKneeUSJoCoHSAdapter:

    def test_import(self):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
        assert KneeUSJoCoHSAdapter.DATASET_ID     == "KneeUSJoCoHS"
        assert KneeUSJoCoHSAdapter.ANATOMY_FAMILY == "joint"
        assert KneeUSJoCoHSAdapter.SONODQS        == "gold"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "KneeUSJoCoHS" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
        entries = list(KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries())
        assert len(entries) == 6

    def test_entry_schema(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
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
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
        for e in KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries():
            assert e.task_type == "weak_label"

    def test_oa_feature_instances(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
        entries = list(KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries())
        entries_with_instances = [e for e in entries if len(e.instances) > 0]
        assert len(entries_with_instances) > 0
        for e in entries_with_instances:
            for inst in e.instances:
                assert inst.label_ontology == "knee_oa_feature"
                assert inst.label_raw.startswith("knee_")
                assert inst.mask_path      is None
                assert inst.is_promptable  is False

    def test_split_from_subject_hash(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
        entries = list(KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries())
        splits = {e.split for e in entries}
        assert "train" in splits

    def test_split_override(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
        for e in KneeUSJoCoHSAdapter(root=knee_us_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_source_meta(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
        for e in KneeUSJoCoHSAdapter(root=knee_us_root).iter_entries():
            assert "subject_id" in e.source_meta
            assert "archive_type" in e.source_meta
            assert e.source_meta["doi"] == "https://doi.org/10.7910/DVN/SKP9IB"

    def test_ssl_only_without_csv(self, knee_us_root_no_csv):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
        entries = list(KneeUSJoCoHSAdapter(root=knee_us_root_no_csv).iter_entries())
        assert len(entries) == 0

    def test_sample_ids_unique(self, knee_us_root):
        from data.adapters.muscle.knee_us_jocohs import KneeUSJoCoHSAdapter
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
