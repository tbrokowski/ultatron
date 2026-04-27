"""
tests/dataset_adapters/test_busv.py  ·  BUSVAdapter contract tests
===================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_busv.py -v
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def busv_root(tmp_path_factory):
    """
    Synthetic BUSV layout:
      rawframes/benign/abc123/    → 5 frames
      rawframes/benign/def456/    → 4 frames
      rawframes/malignant/ghi789/ → 6 frames
      imagenet_vid_train_15frames.json  → ["abc123", "ghi789"]
      imagenet_vid_val.json             → ["def456"]
    """
    root = tmp_path_factory.mktemp("BUSV")

    clips = [
        ("benign",    "abc123", 5),
        ("benign",    "def456", 4),
        ("malignant", "ghi789", 6),
    ]

    for cls, vid_id, n_frames in clips:
        clip_dir = root / "rawframes" / cls / vid_id
        clip_dir.mkdir(parents=True)
        for i in range(n_frames):
            (clip_dir / f"{i:06d}.png").write_bytes(b"\x89PNG")

    # Write split JSON files
    (root / "imagenet_vid_train_15frames.json").write_text(
        json.dumps(["abc123", "ghi789"])
    )
    (root / "imagenet_vid_val.json").write_text(
        json.dumps(["def456"])
    )

    return root


class TestBUSVAdapter:

    def test_import(self):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        assert BUSVAdapter.DATASET_ID     == "BUSV"
        assert BUSVAdapter.ANATOMY_FAMILY == "breast"
        assert BUSVAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "BUSV" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, busv_root):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        entries = list(BUSVAdapter(root=busv_root).iter_entries())
        assert len(entries) == 3  # 2 benign + 1 malignant

    def test_entry_schema(self, busv_root):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in BUSVAdapter(root=busv_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "BUSV"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "video"
            assert e.ssl_stream      == "video"
            assert e.is_cine         is True
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}

    def test_frame_paths(self, busv_root):
        """Each entry should have multiple frame paths in order."""
        from data.adapters.breast.busv_adapter import BUSVAdapter
        for e in BUSVAdapter(root=busv_root).iter_entries():
            assert len(e.image_paths) > 1
            # Frames should be sorted
            assert e.image_paths == sorted(e.image_paths)

    def test_num_frames(self, busv_root):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        entries = {
            e.source_meta["video_id"]: e
            for e in BUSVAdapter(root=busv_root).iter_entries()
        }
        assert entries["abc123"].num_frames == 5
        assert entries["def456"].num_frames == 4
        assert entries["ghi789"].num_frames == 6

    def test_split_from_json(self, busv_root):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        entries = {
            e.source_meta["video_id"]: e
            for e in BUSVAdapter(root=busv_root).iter_entries()
        }
        assert entries["abc123"].split == "train"
        assert entries["def456"].split == "val"
        assert entries["ghi789"].split == "train"

    def test_label_ontology(self, busv_root):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        ontologies = {
            e.instances[0].label_ontology
            for e in BUSVAdapter(root=busv_root).iter_entries()
        }
        assert "breast_lesion_benign"    in ontologies
        assert "breast_lesion_malignant" in ontologies

    def test_no_mask(self, busv_root):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        for e in BUSVAdapter(root=busv_root).iter_entries():
            assert not e.has_mask
            assert e.task_type == "classification"

    def test_split_override(self, busv_root):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        for e in BUSVAdapter(root=busv_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, busv_root):
        from data.adapters.breast.busv_adapter import BUSVAdapter
        ids = [e.sample_id for e in BUSVAdapter(root=busv_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, busv_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "busv.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("BUSV", busv_root, writer)
        assert count == 3
        entries = load_manifest(out)
        assert all(e.dataset_id == "BUSV" for e in entries)
