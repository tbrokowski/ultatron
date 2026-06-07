"""
tests/dataset_adapters/test_deep_mtj_adapters.py  ·  DeepMTJAdapter contract tests
====================================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_deep_mtj_adapters.py -v
"""
from __future__ import annotations

import csv
import pytest
from pathlib import Path


def _make_deepmtj_img(path: Path, stem: str):
    path.write_bytes(b"\x89PNG")


@pytest.fixture(scope="module")
def deepmtj_root(tmp_path_factory):
    """
    Synthetic deepMTJ layout:
      fullres/
        deepMTJ_TS_ffullres_f0001_annotated.jpg
        deepMTJ_TS_ffullres_f0002_annotated.jpg
        deepMTJ_TS_ffullres_f0003_annotated.jpg
      256x128px/
        deepMTJ_TS_f256x128px_f0001_annotated.jpg
        deepMTJ_TS_f256x128px_f0002_annotated.jpg
        deepMTJ_TS_f256x128px_f0003_annotated.jpg
      MTJ_Benchmark_Labels.csv  ← keypoints for fullres frames only
    """
    root = tmp_path_factory.mktemp("deepMTJ")
    for res_dir, prefix in [
        ("fullres",    "deepMTJ_TS_ffullres"),
        ("256x128px",  "deepMTJ_TS_f256x128px"),
    ]:
        (root / res_dir).mkdir()
        for i in range(1, 4):
            stem = f"{prefix}_f{i:04d}_annotated"
            (root / res_dir / f"{stem}.jpg").write_bytes(b"\x89PNG")

    # CSV with keypoints for fullres only
    rows = [
        {"filename": f"deepMTJ_TS_ffullres_f{i:04d}_annotated.jpg", "x": str(100 + i), "y": str(50 + i)}
        for i in range(1, 4)
    ]
    with open(root / "MTJ_Benchmark_Labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "x", "y"])
        w.writeheader()
        w.writerows(rows)
    return root


@pytest.fixture(scope="module")
def deepmtj_root_no_csv(tmp_path_factory):
    """deepMTJ without CSV — ssl_only fallback."""
    root = tmp_path_factory.mktemp("deepMTJ_nocsv")
    (root / "fullres").mkdir()
    for i in range(1, 3):
        (root / "fullres" / f"deepMTJ_TS_ffullres_f{i:04d}_annotated.jpg").write_bytes(b"\x89PNG")
    return root


class TestDeepMTJAdapter:

    def test_import(self):
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        assert DeepMTJAdapter.DATASET_ID     == "deepMTJ"
        assert DeepMTJAdapter.ANATOMY_FAMILY == "muscle"
        assert DeepMTJAdapter.SONODQS        == "bronze"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "deepMTJ" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, deepmtj_root):
        """3 fullres + 3 256x128px = 6 total."""
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        entries = list(DeepMTJAdapter(root=deepmtj_root).iter_entries())
        assert len(entries) == 6

    def test_entry_schema(self, deepmtj_root):
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in DeepMTJAdapter(root=deepmtj_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "deepMTJ"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.has_mask        is False
            assert e.probe_type      == "linear"

    def test_keypoint_task_with_csv(self, deepmtj_root):
        """Fullres frames with CSV → task_type = keypoint with coords."""
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        entries = list(DeepMTJAdapter(root=deepmtj_root, resolutions=["fullres"]).iter_entries())
        assert len(entries) == 3
        for e in entries:
            assert e.task_type == "keypoint"
            assert e.source_meta["mtj_x"] is not None
            assert e.source_meta["mtj_y"] is not None

    def test_ssl_only_without_csv_match(self, deepmtj_root):
        """256x128px frames without CSV entry → task_type = ssl_only."""
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        entries = list(DeepMTJAdapter(root=deepmtj_root, resolutions=["256x128px"]).iter_entries())
        for e in entries:
            assert e.task_type == "ssl_only"
            assert e.source_meta["mtj_x"] is None
            assert e.source_meta["mtj_y"] is None

    def test_ssl_only_no_csv(self, deepmtj_root_no_csv):
        """Without CSV at all, all entries are ssl_only."""
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        entries = list(DeepMTJAdapter(root=deepmtj_root_no_csv).iter_entries())
        assert len(entries) == 2
        for e in entries:
            assert e.task_type == "ssl_only"

    def test_resolution_in_source_meta(self, deepmtj_root):
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        entries = list(DeepMTJAdapter(root=deepmtj_root).iter_entries())
        resolutions = {e.source_meta["resolution"] for e in entries}
        assert "fullres"   in resolutions
        assert "256x128px" in resolutions

    def test_frame_idx_parsed(self, deepmtj_root):
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        entries = list(DeepMTJAdapter(root=deepmtj_root, resolutions=["fullres"]).iter_entries())
        frame_idxs = {e.source_meta["frame_idx"] for e in entries}
        assert frame_idxs == {1, 2, 3}

    def test_resolution_filter(self, deepmtj_root):
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        entries = list(DeepMTJAdapter(root=deepmtj_root, resolutions=["fullres"]).iter_entries())
        assert len(entries) == 3
        assert all(e.source_meta["resolution"] == "fullres" for e in entries)

    def test_split_override(self, deepmtj_root):
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        for e in DeepMTJAdapter(root=deepmtj_root, split_override="test").iter_entries():
            assert e.split == "test"

    def test_sample_ids_unique(self, deepmtj_root):
        from data.adapters.muscle.deep_mtj import DeepMTJAdapter
        ids = [e.sample_id for e in DeepMTJAdapter(root=deepmtj_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, deepmtj_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "deepmtj.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("deepMTJ", deepmtj_root, writer)
        assert count == 6
        entries = load_manifest(out)
        assert all(e.dataset_id == "deepMTJ" for e in entries)
