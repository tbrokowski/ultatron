"""
tests/adapters/liver/test_fatty_liver_bmode.py
===============================================

Unit tests for FattyLiverBmodeAdapter.

Run with:
    pytest tests/adapters/liver/test_fatty_liver_bmode.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

scipy = pytest.importorskip("scipy", reason="scipy required for .mat loading")
PIL   = pytest.importorskip("PIL",   reason="Pillow required for PNG extraction")


# ── Fixture ───────────────────────────────────────────────────────────────────

N_PATIENTS = 5
N_FRAMES   = 2
IMG_H, IMG_W = 8, 8


def _build_fatty_liver(root: Path) -> Path:
    """Create a minimal synthetic .mat file with N_PATIENTS × N_FRAMES frames."""
    import scipy.io

    rng = np.random.default_rng(42)
    root.mkdir(parents=True, exist_ok=True)

    # Build (1, N_PATIENTS) structured numpy array matching the real layout.
    # Fields: id, class, fat, images — all stored as nested numpy arrays so
    # that scipy.io.savemat emits a proper MATLAB struct array.
    dt   = np.dtype([("id", "O"), ("class", "O"), ("fat", "O"), ("images", "O")])
    data = np.empty((1, N_PATIENTS), dtype=dt)

    for i in range(N_PATIENTS):
        frames = rng.integers(0, 256, (N_FRAMES, IMG_H, IMG_W)).astype(np.float64)
        data[0, i]["id"]     = np.array([[i + 1]])
        data[0, i]["class"]  = np.array([[i % 2]])       # alternating 0/1
        data[0, i]["fat"]    = np.array([[i * 15]])       # steatosis %: 0,15,30,45,60
        data[0, i]["images"] = frames

    mat_path = root / "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
    scipy.io.savemat(str(mat_path), {"data": data})
    return root


@pytest.fixture(scope="module")
def fatty_liver_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fatty_liver")
    _build_fatty_liver(root)
    return root


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_entries(root):
    from data.adapters.liver.fatty_liver_bmode import FattyLiverBmodeAdapter
    return list(FattyLiverBmodeAdapter(root).iter_entries())


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFattyLiverBmodeAdapterMeta:

    def test_class_attributes(self):
        from data.adapters.liver.fatty_liver_bmode import FattyLiverBmodeAdapter
        assert FattyLiverBmodeAdapter.DATASET_ID     == "fatty-liver-bmode"
        assert FattyLiverBmodeAdapter.ANATOMY_FAMILY == "liver"
        assert FattyLiverBmodeAdapter.SONODQS        == "gold"
        assert FattyLiverBmodeAdapter.DOI            == "https://doi.org/10.5281/zenodo.1009146"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "fatty-liver-bmode" in ADAPTER_REGISTRY

    def test_missing_mat_raises(self, tmp_path):
        from data.adapters.liver.fatty_liver_bmode import FattyLiverBmodeAdapter
        with pytest.raises(FileNotFoundError, match="fatty-liver-bmode"):
            FattyLiverBmodeAdapter(tmp_path)


class TestFattyLiverBmodeEntryCount:

    def test_total_entries(self, fatty_liver_root):
        entries = _get_entries(fatty_liver_root)
        assert len(entries) == N_PATIENTS * N_FRAMES

    def test_frames_per_patient(self, fatty_liver_root):
        entries = _get_entries(fatty_liver_root)
        from collections import Counter
        counts = Counter(e.study_id for e in entries)
        assert len(counts) == N_PATIENTS
        assert all(c == N_FRAMES for c in counts.values())


class TestFattyLiverBmodeSchema:

    def test_entry_fields(self, fatty_liver_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _get_entries(fatty_liver_root):
            assert e.dataset_id      == "fatty-liver-bmode"
            assert e.anatomy_family  == "liver"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.view_type       == "liver_bmode"
            assert e.ssl_stream      == "image"
            assert e.has_mask        is False
            assert e.task_type       == "classification"
            assert e.has_temporal_order is False
            assert e.num_frames      == 1
            assert e.study_id        == e.series_id
            assert e.curriculum_tier in (1, 2, 3)
            assert len(e.image_paths) == 1

    def test_image_paths_are_pngs_that_exist(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            p = Path(e.image_paths[0])
            assert p.suffix == ".png"
            assert p.exists(), f"Extracted PNG missing: {p}"

    def test_study_id_format(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            assert e.study_id.startswith("patient_")
            assert len(e.study_id) == len("patient_001")


class TestFattyLiverBmodeLazyExtraction:

    def test_images_dir_created_after_iter(self, fatty_liver_root):
        images_dir = fatty_liver_root / "images"
        # images_dir may already exist from a previous test; count files instead
        png_count = len(list(images_dir.glob("patient_*.png")))
        assert png_count == N_PATIENTS * N_FRAMES

    def test_png_naming_convention(self, fatty_liver_root):
        images_dir = fatty_liver_root / "images"
        stems = {p.stem for p in images_dir.glob("patient_*.png")}
        for pid in range(1, N_PATIENTS + 1):
            for fid in range(N_FRAMES):
                expected = f"patient_{pid:03d}_frame_{fid:02d}"
                assert expected in stems, f"Missing PNG: {expected}.png"

    def test_extracted_pngs_are_grayscale(self, fatty_liver_root):
        from PIL import Image
        images_dir = fatty_liver_root / "images"
        for p in sorted(images_dir.glob("patient_*.png"))[:3]:
            img = Image.open(p)
            assert img.mode == "L", f"Expected grayscale, got {img.mode} for {p.name}"


class TestFattyLiverBmodeInstances:

    def test_exactly_one_instance_per_entry(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            assert len(e.instances) == 1

    def test_instance_ontology(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            inst = e.instances[0]
            assert inst.label_ontology == "liver_steatosis_class"
            assert inst.is_promptable  is False

    def test_classification_label_values(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            inst = e.instances[0]
            assert inst.classification_label in (0, 1)

    def test_label_raw_matches_class(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            inst = e.instances[0]
            if inst.classification_label == 1:
                assert inst.label_raw == "fatty_liver"
            else:
                assert inst.label_raw == "normal_liver"

    def test_both_classes_present(self, fatty_liver_root):
        entries   = _get_entries(fatty_liver_root)
        cls_vals  = {e.instances[0].classification_label for e in entries}
        assert 0 in cls_vals, "No normal-liver entries"
        assert 1 in cls_vals, "No fatty-liver entries"


class TestFattyLiverBmodeSourceMeta:

    def test_required_keys(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            sm = e.source_meta
            assert "patient_id"           in sm
            assert "frame_idx"            in sm
            assert "fat_pct"              in sm
            assert "classification_label" in sm

    def test_frame_idx_range(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            assert 0 <= e.source_meta["frame_idx"] < N_FRAMES

    def test_fat_pct_non_negative(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            assert e.source_meta["fat_pct"] >= 0

    def test_classification_label_matches_instance(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            assert e.source_meta["classification_label"] == e.instances[0].classification_label

    def test_patient_id_matches_study_id(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            pid = e.source_meta["patient_id"]
            assert e.study_id == f"patient_{pid:03d}"


class TestFattyLiverBmodeSplit:

    def test_all_splits_valid(self, fatty_liver_root):
        for e in _get_entries(fatty_liver_root):
            assert e.split in ("train", "val", "test")

    def test_patient_level_split_consistency(self, fatty_liver_root):
        """All frames belonging to the same patient must be in the same split."""
        entries    = _get_entries(fatty_liver_root)
        by_patient: dict[str, set[str]] = {}
        for e in entries:
            by_patient.setdefault(e.study_id, set()).add(e.split)
        for study_id, splits in by_patient.items():
            assert len(splits) == 1, (
                f"Patient {study_id} has frames in multiple splits: {splits}"
            )

    def test_split_override(self, fatty_liver_root):
        from data.adapters.liver.fatty_liver_bmode import FattyLiverBmodeAdapter
        entries = list(FattyLiverBmodeAdapter(fatty_liver_root, split_override="val").iter_entries())
        assert all(e.split == "val" for e in entries)

    def test_all_patients_covered(self, fatty_liver_root):
        entries   = _get_entries(fatty_liver_root)
        study_ids = {e.study_id for e in entries}
        assert len(study_ids) == N_PATIENTS
