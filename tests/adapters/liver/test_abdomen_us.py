"""
tests/adapters/liver/test_abdomen_us.py
========================================

Unit tests for AbdomenUSAdapter.

Run with:
    pytest tests/adapters/liver/test_abdomen_us.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

PIL = pytest.importorskip("PIL", reason="Pillow required for mask creation/parsing")
from PIL import Image as PILImage


# ── Canonical colors (must match the adapter's _COLOR_MAP) ────────────────────

LIVER_COLOR       = (128,   0, 128)
KIDNEY_COLOR      = (255, 255,   0)
PANCREAS_COLOR    = (  0,   0, 255)
VESSELS_COLOR     = (255,   0,   0)
ADRENALS_COLOR    = (  0, 255, 255)
GALLBLADDER_COLOR = (  0, 128,   0)
BONES_COLOR       = (255, 255, 255)
SPLEEN_COLOR      = (255, 192, 203)
BACKGROUND        = (  0,   0,   0)


def _make_mask(colors: list[tuple[int, int, int]], size: int = 16) -> np.ndarray:
    """
    Create an (H, W, 3) uint8 array where each unique color occupies an equal
    horizontal stripe.  Background fills the remainder so total rows sum to H.
    """
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    rows_per = max(1, size // max(len(colors), 1))
    for i, color in enumerate(colors):
        arr[i * rows_per: (i + 1) * rows_per, :] = color
    return arr


def _save_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(arr).save(str(path))


def _save_gray_png(path: Path, size: int = 8) -> None:
    _save_png(path, np.zeros((size, size, 3), dtype=np.uint8))


def _build_abdomen_us(root: Path) -> None:
    """
    Synthetic layout:

    AUS/
      images/
        train/: img_a.png, img_b.png, img_c.png
        test/:  img_d.png, img_e.png
      annotations/
        train/: img_a.png (liver+kidney), img_b.png (pancreas+spleen), img_c.png (all-background)
        test/:  img_d.png (vessels)       — img_e has no mask

    RUS/
      images/
        train/: rus_a.png, rus_b.png      — no train annotations
        test/:  rus_c.png, rus_d.png
      annotations/
        test/:  rus_c.png (gallbladder+bones)   — rus_d has no mask
    """
    # ── AUS ──────────────────────────────────────────────────────────────────
    for stem in ("img_a", "img_b", "img_c"):
        _save_gray_png(root / "AUS" / "images" / "train" / f"{stem}.png")
    for stem in ("img_d", "img_e"):
        _save_gray_png(root / "AUS" / "images" / "test" / f"{stem}.png")

    # img_a: liver + kidney
    _save_png(
        root / "AUS" / "annotations" / "train" / "img_a.png",
        _make_mask([LIVER_COLOR, KIDNEY_COLOR]),
    )
    # img_b: pancreas + spleen
    _save_png(
        root / "AUS" / "annotations" / "train" / "img_b.png",
        _make_mask([PANCREAS_COLOR, SPLEEN_COLOR]),
    )
    # img_c: all-background mask → no structures
    _save_png(
        root / "AUS" / "annotations" / "train" / "img_c.png",
        np.zeros((16, 16, 3), dtype=np.uint8),
    )
    # img_d: vessels only
    _save_png(
        root / "AUS" / "annotations" / "test" / "img_d.png",
        _make_mask([VESSELS_COLOR]),
    )
    # img_e: no annotation mask (intentionally absent)

    # ── RUS ──────────────────────────────────────────────────────────────────
    for stem in ("rus_a", "rus_b"):
        _save_gray_png(root / "RUS" / "images" / "train" / f"{stem}.png")
    for stem in ("rus_c", "rus_d"):
        _save_gray_png(root / "RUS" / "images" / "test" / f"{stem}.png")

    # rus_c: gallbladder + bones
    _save_png(
        root / "RUS" / "annotations" / "test" / "rus_c.png",
        _make_mask([GALLBLADDER_COLOR, BONES_COLOR]),
    )
    # rus_d: no annotation mask (intentionally absent)


@pytest.fixture(scope="module")
def abdomen_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("abdomen_us")
    _build_abdomen_us(root)
    return root


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entries(root, subset="AUS", **kwargs):
    from data.adapters.liver.abdomen_us import AbdomenUSAdapter
    return list(AbdomenUSAdapter(root, subset=subset, **kwargs).iter_entries())


def _by_stem(root, subset="AUS", **kwargs):
    return {Path(e.image_paths[0]).stem: e for e in _entries(root, subset=subset, **kwargs)}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAbdomenUSMeta:

    def test_class_attributes(self):
        from data.adapters.liver.abdomen_us import AbdomenUSAdapter
        assert AbdomenUSAdapter.DATASET_ID     == "AbdomenUS"
        assert AbdomenUSAdapter.ANATOMY_FAMILY == "liver"
        assert AbdomenUSAdapter.SONODQS        == "gold"
        assert AbdomenUSAdapter.DOI            == "https://doi.org/10.1007/s11548-019-02046-5"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "AbdomenUS" in ADAPTER_REGISTRY

    def test_missing_root_raises(self, tmp_path):
        from data.adapters.liver.abdomen_us import AbdomenUSAdapter
        with pytest.raises(FileNotFoundError, match="AbdomenUS"):
            AbdomenUSAdapter(tmp_path)

    def test_default_subset_is_aus(self, abdomen_root):
        from data.adapters.liver.abdomen_us import AbdomenUSAdapter
        adapter = AbdomenUSAdapter(abdomen_root)
        assert adapter.subset == "AUS"

    def test_rus_subset_accepted(self, abdomen_root):
        from data.adapters.liver.abdomen_us import AbdomenUSAdapter
        adapter = AbdomenUSAdapter(abdomen_root, subset="RUS")
        assert adapter.subset == "RUS"

    def test_unknown_subset_raises(self, abdomen_root):
        from data.adapters.liver.abdomen_us import AbdomenUSAdapter
        with pytest.raises(FileNotFoundError):
            list(AbdomenUSAdapter(abdomen_root, subset="XYZ").iter_entries())


class TestAbdomenUSEntryCount:

    def test_aus_total_entries(self, abdomen_root):
        # 3 train + 2 test = 5
        assert len(_entries(abdomen_root, subset="AUS")) == 5

    def test_rus_total_entries(self, abdomen_root):
        # 2 train + 2 test = 4
        assert len(_entries(abdomen_root, subset="RUS")) == 4

    def test_aus_split_counts(self, abdomen_root):
        entries = _entries(abdomen_root, subset="AUS")
        assert sum(1 for e in entries if e.split == "train") == 3
        assert sum(1 for e in entries if e.split == "test")  == 2

    def test_rus_split_counts(self, abdomen_root):
        entries = _entries(abdomen_root, subset="RUS")
        assert sum(1 for e in entries if e.split == "train") == 2
        assert sum(1 for e in entries if e.split == "test")  == 2


class TestAbdomenUSSchema:

    def test_entry_fields(self, abdomen_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _entries(abdomen_root):
            assert e.dataset_id     == "AbdomenUS"
            assert e.anatomy_family == "liver"
            assert e.anatomy_family in ANATOMY_FAMILIES
            assert e.modality_type  == "image"
            assert e.view_type      == "abdominal_us"
            assert e.ssl_stream     == "image"
            assert e.curriculum_tier in (1, 2, 3)
            assert len(e.image_paths) == 1

    def test_subset_stored_in_source_meta(self, abdomen_root):
        for e in _entries(abdomen_root, subset="AUS"):
            assert e.source_meta["subset"] == "AUS"
        for e in _entries(abdomen_root, subset="RUS"):
            assert e.source_meta["subset"] == "RUS"

    def test_source_meta_keys(self, abdomen_root):
        for e in _entries(abdomen_root):
            sm = e.source_meta
            assert "subset"     in sm
            assert "image_id"   in sm
            assert "has_mask"   in sm
            assert "structures" in sm

    def test_structures_list_matches_instances(self, abdomen_root):
        for e in _entries(abdomen_root):
            assert e.source_meta["structures"] == [i.label_raw for i in e.instances]


class TestAbdomenUSAnnotated:

    def test_annotated_entry_is_segmentation(self, abdomen_root):
        by = _by_stem(abdomen_root)
        for stem in ("img_a", "img_b", "img_d"):
            e = by[stem]
            assert e.task_type     == "segmentation"
            assert e.has_mask      is True
            assert e.is_promptable is True

    def test_unannotated_entry_is_ssl_only(self, abdomen_root):
        by = _by_stem(abdomen_root)
        assert by["img_e"].task_type     == "ssl_only"
        assert by["img_e"].has_mask      is False
        assert by["img_e"].is_promptable is False
        assert by["img_e"].instances     == []

    def test_all_background_mask_produces_no_instances(self, abdomen_root):
        """img_c has a mask file but it's all-black → no structures found."""
        by = _by_stem(abdomen_root)
        e  = by["img_c"]
        assert e.task_type == "ssl_only"
        assert e.has_mask  is False
        assert e.instances == []


class TestAbdomenUSColorParsing:

    def test_liver_kidney_mask_creates_two_instances(self, abdomen_root):
        by  = _by_stem(abdomen_root)
        e   = by["img_a"]
        assert len(e.instances) == 2
        ontologies = {i.label_ontology for i in e.instances}
        assert "liver_parenchyma" in ontologies
        assert "kidney"           in ontologies

    def test_pancreas_spleen_mask(self, abdomen_root):
        by = _by_stem(abdomen_root)
        e  = by["img_b"]
        assert len(e.instances) == 2
        ontologies = {i.label_ontology for i in e.instances}
        assert "pancreas" in ontologies
        assert "spleen"   in ontologies

    def test_vessels_mask(self, abdomen_root):
        by = _by_stem(abdomen_root)
        e  = by["img_d"]
        assert len(e.instances) == 1
        assert e.instances[0].label_ontology == "abdominal_vessel"
        assert e.instances[0].label_raw      == "vessels"

    def test_gallbladder_bones_in_rus(self, abdomen_root):
        by = _by_stem(abdomen_root, subset="RUS")
        e  = by["rus_c"]
        ontologies = {i.label_ontology for i in e.instances}
        assert "gallbladder" in ontologies
        assert "bone"        in ontologies

    def test_full_ontology_coverage(self):
        """Every canonical color maps to a distinct ontology."""
        from data.adapters.liver.abdomen_us import _COLOR_MAP
        ontologies = [ont for _, _, ont in _COLOR_MAP]
        assert len(ontologies) == len(set(ontologies)), "Duplicate ontologies in color map"


class TestAbdomenUSInstances:

    def test_all_instances_share_mask_path_within_image(self, abdomen_root):
        by = _by_stem(abdomen_root)
        for stem in ("img_a", "img_b"):
            e = by[stem]
            paths = {i.mask_path for i in e.instances}
            assert len(paths) == 1, "All instances should share a single mask PNG"

    def test_mask_path_points_to_existing_file(self, abdomen_root):
        by = _by_stem(abdomen_root)
        for stem in ("img_a", "img_b", "img_d"):
            for inst in by[stem].instances:
                assert inst.mask_path is not None
                assert Path(inst.mask_path).exists()

    def test_instances_are_promptable(self, abdomen_root):
        for e in _entries(abdomen_root):
            for inst in e.instances:
                assert inst.is_promptable is True

    def test_instance_id_contains_stem_and_label(self, abdomen_root):
        by = _by_stem(abdomen_root)
        for inst in by["img_a"].instances:
            assert "img_a" in inst.instance_id
            assert inst.label_raw in inst.instance_id


class TestAbdomenUSSplit:

    def test_predefined_splits(self, abdomen_root):
        by = _by_stem(abdomen_root)
        for stem in ("img_a", "img_b", "img_c"):
            assert by[stem].split == "train"
        for stem in ("img_d", "img_e"):
            assert by[stem].split == "test"

    def test_split_override(self, abdomen_root):
        entries = _entries(abdomen_root, split_override="val")
        assert all(e.split == "val" for e in entries)

    def test_rus_no_train_annotations(self, abdomen_root):
        by = _by_stem(abdomen_root, subset="RUS")
        for stem in ("rus_a", "rus_b"):
            assert by[stem].task_type == "ssl_only"
            assert by[stem].has_mask  is False
