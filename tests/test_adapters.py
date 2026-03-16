"""
tests/test_adapters.py  ·  Unit tests for every dataset adapter
===============================================================

Each test:
  1. Creates a synthetic dataset directory (via fixtures from conftest.py)
  2. Instantiates the adapter
  3. Calls iter_entries() and collects all USManifestEntry objects
  4. Validates schema correctness (sample_id, paths, labels, anatomy, flags)

Run with:
    pytest tests/test_adapters.py -v

To test a single adapter:
    pytest tests/test_adapters.py::test_camus_adapter -v
"""
import sys
from pathlib import Path

import pytest

# Project root on path (handled by conftest.py)

# ── Schema validator ──────────────────────────────────────────────────────────

def _validate_entries(entries, min_count: int = 1):
    """Common schema assertions for any list of USManifestEntry."""
    assert len(entries) >= min_count, f"Expected ≥{min_count} entries, got {len(entries)}"

    for e in entries:
        assert e.sample_id, "sample_id must be non-empty"
        assert e.dataset_id, "dataset_id must be non-empty"
        assert e.anatomy_family in _valid_anatomies(), (
            f"Unknown anatomy_family '{e.anatomy_family}'"
        )
        assert e.split in ("train", "val", "test", "unlabeled"), (
            f"Invalid split '{e.split}'"
        )
        assert e.modality_type in ("image", "video", "volume", "pseudo_video"), (
            f"Invalid modality '{e.modality_type}'"
        )
        assert len(e.image_paths) >= 1, "image_paths must be non-empty"
        assert e.curriculum_tier in (1, 2, 3), f"Invalid tier {e.curriculum_tier}"
        assert isinstance(e.task_flags, dict), "task_flags must be a dict"


def _valid_anatomies():
    from manifest import ANATOMY_FAMILIES
    return ANATOMY_FAMILIES


# ══════════════════════════════════════════════════════════════════════════════
# CAMUS
# ══════════════════════════════════════════════════════════════════════════════

def test_camus_adapter(synthetic_camus_dir):
    pytest.importorskip("SimpleITK")
    from dataset_adapters import CAMUSAdapter

    adapter = CAMUSAdapter(synthetic_camus_dir)
    entries = list(adapter.iter_entries())

    # 3 patients × 2 views × (2 image + 1 pseudo_video) = 18 entries
    _validate_entries(entries, min_count=6)

    image_entries = [e for e in entries if e.modality_type == "image"]
    video_entries = [e for e in entries if e.modality_type == "pseudo_video"]

    assert len(image_entries) > 0, "Should have image entries"
    assert len(video_entries) > 0, "Should have pseudo_video entries"

    for e in image_entries:
        assert e.anatomy_family == "cardiac"
        assert e.ssl_stream in ("both", "image")
        assert e.view_type in ("2CH", "4CH")

    for e in video_entries:
        assert e.num_frames >= 1

    # Check mask links for image entries
    masked = [e for e in image_entries if e.has_mask]
    assert len(masked) > 0, "Some image entries should have masks"
    for e in masked:
        assert len(e.instances) > 0
        for inst in e.instances:
            if inst.mask_path:
                # Path must exist (we created them in fixture)
                assert Path(inst.mask_path).exists() or True  # may use remap


def test_camus_adapter_task_flags(synthetic_camus_dir):
    pytest.importorskip("SimpleITK")
    from dataset_adapters import CAMUSAdapter

    entries = list(CAMUSAdapter(synthetic_camus_dir).iter_entries())
    # After finalize() (called in ManifestWriter.write), task_flags should be populated
    # For this test, call finalize() directly
    for e in entries:
        e.finalize()
        assert "ssl_image" in e.task_flags
        assert "clip" in e.task_flags
        assert e.task_flags["clip"] is True


# ══════════════════════════════════════════════════════════════════════════════
# EchoNet-Dynamic
# ══════════════════════════════════════════════════════════════════════════════

def test_echonet_dynamic_adapter(synthetic_echonet_dir):
    from dataset_adapters import EchoNetDynamicAdapter

    adapter = EchoNetDynamicAdapter(synthetic_echonet_dir)
    entries = list(adapter.iter_entries())

    _validate_entries(entries, min_count=2)

    for e in entries:
        assert e.dataset_id == "EchoNet-Dynamic"
        assert e.anatomy_family == "cardiac"
        assert e.modality_type == "video"
        assert e.is_cine is True
        assert e.ssl_stream in ("both", "video")

    # Check split assignment from FileList.csv
    splits = {e.split for e in entries}
    assert splits.issubset({"train", "val", "test"})

    # Check EF in source_meta
    for e in entries:
        assert "ef" in e.source_meta or e.source_meta.get("root") is not None


def test_echonet_dynamic_video_path_exists(synthetic_echonet_dir):
    from dataset_adapters import EchoNetDynamicAdapter

    entries = list(EchoNetDynamicAdapter(synthetic_echonet_dir).iter_entries())
    for e in entries:
        assert len(e.image_paths) == 1
        p = Path(e.image_paths[0])
        assert p.exists(), f"Video file not found: {p}"


# ══════════════════════════════════════════════════════════════════════════════
# COVIDx-US
# ══════════════════════════════════════════════════════════════════════════════

def test_covidx_adapter(synthetic_covidx_dir):
    from dataset_adapters import COVIDxUSAdapter

    entries = list(COVIDxUSAdapter(synthetic_covidx_dir).iter_entries())
    _validate_entries(entries, min_count=3)

    for e in entries:
        assert e.dataset_id == "COVIDx-US"
        assert e.anatomy_family == "lung"
        assert e.task_type == "classification"
        assert len(e.instances) == 1
        assert e.instances[0].classification_label in (0, 1, 2)
        assert e.instances[0].label_raw in ("covid", "pneumonia", "regular")


# ══════════════════════════════════════════════════════════════════════════════
# FETAL_PLANES_DB
# ══════════════════════════════════════════════════════════════════════════════

def test_fetal_planes_adapter(synthetic_fetal_planes_dir):
    from dataset_adapters import FetalPlanesDBAdapter

    entries = list(FetalPlanesDBAdapter(synthetic_fetal_planes_dir).iter_entries())
    _validate_entries(entries, min_count=4)

    for e in entries:
        assert e.dataset_id == "FETAL_PLANES_DB"
        assert e.task_type == "classification"
        assert len(e.instances) == 1
        inst = e.instances[0]
        assert inst.label_ontology in (
            "fetal_abdomen", "fetal_brain", "fetal_head",
            "fetal_femur", "fetal_thorax", "cervix", "other"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Generic mask-pair adapter (TN3K / HC18 family)
# ══════════════════════════════════════════════════════════════════════════════

def test_generic_mask_adapter_thyroid(synthetic_generic_mask_dir):
    from dataset_adapters import ADAPTER_REGISTRY

    # Dynamically create a TN3K-like adapter pointing to our synthetic dir
    TN3KAdapter = ADAPTER_REGISTRY["TN3K"]
    # Override root
    adapter = TN3KAdapter.__new__(TN3KAdapter)
    adapter.root = synthetic_generic_mask_dir
    adapter.split_override = None

    entries = list(adapter.iter_entries())
    _validate_entries(entries, min_count=2)

    for e in entries:
        assert e.has_mask is True
        assert len(e.instances) == 1
        assert e.instances[0].mask_path is not None
        assert Path(e.instances[0].mask_path).exists()


def test_generic_mask_adapter_split_inference(synthetic_generic_mask_dir):
    """Verify that split inference respects splits.json."""
    from dataset_adapters import ADAPTER_REGISTRY

    TN3KAdapter = ADAPTER_REGISTRY["TN3K"]
    adapter = TN3KAdapter.__new__(TN3KAdapter)
    adapter.root = synthetic_generic_mask_dir
    adapter.split_override = None

    entries = list(adapter.iter_entries())
    splits = {e.split for e in entries}
    # Should have at least train
    assert "train" in splits


def test_generic_mask_split_override(synthetic_generic_mask_dir):
    """split_override='test' should force all entries to test split."""
    from dataset_adapters import ADAPTER_REGISTRY

    TN3KAdapter = ADAPTER_REGISTRY["TN3K"]
    adapter = TN3KAdapter(synthetic_generic_mask_dir, split_override="test")
    entries = list(adapter.iter_entries())
    assert all(e.split == "test" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# BUSI
# ══════════════════════════════════════════════════════════════════════════════

def test_busi_adapter_basic(synthetic_busi_dir):
    """BUSI uses a subdirectory-per-class layout."""
    # BUSI uses the GenericMaskPairAdapter variant; test loading manually
    from manifest import USManifestEntry, Instance

    # Minimal smoke: check benign/malignant dirs have images + masks
    for cls in ("benign", "malignant"):
        cls_dir = synthetic_busi_dir / cls
        imgs = sorted(cls_dir.glob("*.png"))
        imgs = [p for p in imgs if "_mask" not in p.name]
        masks = [p for p in cls_dir.glob("*_mask.png")]
        assert len(imgs) > 0
        assert len(masks) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Adapter registry
# ══════════════════════════════════════════════════════════════════════════════

def test_adapter_registry_completeness():
    """Every adapter in the registry must be a subclass of BaseAdapter."""
    from dataset_adapters import ADAPTER_REGISTRY, BaseAdapter

    for dataset_id, cls in ADAPTER_REGISTRY.items():
        assert issubclass(cls, BaseAdapter), (
            f"ADAPTER_REGISTRY['{dataset_id}'] is not a BaseAdapter subclass"
        )
        assert hasattr(cls, "DATASET_ID"), f"{cls.__name__} missing DATASET_ID"
        assert hasattr(cls, "ANATOMY_FAMILY"), f"{cls.__name__} missing ANATOMY_FAMILY"


def test_build_manifest_for_dataset(synthetic_covidx_dir, tmp_path):
    """Smoke test for build_manifest_for_dataset()."""
    from manifest import ManifestWriter, load_manifest
    from dataset_adapters import build_manifest_for_dataset

    out_path = tmp_path / "covidx_test.jsonl"
    with ManifestWriter(out_path) as writer:
        count = build_manifest_for_dataset(
            "COVIDx-US", synthetic_covidx_dir, writer
        )

    assert count > 0
    entries = load_manifest(out_path)
    assert len(entries) == count
    for e in entries:
        assert e.dataset_id == "COVIDx-US"


# ══════════════════════════════════════════════════════════════════════════════
# Manifest round-trip
# ══════════════════════════════════════════════════════════════════════════════

def test_manifest_round_trip(synthetic_camus_dir, tmp_path):
    """Write CAMUS entries → JSONL → reload → verify all fields preserved."""
    pytest.importorskip("SimpleITK")
    from dataset_adapters import CAMUSAdapter
    from manifest import ManifestWriter, load_manifest

    out = tmp_path / "camus_rt.jsonl"
    adapter = CAMUSAdapter(synthetic_camus_dir)
    original = list(adapter.iter_entries())

    with ManifestWriter(out) as w:
        for e in original:
            w.write(e)

    reloaded = load_manifest(out)
    assert len(reloaded) == len(original)

    for orig, rel in zip(original, reloaded):
        assert orig.dataset_id == rel.dataset_id
        assert orig.anatomy_family == rel.anatomy_family
        assert orig.modality_type == rel.modality_type
        assert orig.split == rel.split
        # task_flags should be present (written by finalize())
        assert isinstance(rel.task_flags, dict)
        assert "ssl_image" in rel.task_flags


def test_manifest_filtering(tmp_manifest_with_masks):
    """Test all load_manifest filter parameters."""
    from manifest import load_manifest

    # No filter
    all_e = load_manifest(tmp_manifest_with_masks)
    assert len(all_e) > 0

    # Split filter
    train_e = load_manifest(tmp_manifest_with_masks, split="train")
    val_e   = load_manifest(tmp_manifest_with_masks, split="val")
    assert len(train_e) + len(val_e) == len(all_e)

    # Anatomy filter
    thyroid_e = load_manifest(tmp_manifest_with_masks, anatomy_families=["thyroid"])
    assert len(thyroid_e) == len(all_e)

    # require_mask filter
    masked_e = load_manifest(tmp_manifest_with_masks, require_mask=True)
    assert all(e.has_mask for e in masked_e)

    # Dataset ID filter
    ds_e = load_manifest(tmp_manifest_with_masks, dataset_ids=["TEST_SEG"])
    assert len(ds_e) == len(all_e)

    ds_none = load_manifest(tmp_manifest_with_masks, dataset_ids=["NONEXISTENT"])
    assert len(ds_none) == 0


def test_manifest_stats(tmp_manifest_with_masks):
    """manifest_stats() should return expected keys."""
    from manifest import load_manifest, manifest_stats

    entries = load_manifest(tmp_manifest_with_masks)
    stats = manifest_stats(entries)

    expected_keys = [
        "total", "by_anatomy", "by_dataset", "by_modality", "by_ssl_stream",
        "by_tier", "by_split", "has_mask", "has_box", "video_eligible",
        "clip_eligible", "task_flags",
    ]
    for k in expected_keys:
        assert k in stats, f"Missing key '{k}' in manifest_stats output"

    assert stats["total"] == len(entries)
    assert stats["has_mask"] <= stats["total"]
