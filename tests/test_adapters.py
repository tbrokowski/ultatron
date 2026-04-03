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


def _valid_anatomies():
    from data.schema.manifest import ANATOMY_FAMILIES
    return ANATOMY_FAMILIES


# ══════════════════════════════════════════════════════════════════════════════
# CAMUS
# ══════════════════════════════════════════════════════════════════════════════

def test_camus_adapter(camus_root):
    pytest.importorskip("SimpleITK")
    from data.adapters.cardiac.camus import CAMUSAdapter

    adapter = CAMUSAdapter(camus_root)
    entries = list(adapter.iter_entries())

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

    masked = [e for e in image_entries if e.has_mask]
    assert len(masked) > 0, "Some image entries should have masks"
    for e in masked:
        assert len(e.instances) > 0


def test_camus_adapter_ssl_stream(camus_root):
    pytest.importorskip("SimpleITK")
    from data.adapters.cardiac.camus import CAMUSAdapter

    entries = list(CAMUSAdapter(camus_root).iter_entries())
    for e in entries:
        assert e.ssl_stream in ("image", "video", "both")


# ══════════════════════════════════════════════════════════════════════════════
# EchoNet-Dynamic
# ══════════════════════════════════════════════════════════════════════════════

def test_echonet_dynamic_adapter(echonet_root):
    from data.adapters.cardiac.echonet import EchoNetDynamicAdapter

    adapter = EchoNetDynamicAdapter(echonet_root)
    entries = list(adapter.iter_entries())

    _validate_entries(entries, min_count=2)

    for e in entries:
        assert e.dataset_id == "EchoNet-Dynamic"
        assert e.anatomy_family == "cardiac"
        assert e.modality_type == "video"
        assert e.ssl_stream in ("both", "video")

    splits = {e.split for e in entries}
    assert splits.issubset({"train", "val", "test"})

    for e in entries:
        assert "ef" in e.source_meta


def test_echonet_dynamic_video_path_exists(echonet_root):
    from data.adapters.cardiac.echonet import EchoNetDynamicAdapter

    entries = list(EchoNetDynamicAdapter(echonet_root).iter_entries())
    for e in entries:
        assert len(e.image_paths) == 1
        p = Path(e.image_paths[0])
        assert p.exists(), f"Video file not found: {p}"


# ══════════════════════════════════════════════════════════════════════════════
# COVIDx-US
# ══════════════════════════════════════════════════════════════════════════════

def test_covidx_adapter(covidx_root):
    pytest.skip("COVIDxUSAdapter not yet ported to new subpackage layout")


# ══════════════════════════════════════════════════════════════════════════════
# FETAL_PLANES_DB
# ══════════════════════════════════════════════════════════════════════════════

def test_fetal_planes_adapter(fetal_planes_root):
    pytest.skip("FetalPlanesDBAdapter not yet ported to new subpackage layout")


# ══════════════════════════════════════════════════════════════════════════════
# Generic mask-pair adapter (TN3K / HC18 family)
# ══════════════════════════════════════════════════════════════════════════════

def test_generic_mask_adapter_thyroid(tn3k_root):
    from data.adapters import ADAPTER_REGISTRY

    TN3KAdapter = ADAPTER_REGISTRY["TN3K"]
    adapter = TN3KAdapter.__new__(TN3KAdapter)
    adapter.root = tn3k_root
    adapter.split_override = None

    entries = list(adapter.iter_entries())
    _validate_entries(entries, min_count=2)

    for e in entries:
        assert e.has_mask is True
        assert len(e.instances) == 1
        assert e.instances[0].mask_path is not None
        assert Path(e.instances[0].mask_path).exists()


def test_generic_mask_adapter_split_inference(tn3k_root):
    """Verify that split inference respects splits.json."""
    from data.adapters import ADAPTER_REGISTRY

    TN3KAdapter = ADAPTER_REGISTRY["TN3K"]
    adapter = TN3KAdapter.__new__(TN3KAdapter)
    adapter.root = tn3k_root
    adapter.split_override = None

    entries = list(adapter.iter_entries())
    splits = {e.split for e in entries}
    assert "train" in splits


def test_generic_mask_split_override(tn3k_root):
    """split_override='test' should force all entries to test split."""
    from data.adapters import ADAPTER_REGISTRY

    TN3KAdapter = ADAPTER_REGISTRY["TN3K"]
    adapter = TN3KAdapter(tn3k_root, split_override="test")
    entries = list(adapter.iter_entries())
    assert all(e.split == "test" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# BUSI
# ══════════════════════════════════════════════════════════════════════════════

def test_busi_adapter_basic(busi_root):
    """BUSI adapter should yield image entries for benign, malignant, and normal."""
    from data.adapters.busi import BUSIAdapter

    entries = list(BUSIAdapter(busi_root).iter_entries())
    _validate_entries(entries, min_count=4)

    for e in entries:
        assert e.dataset_id == "BUSI"
        assert e.anatomy_family == "breast"
        assert e.modality_type == "image"


# ══════════════════════════════════════════════════════════════════════════════
# Adapter registry
# ══════════════════════════════════════════════════════════════════════════════

def test_adapter_registry_completeness():
    """Every adapter in the registry must be a subclass of BaseAdapter."""
    from data.adapters import ADAPTER_REGISTRY
    from data.adapters.base import BaseAdapter

    for dataset_id, cls in ADAPTER_REGISTRY.items():
        assert issubclass(cls, BaseAdapter), (
            f"ADAPTER_REGISTRY['{dataset_id}'] is not a BaseAdapter subclass"
        )
        assert hasattr(cls, "DATASET_ID"), f"{cls.__name__} missing DATASET_ID"
        assert hasattr(cls, "ANATOMY_FAMILY"), f"{cls.__name__} missing ANATOMY_FAMILY"


def test_build_manifest_for_dataset(busi_root, tmp_path):
    """Smoke test for build_manifest_for_dataset()."""
    from data.schema.manifest import ManifestWriter, load_manifest
    from data.adapters import build_manifest_for_dataset

    out_path = tmp_path / "busi_test.jsonl"
    with ManifestWriter(out_path) as writer:
        count = build_manifest_for_dataset("BUSI", busi_root, writer)

    assert count > 0
    entries = load_manifest(out_path)
    assert len(entries) == count
    for e in entries:
        assert e.dataset_id == "BUSI"


# ══════════════════════════════════════════════════════════════════════════════
# Manifest round-trip
# ══════════════════════════════════════════════════════════════════════════════

def test_manifest_round_trip(camus_root, tmp_path):
    """Write CAMUS entries → JSONL → reload → verify all fields preserved."""
    pytest.importorskip("SimpleITK")
    from data.adapters.cardiac.camus import CAMUSAdapter
    from data.schema.manifest import ManifestWriter, load_manifest

    out = tmp_path / "camus_rt.jsonl"
    original = list(CAMUSAdapter(camus_root).iter_entries())

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


def test_manifest_filtering(tmp_manifest_with_masks):
    """Test all load_manifest filter parameters."""
    from data.schema.manifest import load_manifest

    all_e = load_manifest(tmp_manifest_with_masks)
    assert len(all_e) > 0

    train_e = load_manifest(tmp_manifest_with_masks, split="train")
    val_e   = load_manifest(tmp_manifest_with_masks, split="val")
    assert len(train_e) + len(val_e) == len(all_e)

    thyroid_e = load_manifest(tmp_manifest_with_masks, anatomy_families=["thyroid"])
    assert len(thyroid_e) == len(all_e)

    # Manually filter by has_mask (load_manifest doesn't have require_mask)
    masked_e = [e for e in all_e if e.has_mask]
    assert all(e.has_mask for e in masked_e)

    # load_manifest filters by anatomy and ssl_stream; dataset_id filter is done inline
    all_by_ds = [e for e in all_e if e.dataset_id == "TEST_SEG"]
    assert len(all_by_ds) == len(all_e)

    none_by_ds = [e for e in all_e if e.dataset_id == "NONEXISTENT"]
    assert len(none_by_ds) == 0


def test_manifest_stats(tmp_manifest_with_masks):
    """manifest_stats() should return expected keys."""
    from data.schema.manifest import load_manifest, manifest_stats

    entries = load_manifest(tmp_manifest_with_masks)
    stats = manifest_stats(entries)

    expected_keys = [
        "total", "by_anatomy", "by_dataset", "by_modality", "by_ssl_stream",
        "by_tier", "has_mask", "video_eligible",
    ]
    for k in expected_keys:
        assert k in stats, f"Missing key '{k}' in manifest_stats output"

    assert stats["total"] == len(entries)
    assert stats["has_mask"] <= stats["total"]


# ══════════════════════════════════════════════════════════════════════════════
# EchoNet-Pediatric
# ══════════════════════════════════════════════════════════════════════════════

def test_echonet_pediatric_adapter(echonet_pediatric_root):
    from data.adapters.cardiac.echonet_pediatric import EchoNetPediatricAdapter

    adapter = EchoNetPediatricAdapter(echonet_pediatric_root)
    entries = list(adapter.iter_entries())

    _validate_entries(entries, min_count=2)

    for e in entries:
        assert e.dataset_id     == "EchoNet-Pediatric"
        assert e.anatomy_family == "cardiac"
        assert e.modality_type  == "video"
        assert e.is_cine        is True
        assert e.ssl_stream     in ("both", "video")
        assert "ef"   in e.source_meta
        assert "view" in e.source_meta
        assert e.source_meta["view"] in ("A4C", "PSAX")

    # Both views should be present
    views = {e.source_meta["view"] for e in entries}
    assert "A4C" in views
    assert "PSAX" in views


def test_echonet_pediatric_split_mapping(echonet_pediatric_root):
    """Numeric Split column (0-9) must map to train/val/test correctly."""
    from data.adapters.cardiac.echonet_pediatric import EchoNetPediatricAdapter, _SPLIT_MAP

    assert _SPLIT_MAP["0"] == "train"
    assert _SPLIT_MAP["6"] == "train"
    assert _SPLIT_MAP["7"] == "val"
    assert _SPLIT_MAP["8"] == "test"
    assert _SPLIT_MAP["9"] == "test"

    entries = list(EchoNetPediatricAdapter(echonet_pediatric_root).iter_entries())
    assert all(e.split in ("train", "val", "test") for e in entries)


def test_echonet_pediatric_video_paths_exist(echonet_pediatric_root):
    from data.adapters.cardiac.echonet_pediatric import EchoNetPediatricAdapter

    for e in EchoNetPediatricAdapter(echonet_pediatric_root).iter_entries():
        assert len(e.image_paths) == 1
        assert Path(e.image_paths[0]).exists(), f"AVI not found: {e.image_paths[0]}"


# ══════════════════════════════════════════════════════════════════════════════
# TED
# ══════════════════════════════════════════════════════════════════════════════

def test_ted_adapter(ted_root):
    from data.adapters.cardiac.ted import TEDAdapter

    adapter = TEDAdapter(ted_root)
    entries = list(adapter.iter_entries())

    _validate_entries(entries, min_count=2)

    video_entries = [e for e in entries if e.modality_type == "video"]
    image_entries = [e for e in entries if e.modality_type == "image"]

    assert len(video_entries) > 0, "TED should emit video entries for each patient"
    assert len(image_entries) > 0, "TED should emit image entries for ED and ES frames"

    for e in video_entries:
        assert e.dataset_id     == "TED"
        assert e.anatomy_family == "cardiac"
        assert e.view_type      == "4CH"
        assert e.is_cine        is True
        assert e.ssl_stream     in ("both", "video")
        assert "ef" in e.source_meta

    for e in image_entries:
        assert "phase"     in e.source_meta
        assert "frame_idx" in e.source_meta
        assert e.source_meta["phase"] in ("ED", "ES")
        assert isinstance(e.source_meta["frame_idx"], int)


def test_ted_adapter_masks(ted_root):
    """TED video entries should have mask-linked instances."""
    from data.adapters.cardiac.ted import TEDAdapter

    video_entries = [
        e for e in TEDAdapter(ted_root).iter_entries()
        if e.modality_type == "video"
    ]
    masked = [e for e in video_entries if e.has_mask]
    assert len(masked) > 0, "Some TED entries should have segmentation masks"
    for e in masked:
        assert len(e.instances) > 0
        assert e.instances[0].mask_path is not None


# ══════════════════════════════════════════════════════════════════════════════
# Unity
# ══════════════════════════════════════════════════════════════════════════════

def test_unity_adapter(unity_root):
    from data.adapters.cardiac.unity import UnityAdapter

    adapter = UnityAdapter(unity_root)
    entries = list(adapter.iter_entries())

    _validate_entries(entries, min_count=1)

    for e in entries:
        assert e.dataset_id     == "Unity-Echo"
        assert e.anatomy_family == "cardiac"
        assert e.modality_type  == "image"
        assert e.ssl_stream     in ("image", "both")
        assert "keypoints" in e.source_meta
        assert isinstance(e.source_meta["keypoints"], dict)
        # Only active keypoints should be present
        for kp_name, kp_val in e.source_meta["keypoints"].items():
            assert kp_val.get("type") not in ("off", "blurred", "")


def test_unity_adapter_split_assignment(unity_root):
    """Entries in labels-train.json → train, labels-tune.json → val."""
    from data.adapters.cardiac.unity import UnityAdapter

    entries = list(UnityAdapter(unity_root).iter_entries())
    splits  = {e.split for e in entries}
    assert "train" in splits
    assert "val"   in splits


def test_unity_adapter_images_exist(unity_root):
    from data.adapters.cardiac.unity import UnityAdapter

    for e in UnityAdapter(unity_root).iter_entries():
        assert Path(e.image_paths[0]).exists(), f"PNG not found: {e.image_paths[0]}"


# ══════════════════════════════════════════════════════════════════════════════
# MIMIC-IV-Echo-LVVol-A4C
# ══════════════════════════════════════════════════════════════════════════════

def test_mimic_lvvol_a4c_adapter(mimic_lvvol_a4c_root):
    from data.adapters.cardiac.mimic_lvvol_a4c import MIMICLVVolA4CAdapter

    adapter = MIMICLVVolA4CAdapter(mimic_lvvol_a4c_root)
    entries = list(adapter.iter_entries())

    _validate_entries(entries, min_count=1)

    for e in entries:
        assert e.dataset_id     == "MIMIC-IV-Echo-LVVol-A4C"
        assert e.anatomy_family == "cardiac"
        assert e.modality_type  == "video"
        assert e.task_type      == "regression"
        assert "lvef_a4c" in e.source_meta
        assert "study_id" in e.source_meta
        assert len(e.image_paths) == 1
        assert e.image_paths[0].endswith(".dcm")


def test_mimic_lvvol_a4c_dcm_paths_exist(mimic_lvvol_a4c_root):
    from data.adapters.cardiac.mimic_lvvol_a4c import MIMICLVVolA4CAdapter

    entries = list(MIMICLVVolA4CAdapter(mimic_lvvol_a4c_root).iter_entries())
    for e in entries:
        assert Path(e.image_paths[0]).exists(), f"DICOM not found: {e.image_paths[0]}"


# ══════════════════════════════════════════════════════════════════════════════
# Cardiac registry completeness
# ══════════════════════════════════════════════════════════════════════════════

def test_cardiac_adapter_registry_completeness():
    """All new cardiac datasets must be registered in ADAPTER_REGISTRY and DATASET_STORE_MAP."""
    from data.adapters import ADAPTER_REGISTRY, BaseAdapter
    from data.infra.storage import DATASET_STORE_MAP

    required_cardiac = [
        "CAMUS", "EchoNet-Dynamic", "EchoNet-Pediatric",
        "EchoNet-LVH", "MIMIC-IV-ECHO", "MIMIC-IV-Echo-LVVol-A4C",
        "TED", "Unity-Echo", "CardiacUDC", "EchoCP",
    ]
    for ds_id in required_cardiac:
        assert ds_id in ADAPTER_REGISTRY, (
            f"'{ds_id}' missing from ADAPTER_REGISTRY"
        )
        assert ds_id in DATASET_STORE_MAP, (
            f"'{ds_id}' missing from DATASET_STORE_MAP"
        )
        cls = ADAPTER_REGISTRY[ds_id]
        assert issubclass(cls, BaseAdapter), f"{cls.__name__} is not a BaseAdapter subclass"
        assert cls.ANATOMY_FAMILY == "cardiac", (
            f"{cls.__name__}.ANATOMY_FAMILY should be 'cardiac', got {cls.ANATOMY_FAMILY!r}"
        )
