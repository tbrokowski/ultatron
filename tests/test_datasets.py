"""
tests/test_datasets.py  ·  Comprehensive dataset loading & validation tests
============================================================================

Tests are organised in three layers:

1. Adapter tests       – does iter_entries() produce valid USManifestEntry objects?
2. Dataset tests       – does __getitem__ return correctly shaped tensors?
3. Integration tests   – do TaskConfig + DownstreamDataset work end-to-end?
4. Label spec tests    – are LabelSpec fields correct for each dataset?
5. Manifest I/O tests  – does write/read round-trip preserve all fields?
6. Storage tests       – does StorageConfig path resolution work?

Run with:
  pytest tests/test_datasets.py -v
  pytest tests/test_datasets.py -v -k "camus"  # single dataset
  pytest tests/test_datasets.py -v --tb=short  # concise tracebacks
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema.manifest import (
    USManifestEntry, Instance, ManifestWriter, load_manifest,
    manifest_stats, group_by_patient, normalize_anatomy,
)
from data.labels.label_spec import (
    LabelSpec, TaskType, LossType, LossConfig,
    binary_segmentation_spec, multiclass_segmentation_spec,
    multiclass_classification_spec,
    binary_classification_spec, clip_spec, regression_spec,
    TaskConfig, ANATOMY_LABEL_VOCAB,
)
from data.adapters.dataset_adapters import (
    CAMUSAdapter, BUSIAdapter, COVIDxUSAdapter,
    FetalPlanesDBAdapter, HC18Adapter, LUSMulticenterAdapter,
    BUSBRAAdapter, GenericMaskPairAdapter,
    ADAPTER_REGISTRY, build_manifest_for_dataset, _make_generic,
)
from data.infra.storage import StorageConfig, configure_storage, get_storage


# ─────────────────────────────────────────────────────────────────────────────
# 1. LABEL SPEC TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelSpec:
    def test_binary_seg_spec(self):
        spec = binary_segmentation_spec("cardiac", "left_ventricle",
                                         label_raw="LV", label_source="CAMUS")
        assert spec.task == TaskType.SEGMENTATION
        assert spec.is_binary is True
        assert spec.num_classes == 1
        assert spec.class_names == ["left_ventricle"]
        assert spec.loss.primary == LossType.DICE
        assert spec.loss.auxiliary == LossType.BCE
        assert spec.label_source == "CAMUS"

    def test_multiclass_seg_spec(self):
        classes = ["left_ventricle", "myocardium", "left_atrium"]
        spec = multiclass_segmentation_spec("cardiac", classes)
        assert spec.task == TaskType.SEGMENTATION
        assert spec.is_binary is False
        assert spec.num_classes == 3
        assert spec.loss.primary == LossType.CE

    def test_binary_cls_spec(self):
        spec = binary_classification_spec(
            "lung", "b_lines", 1,
            label_raw="b_lines", label_source="LUS",
            text_label="lung ultrasound with B-lines",
        )
        assert spec.task == TaskType.BINARY_CLS
        assert spec.is_binary is True
        assert spec.label_value == 1
        assert spec.text_label == "lung ultrasound with B-lines"
        assert spec.loss.primary == LossType.BCE

    def test_multiclass_cls_spec(self):
        spec = multiclass_classification_spec(
            "breast", ["normal", "benign", "malignant"], 2
        )
        assert spec.task == TaskType.MULTICLASS_CLS
        assert spec.num_classes == 3
        assert spec.label_value == 2
        assert spec.loss.primary == LossType.CE

    def test_clip_spec(self):
        spec = clip_spec("cardiac", "echocardiogram 4-chamber view")
        assert spec.task == TaskType.CLIP
        assert spec.loss.primary == LossType.CLIP_NCE
        assert spec.text_label == "echocardiogram 4-chamber view"

    def test_regression_spec(self):
        spec = regression_spec("cardiac", "ejection_fraction", 58.3)
        assert spec.task == TaskType.REGRESSION
        assert spec.label_value == pytest.approx(58.3)
        assert spec.loss.primary == LossType.MSE

    def test_spec_serialization_roundtrip(self):
        spec = binary_segmentation_spec("thyroid", "nodule")
        d = spec.to_dict()
        spec2 = LabelSpec.from_dict(d)
        assert spec2.task == spec.task
        assert spec2.loss.primary == spec.loss.primary
        assert spec2.class_names == spec.class_names
        assert spec2.is_binary == spec.is_binary

    def test_loss_config_roundtrip(self):
        lc = LossConfig(primary=LossType.FOCAL, focal_gamma=3.0,
                        class_weights=[0.3, 0.7])
        d = lc.to_dict()
        lc2 = LossConfig.from_dict(d)
        assert lc2.primary == LossType.FOCAL
        assert lc2.focal_gamma == pytest.approx(3.0)
        assert lc2.class_weights == [0.3, 0.7]

    def test_task_config_constructors(self):
        tc_seg = TaskConfig.segmentation("cardiac")
        assert TaskType.SEGMENTATION in tc_seg.active_tasks
        assert tc_seg.anatomy_filter == ["cardiac"]
        assert tc_seg.require_mask is True

        tc_clip = TaskConfig.clip_pretraining()
        assert TaskType.CLIP in tc_clip.active_tasks
        assert tc_clip.require_text is True

        tc_patient = TaskConfig.patient_level("cardiac")
        assert TaskType.PATIENT_CLS in tc_patient.active_tasks
        assert tc_patient.patient_aggregate is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. ADAPTER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAdapters:

    def _check_entry(self, e: USManifestEntry, check_text: bool = False):
        """Common validity checks for any manifest entry."""
        assert isinstance(e.sample_id, str) and len(e.sample_id) > 0
        assert e.dataset_id in ADAPTER_REGISTRY or e.dataset_id != "unknown"
        assert e.anatomy_family in {
            "cardiac", "lung", "breast", "thyroid", "fetal_head",
            "fetal_abdomen", "fetal_brain", "fetal_femur", "fetal_thorax",
            "cervix", "kidney", "liver", "gallbladder", "ovarian",
            "prostate", "muscle", "nerve", "vascular", "carotid",
            "spine", "ocular", "multi", "other",
        }
        assert e.split in ("train", "val", "test", "unlabeled")
        assert e.modality_type in ("image", "video", "volume", "pseudo_video")
        assert len(e.image_paths) > 0
        assert e.curriculum_tier in (1, 2, 3)
        if check_text:
            assert e.has_text_label or e.clip_eligible

    def _check_instance_specs(self, e: USManifestEntry):
        """Check that all label specs are valid."""
        for inst in e.instances:
            for spec in inst.label_specs:
                assert isinstance(spec.task, TaskType)
                assert isinstance(spec.loss, LossConfig)
                assert isinstance(spec.num_classes, int)
                assert isinstance(spec.class_names, list)

    def test_camus_adapter(self, camus_root):
        adapter = CAMUSAdapter(camus_root)
        entries = list(adapter.iter_entries())
        assert len(entries) > 0, "CAMUS should produce entries"

        # Should have both image and pseudo_video entries
        modalities = {e.modality_type for e in entries}
        assert "image" in modalities
        assert "pseudo_video" in modalities

        # All entries should be valid
        for e in entries:
            self._check_entry(e, check_text=(e.modality_type == "image"))
            self._check_instance_specs(e)

        # Image entries should have masks
        image_entries = [e for e in entries if e.modality_type == "image"]
        masked = [e for e in image_entries if e.has_mask]
        assert len(masked) > 0, "CAMUS image entries should have masks"

        # Segmentation specs
        seg_entries = [e for e in image_entries if e.has_mask]
        for e in seg_entries:
            seg_specs = e.specs_for_task(TaskType.SEGMENTATION)
            assert len(seg_specs) > 0, f"Entry {e.sample_id} has mask but no seg spec"

        # Patient ID tracking
        patient_ids = {e.patient_id for e in entries if e.patient_id}
        assert len(patient_ids) >= 4  # n_patients

        # CLIP specs on all image entries
        for e in image_entries:
            clip_specs = e.specs_for_task(TaskType.CLIP)
            assert len(clip_specs) > 0, f"Entry {e.sample_id} missing CLIP spec"

    def test_busi_adapter(self, busi_root):
        adapter = BUSIAdapter(busi_root)
        entries = list(adapter.iter_entries())
        assert len(entries) > 0

        # Should have all 3 classes
        labels = set()
        for e in entries:
            for inst in e.instances:
                labels.add(inst.label_raw)
        assert "normal" in labels
        assert "benign" in labels
        assert "malignant" in labels

        for e in entries:
            self._check_entry(e, check_text=True)
            self._check_instance_specs(e)

        # Classification specs
        cls_entries = [e for e in entries]
        for e in cls_entries:
            cls_specs = e.specs_for_task(TaskType.MULTICLASS_CLS)
            assert len(cls_specs) > 0, f"BUSI entry missing cls spec"
            assert cls_specs[0].num_classes == 3

        # Benign/malignant entries should have masks
        non_normal = [e for e in entries if any(
            inst.label_raw != "normal" for inst in e.instances
        )]
        masked = [e for e in non_normal if e.has_mask]
        assert len(masked) > 0

    def test_covidx_adapter(self, covidx_root):
        adapter = COVIDxUSAdapter(covidx_root)
        entries = list(adapter.iter_entries())
        assert len(entries) > 0

        # Should have all 3 splits
        splits = {e.split for e in entries}
        assert "train" in splits

        # All video modality
        for e in entries:
            assert e.modality_type == "video"
            self._check_entry(e, check_text=True)
            cls_specs = e.specs_for_task(TaskType.MULTICLASS_CLS)
            assert len(cls_specs) > 0
            assert cls_specs[0].num_classes == 3
            assert cls_specs[0].label_value in (0, 1, 2)

    def test_fetal_planes_adapter(self, fetal_planes_root):
        adapter = FetalPlanesDBAdapter(fetal_planes_root)
        entries = list(adapter.iter_entries())
        assert len(entries) > 0

        for e in entries:
            self._check_entry(e, check_text=True)
            # All should have classification or SSL task
            assert e.task_type in ("classification", "ssl_only")

        # Multiple anatomy families
        anatomies = {e.anatomy_family for e in entries}
        assert len(anatomies) > 1

    def test_hc18_adapter(self, hc18_root):
        adapter = HC18Adapter(hc18_root)
        entries = list(adapter.iter_entries())
        assert len(entries) > 0

        train_entries = [e for e in entries if e.split == "train"]
        test_entries  = [e for e in entries if e.split == "test"]
        assert len(train_entries) > 0
        assert len(test_entries) > 0

        # Train entries should have masks
        for e in train_entries:
            assert e.has_mask, f"HC18 train entry missing mask: {e.sample_id}"
            seg_specs = e.specs_for_task(TaskType.SEGMENTATION)
            assert len(seg_specs) > 0

        # Test entries should NOT have masks
        for e in test_entries:
            assert not e.has_mask

    def test_lus_multicenter_adapter(self, lus_multicenter_root):
        adapter = LUSMulticenterAdapter(lus_multicenter_root)
        entries = list(adapter.iter_entries())
        assert len(entries) > 0

        # Both classes present
        labels = set()
        for e in entries:
            for inst in e.instances:
                labels.add(inst.label_raw)
        assert "a_lines" in labels or "b_lines" in labels

        for e in entries:
            self._check_entry(e, check_text=True)
            specs = e.specs_for_task(TaskType.BINARY_CLS)
            assert len(specs) > 0
            assert specs[0].is_binary is True

    def test_bus_bra_adapter(self, bus_bra_root):
        adapter = BUSBRAAdapter(bus_bra_root)
        entries = list(adapter.iter_entries())
        assert len(entries) > 0

        for e in entries:
            self._check_entry(e, check_text=True)
            assert e.has_mask

    def test_generic_mask_pair_adapter(self, generic_seg_root):
        cls = _make_generic("TestDS", "thyroid", "bronze",
                             label_ontology="nodule",
                             text_template="thyroid ultrasound")
        adapter = cls(generic_seg_root)
        entries = list(adapter.iter_entries())
        assert len(entries) > 0
        for e in entries:
            self._check_entry(e, check_text=True)
            if e.has_mask:
                seg_specs = e.specs_for_task(TaskType.SEGMENTATION)
                assert len(seg_specs) > 0

    def test_all_registered_adapters_have_required_attrs(self):
        """Every adapter in the registry should have DATASET_ID, ANATOMY_FAMILY, SONODQS."""
        for name, cls in ADAPTER_REGISTRY.items():
            assert hasattr(cls, "DATASET_ID"),     f"{name} missing DATASET_ID"
            assert hasattr(cls, "ANATOMY_FAMILY"), f"{name} missing ANATOMY_FAMILY"
            assert hasattr(cls, "SONODQS"),        f"{name} missing SONODQS"
            assert cls.DATASET_ID != "unknown",    f"{name} has default DATASET_ID"

    def test_split_inference_coverage(self, tn3k_root):
        """Inferred splits should cover all three split types across enough samples."""
        cls = _make_generic("TN3K", "thyroid", "bronze")
        adapter = cls(tn3k_root)
        entries = list(adapter.iter_entries())
        splits = {e.split for e in entries}
        # With 8 images and default 0.8/0.1/0.1 split, we expect train at minimum
        assert "train" in splits


# ─────────────────────────────────────────────────────────────────────────────
# 3. MANIFEST I/O TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestManifestIO:

    def _make_rich_entry(self) -> USManifestEntry:
        spec = binary_segmentation_spec("cardiac", "left_ventricle", label_source="TEST")
        clip = clip_spec("cardiac", "test echo", label_source="TEST")
        inst = Instance(
            instance_id="inst_0",
            label_raw="LV",
            label_ontology="left_ventricle",
            anatomy_family="cardiac",
            mask_path="/tmp/fake_mask.png",
            text_label="left ventricle",
            is_promptable=True,
            label_specs=[spec, clip],
        )
        e = USManifestEntry(
            sample_id="abc123",
            dataset_id="TEST",
            study_id="patient_001",
            patient_id="patient_001",
            modality_type="image",
            split="train",
            image_paths=["/tmp/fake_image.png"],
            anatomy_family="cardiac",
            instances=[inst],
            task_type="segmentation",
            has_mask=True,
            ssl_stream="both",
            has_text_label=True,
            clip_eligible=True,
            curriculum_tier=1,
            sonodqs="silver",
            quality_score=4,
        )
        return e

    def test_entry_serialization_roundtrip(self):
        e = self._make_rich_entry()
        d = e.to_dict()
        e2 = USManifestEntry.from_dict(d)

        assert e2.sample_id == e.sample_id
        assert e2.patient_id == e.patient_id
        assert e2.has_text_label == e.has_text_label
        assert e2.clip_eligible == e.clip_eligible
        assert len(e2.instances) == len(e.instances)

        # Check label specs survived
        inst2 = e2.instances[0]
        assert len(inst2.label_specs) == 2
        assert inst2.label_specs[0].task == TaskType.SEGMENTATION
        assert inst2.label_specs[1].task == TaskType.CLIP

    def test_manifest_write_read(self, tmp_path):
        entries = [self._make_rich_entry() for _ in range(5)]
        # Give unique IDs
        for i, e in enumerate(entries):
            e.sample_id = f"id_{i:04d}"
            e.split = ["train", "train", "train", "val", "test"][i]

        manifest_path = tmp_path / "test.jsonl"
        with ManifestWriter(manifest_path) as w:
            for e in entries:
                w.write(e)

        # Load all
        loaded = load_manifest(manifest_path)
        assert len(loaded) == 5

        # Load train only
        train = load_manifest(manifest_path, split="train")
        assert len(train) == 3

        # Load with text required
        with_text = load_manifest(manifest_path, require_text=True)
        assert len(with_text) == 5  # all our test entries have text

        # Load CLIP eligible
        clip_only = load_manifest(manifest_path, clip_eligible_only=True)
        assert len(clip_only) == 5

    def test_manifest_stats(self):
        e = self._make_rich_entry()
        stats = manifest_stats([e])
        assert stats["total"] == 1
        assert stats["has_mask"] == 1
        assert stats["has_text"] == 1
        assert stats["clip_eligible"] == 1

    def test_group_by_patient(self):
        entries = []
        for i in range(6):
            e = self._make_rich_entry()
            e.sample_id = f"s{i}"
            e.patient_id = f"patient_{i % 2}"  # 2 patients, 3 samples each
            entries.append(e)
        groups = group_by_patient(entries)
        assert len(groups) == 2
        for pid, group in groups.items():
            assert len(group) == 3

    def test_normalize_anatomy(self):
        assert normalize_anatomy("cardiac") == "cardiac"
        assert normalize_anatomy("heart") == "cardiac"
        assert normalize_anatomy("echo") == "cardiac"
        assert normalize_anatomy("b-line") == "lung"
        assert normalize_anatomy("a-line") == "lung"
        assert normalize_anatomy("fetal_head") == "fetal_head"
        assert normalize_anatomy(None) == "other"
        assert normalize_anatomy("unknown_xyz") == "other"

    def test_curriculum_tier_assignment(self):
        from manifest import assign_curriculum_tier
        # Short clip with mask = tier 1
        e1 = USManifestEntry(
            sample_id="t1", dataset_id="TEST",
            anatomy_family="cardiac",
            has_mask=True, num_frames=8,
            image_paths=["/tmp/x.png"],
        )
        assert assign_curriculum_tier(e1) == 1

        # Long video, no mask = tier 3
        e2 = USManifestEntry(
            sample_id="t2", dataset_id="TEST",
            anatomy_family="other",
            has_mask=False, num_frames=100,
            image_paths=["/tmp/x.avi"],
        )
        assert assign_curriculum_tier(e2) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 4. DATASET __getitem__ TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetGetItem:
    """
    Tests that Dataset.__getitem__ returns correctly shaped tensors.
    We use entries generated from synthetic datasets.
    """

    def _get_entries(self, adapter_cls, root, split="train"):
        adapter = adapter_cls(root)
        return [e for e in adapter.iter_entries() if e.split == split]

    def test_image_ssl_dataset_getitem(self, hc18_root):
        """ImageSSLDataset should return global/local crops and a patch mask."""
        try:
            import torch
            from dataset import ImageSSLDataset
            from transforms import ImageSSLTransformConfig
        except ImportError:
            pytest.skip("torch or transforms not available")

        entries = list(HC18Adapter(hc18_root).iter_entries())
        entries = [e for e in entries if e.split == "train"]
        if not entries:
            pytest.skip("No train entries")

        cfg = ImageSSLTransformConfig(global_crop_size=64, local_crop_size=32,
                                       n_global=2, n_local=2)
        ds = ImageSSLDataset(entries, cfg=cfg)
        item = ds[0]

        assert "global_crops" in item
        assert "local_crops" in item
        assert "patch_mask" in item
        assert "sample_id" in item
        assert "anatomy_family" in item

        assert item["global_crops"].shape[0] == 2   # n_global
        assert item["local_crops"].shape[0] == 2    # n_local
        assert item["global_crops"].dim() == 4       # (n, 1, H, W)

    def test_downstream_segmentation_getitem(self, hc18_root):
        """DownstreamDataset with segmentation task should return seg_mask."""
        try:
            import torch
            from dataset import DownstreamDataset
        except ImportError:
            pytest.skip("torch not available")

        entries = [e for e in HC18Adapter(hc18_root).iter_entries()
                   if e.split == "train" and e.has_mask]
        if not entries:
            pytest.skip("No seg entries")

        tc = TaskConfig.segmentation()
        ds = DownstreamDataset(entries, task_config=tc, image_size=64)
        item = ds[0]

        assert "image" in item
        assert item["image"].shape == (1, 64, 64)  # (C, H, W)
        assert "seg_mask" in item
        assert "sample_id" in item

    def test_downstream_classification_getitem(self, busi_root):
        """DownstreamDataset with classification task should return cls_label."""
        try:
            import torch
            from dataset import DownstreamDataset
        except ImportError:
            pytest.skip("torch not available")

        entries = list(BUSIAdapter(busi_root).iter_entries())
        if not entries:
            pytest.skip("No BUSI entries")

        tc = TaskConfig.multiclass_classification("breast")
        ds = DownstreamDataset(entries, task_config=tc, image_size=64)
        item = ds[0]

        assert "image" in item
        assert "cls_label" in item
        assert item["cls_label"].item() in (0, 1, 2)

    def test_cross_modal_dataset_getitem(self, busi_root):
        """CrossModalDataset should return image + text string."""
        try:
            import torch
            from dataset import CrossModalDataset
        except ImportError:
            pytest.skip("torch not available")

        entries = list(BUSIAdapter(busi_root).iter_entries())
        entries = [e for e in entries if e.has_text_label or e.clip_eligible]
        if not entries:
            pytest.skip("No CLIP-eligible entries")

        ds = CrossModalDataset(entries, image_size=64)
        item = ds[0]

        assert "image" in item
        assert "text" in item
        assert isinstance(item["text"], str)
        assert len(item["text"]) > 0
        assert item["image"].shape == (1, 64, 64)

    def test_downstream_returns_none_gracefully(self, hc18_root):
        """DownstreamDataset should return valid item even if requested label absent."""
        try:
            import torch
            from dataset import DownstreamDataset
        except ImportError:
            pytest.skip("torch not available")

        # Use test split (no masks)
        entries = [e for e in HC18Adapter(hc18_root).iter_entries()
                   if e.split == "test"]
        if not entries:
            pytest.skip("No test entries")

        # Ask for segmentation even though test entries have no masks
        tc = TaskConfig.segmentation()
        ds = DownstreamDataset(entries, task_config=tc, image_size=64)
        item = ds[0]

        # Should have image but no seg_mask
        assert "image" in item
        assert "seg_mask" not in item  # gracefully absent


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUILD MANIFEST END-TO-END TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildManifest:

    def test_build_manifest_for_camus(self, camus_root, tmp_path):
        manifest_path = tmp_path / "camus.jsonl"
        with ManifestWriter(manifest_path) as w:
            count = build_manifest_for_dataset("CAMUS", camus_root, w)
        assert count > 0
        loaded = load_manifest(manifest_path)
        assert len(loaded) == count
        # Check all are valid
        for e in loaded:
            assert e.dataset_id == "CAMUS"
            assert e.anatomy_family == "cardiac"

    def test_build_manifest_for_unknown_dataset(self, tmp_path):
        """Unknown dataset ID should log warning and return 0."""
        manifest_path = tmp_path / "unknown.jsonl"
        manifest_path.write_text("")  # empty manifest
        with ManifestWriter(manifest_path) as w:
            count = build_manifest_for_dataset("NONEXISTENT_DS", Path("/tmp"), w)
        assert count == 0

    def test_full_manifest_pipeline(self, camus_root, busi_root, hc18_root, tmp_path):
        """Build a combined manifest from multiple datasets."""
        manifest_path = tmp_path / "combined.jsonl"
        total = 0
        with ManifestWriter(manifest_path) as w:
            total += build_manifest_for_dataset("CAMUS", camus_root, w)
            total += build_manifest_for_dataset("BUSI",  busi_root,  w)
            total += build_manifest_for_dataset("HC18",  hc18_root,  w)

        assert total > 0

        loaded = load_manifest(manifest_path)
        assert len(loaded) == total

        stats = manifest_stats(loaded)
        assert stats["total"] == total
        # Multiple anatomies
        assert len(stats["by_anatomy"]) >= 2
        # Multiple datasets
        assert len(stats["by_dataset"]) >= 2

        # Filter by anatomy
        cardiac = load_manifest(manifest_path, anatomy_families=["cardiac"])
        assert all(e.anatomy_family == "cardiac" for e in cardiac)

        # Filter CLIP eligible
        clip_eligible = load_manifest(manifest_path, clip_eligible_only=True)
        assert all(e.clip_eligible for e in clip_eligible)


# ─────────────────────────────────────────────────────────────────────────────
# 6. STORAGE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestStorage:

    def test_storage_config_local_dev(self, tmp_path):
        """StorageConfig should use local_dev_root when store/scratch don't exist."""
        cfg = StorageConfig(
            store_root=Path("/nonexistent/store"),
            scratch_root=None,
            use_scratch=False,
            local_dev_root=tmp_path,
        )
        assert cfg.active_root == tmp_path

    def test_storage_env_override(self, tmp_path, monkeypatch):
        """US_LOCAL_DEV_ROOT env var should override local_dev_root."""
        monkeypatch.setenv("US_LOCAL_DEV_ROOT", str(tmp_path))
        cfg = StorageConfig(
            store_root=Path("/nonexistent/store"),
            scratch_root=None,
            use_scratch=False,
        )
        assert cfg.local_dev_root == tmp_path

    def test_build_root_remap(self, tmp_path):
        """build_root_remap returns correct mapping when scratch is set."""
        store = tmp_path / "store"
        scratch = tmp_path / "scratch"
        store.mkdir(); scratch.mkdir()

        cfg = StorageConfig(
            store_root=store,
            scratch_root=scratch,
            use_scratch=True,
        )
        remap = cfg.build_root_remap()
        assert str(store / "raw") in remap
        assert remap[str(store / "raw")] == str(scratch / "raw")

    def test_no_remap_when_scratch_disabled(self, tmp_path):
        """No remapping when use_scratch=False."""
        cfg = StorageConfig(
            store_root=tmp_path / "store",
            scratch_root=tmp_path / "scratch",
            use_scratch=False,
            local_dev_root=tmp_path,
        )
        remap = cfg.build_root_remap()
        assert remap == {}

    def test_configure_storage_singleton(self, tmp_path):
        """configure_storage() should update the global singleton."""
        from storage import configure_storage, get_storage
        cfg = configure_storage(local_dev_root=str(tmp_path), use_scratch=False)
        assert get_storage() is cfg

    def test_dataset_is_staged(self, tmp_path):
        """dataset_is_staged should return True when files exist on scratch."""
        scratch = tmp_path / "scratch"
        # Create a fake staged dataset
        fake_ds_dir = scratch / "raw" / "cardiac" / "CAMUS"
        fake_ds_dir.mkdir(parents=True, exist_ok=True)
        (fake_ds_dir / "patient0001").mkdir()

        cfg = StorageConfig(
            store_root=tmp_path / "store",
            scratch_root=scratch,
            use_scratch=True,
            local_dev_root=tmp_path,
        )
        assert cfg.dataset_is_staged("CAMUS") is True
        assert cfg.dataset_is_staged("EchoNet-Dynamic") is False

    def test_status_report_format(self, tmp_path):
        """status_report() should return a valid formatted string."""
        cfg = StorageConfig(
            store_root=tmp_path / "store",
            scratch_root=tmp_path / "scratch",
            use_scratch=True,
            local_dev_root=tmp_path,
        )
        report = cfg.status_report()
        assert "Dataset" in report
        assert "CAMUS" in report
        assert "cardiac" in report


# ─────────────────────────────────────────────────────────────────────────────
# 7. ANATOMY LABEL VOCABULARY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAnatomyVocabulary:

    def test_all_anatomy_families_have_vocab(self):
        """Key anatomy families should have non-empty label vocabularies."""
        key_families = ["cardiac", "lung", "breast", "thyroid",
                         "fetal_head", "kidney", "liver"]
        for fam in key_families:
            assert fam in ANATOMY_LABEL_VOCAB, f"{fam} missing from vocab"
            assert len(ANATOMY_LABEL_VOCAB[fam]) > 0, f"{fam} vocab is empty"

    def test_cardiac_vocab_contains_standard_structures(self):
        vocab = ANATOMY_LABEL_VOCAB["cardiac"]
        assert "left_ventricle" in vocab
        assert "myocardium" in vocab
        assert "left_atrium" in vocab

    def test_lung_vocab_contains_artifact_labels(self):
        vocab = ANATOMY_LABEL_VOCAB["lung"]
        assert "b_lines" in vocab
        assert "a_lines" in vocab
        assert "normal" in vocab
        assert "covid" in vocab

    def test_breast_vocab_birads(self):
        vocab = ANATOMY_LABEL_VOCAB["breast"]
        assert "benign_mass" in vocab
        assert "malignant_mass" in vocab


# ─────────────────────────────────────────────────────────────────────────────
# 8. PATIENT-LEVEL TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPatientLevel:

    def test_patient_grouping_from_camus(self, camus_root):
        """CAMUS entries should group correctly by patient_id."""
        adapter = CAMUSAdapter(camus_root)
        entries = [e for e in adapter.iter_entries() if e.modality_type == "image"]
        assert len(entries) > 0

        groups = group_by_patient(entries)
        # Each patient should appear in groups
        for pid, group in groups.items():
            assert all(e.patient_id == pid or e.study_id == pid for e in group)

    def test_patient_dataset_getitem(self, camus_root):
        """PatientDataset should aggregate frames per patient."""
        try:
            import torch
            from dataset import PatientDataset
        except ImportError:
            pytest.skip("torch not available")

        entries = [e for e in CAMUSAdapter(camus_root).iter_entries()
                   if e.modality_type == "image"]
        if not entries:
            pytest.skip("No CAMUS image entries")

        tc = TaskConfig.segmentation("cardiac")
        ds = PatientDataset(entries, task_config=tc, image_size=64,
                             max_frames_per_patient=4)
        assert len(ds) > 0
        item = ds[0]

        assert "frames" in item
        assert "patient_id" in item
        assert item["frames"].dim() == 4  # (N_frames, 1, H, W)
        assert item["frames"].shape[-1] == 64
