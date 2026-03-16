"""
tests/test_label_spec.py  ·  Unit tests for the label specification system
==========================================================================
"""
import pytest
import sys
from pathlib import Path

# ── LabelSpec system ──────────────────────────────────────────────────────────

def test_label_spec_lookup_known():
    from label_spec import get_label_spec, LossType, LabelGranularity

    spec = get_label_spec("CAMUS", "segmentation")
    assert spec is not None
    assert spec.loss_type == LossType.CE
    assert spec.n_classes == 4
    assert spec.granularity == LabelGranularity.PIXEL
    assert "left_ventricle" in spec.class_names


def test_label_spec_lookup_hc18_regression():
    from label_spec import get_label_spec, LossType, LabelGranularity

    spec = get_label_spec("HC18", "regression")
    assert spec is not None
    assert spec.loss_type == LossType.HUBER
    assert spec.n_classes == 1
    assert spec.normalize_target is not None
    assert spec.is_regression is True


def test_label_spec_lookup_clip_fallback():
    """Unknown dataset should fall back to generic CLIP spec."""
    from label_spec import get_label_spec, LossType

    spec = get_label_spec("NONEXISTENT_DATASET", "clip")
    assert spec is not None
    assert spec.loss_type == LossType.CLIP
    assert spec.is_clip is True


def test_label_spec_lookup_unknown_task_returns_none():
    from label_spec import get_label_spec

    spec = get_label_spec("CAMUS", "nonexistent_task")
    assert spec is None


def test_label_spec_binary_properties():
    from label_spec import get_label_spec

    spec = get_label_spec("BUSI", "segmentation")
    assert spec is not None
    assert spec.is_binary is True
    assert spec.n_classes == 1


def test_label_spec_multilabel_properties():
    from label_spec import get_label_spec

    spec = get_label_spec("MIMIC-IV-ECHO", "classification")
    assert spec is not None
    assert spec.is_multilabel is True
    assert spec.n_classes == 8


def test_label_spec_serialization():
    """LabelSpec should survive to_dict() → from_dict() round-trip."""
    from label_spec import get_label_spec, LabelSpec

    original = get_label_spec("CAMUS", "segmentation")
    d = original.to_dict()
    assert isinstance(d, dict)
    restored = LabelSpec.from_dict(d)
    assert restored.task_name == original.task_name
    assert restored.loss_type == original.loss_type
    assert restored.n_classes == original.n_classes
    assert restored.class_names == original.class_names


def test_anatomy_ontology_consistency():
    """All anatomies in ANATOMY_SEG_ONTOLOGY must have background at index 0."""
    from label_spec import ANATOMY_SEG_ONTOLOGY

    for anatomy, classes in ANATOMY_SEG_ONTOLOGY.items():
        assert "background" in classes, f"'{anatomy}' missing background class"
        assert classes["background"] == 0, (
            f"'{anatomy}': background must have class ID 0"
        )


def test_ontology_ids_no_overlap():
    """Within one anatomy, class IDs must be unique."""
    from label_spec import ANATOMY_SEG_ONTOLOGY

    for anatomy, classes in ANATOMY_SEG_ONTOLOGY.items():
        ids = list(classes.values())
        assert len(ids) == len(set(ids)), (
            f"Duplicate class IDs in '{anatomy}' ontology: {ids}"
        )


def test_get_ontology_class_ids():
    from label_spec import get_ontology_class_ids

    ids = get_ontology_class_ids("cardiac", ["background", "left_ventricle", "myocardium"])
    assert ids == [0, 1, 2]


def test_generate_clip_text_with_anatomy():
    from label_spec import generate_clip_text

    text = generate_clip_text(
        anatomy_family="lung",
        dataset_id="COVIDx-US",
        label_raw=["covid"],
    )
    assert len(text) > 5
    assert isinstance(text, str)


def test_generate_clip_text_with_report():
    from label_spec import generate_clip_text

    report = "Echocardiogram shows normal LV function with EF 55%."
    text = generate_clip_text(
        anatomy_family="cardiac",
        dataset_id="MIMIC-IV-ECHO",
        report_text=report,
    )
    assert text == report


def test_generate_clip_text_fallback():
    """Unknown anatomy should use 'other' templates."""
    from label_spec import generate_clip_text

    text = generate_clip_text(
        anatomy_family="unknown_organ",
        dataset_id="UNKNOWN_DS",
    )
    assert len(text) > 5


def test_list_available_tasks():
    from label_spec import list_available_tasks

    tasks = list_available_tasks("CAMUS")
    assert "segmentation" in tasks
    assert "clip" in tasks


def test_all_segmentation_datasets():
    from label_spec import all_segmentation_datasets

    ds = all_segmentation_datasets()
    assert "CAMUS" in ds
    assert "BUSI" in ds
    assert "HC18" in ds


# ── Downstream dataset tests ───────────────────────────────────────────────────

class TestSegmentationDataset:
    def test_binary_seg_output_shape(self, tmp_manifest_with_masks):
        from manifest import load_manifest
        from label_spec import get_label_spec
        from downstream_dataset import SegmentationDataset

        entries = load_manifest(tmp_manifest_with_masks, split="train")
        spec = get_label_spec("TN3K", "segmentation")
        if spec is None:
            from label_spec import LabelSpec, LossType, LabelGranularity
            spec = LabelSpec(
                task_name="test_seg", loss_type=LossType.DICE_BCE,
                n_classes=1, class_names=["nodule"],
                granularity=LabelGranularity.PIXEL,
            )

        ds = SegmentationDataset(entries, spec, image_size=64)
        assert len(ds) > 0

        sample = ds[0]
        assert "image" in sample
        assert "seg_mask" in sample
        assert sample["image"].shape == (1, 64, 64)
        if sample["seg_mask"] is not None:
            assert sample["seg_mask"].shape == (1, 64, 64)

    def test_dataset_is_filterable(self, tmp_manifest_with_masks):
        from manifest import load_manifest
        from label_spec import get_label_spec, LabelSpec, LossType, LabelGranularity
        from downstream_dataset import SegmentationDataset

        all_e = load_manifest(tmp_manifest_with_masks)
        spec = LabelSpec(
            task_name="test_seg", loss_type=LossType.DICE_BCE,
            n_classes=1, class_names=["nodule"],
            granularity=LabelGranularity.PIXEL,
        )
        ds = SegmentationDataset(all_e, spec, split="train")
        assert all(e.split == "train" for e in ds.entries)


class TestClassificationDataset:
    def test_multiclass_output(self, tmp_manifest_with_cls):
        from manifest import load_manifest
        from label_spec import get_label_spec
        from downstream_dataset import ClassificationDataset

        entries = load_manifest(tmp_manifest_with_cls, split="train")
        spec = get_label_spec("COVIDx-US", "classification")

        ds = ClassificationDataset(entries, spec, image_size=64)
        assert len(ds) > 0

        sample = ds[0]
        assert sample["image"].shape[0] == 1  # grayscale or (T, 1, H, W)
        assert "cls_label" in sample

    def test_binary_cls(self, tmp_manifest_with_masks):
        import torch
        from manifest import load_manifest
        from label_spec import LabelSpec, LossType, LabelGranularity
        from downstream_dataset import ClassificationDataset

        entries = load_manifest(tmp_manifest_with_masks)
        spec = LabelSpec(
            task_name="binary_test", loss_type=LossType.BCE,
            n_classes=1, class_names=["positive"],
            granularity=LabelGranularity.FRAME,
        )
        ds = ClassificationDataset(entries, spec, image_size=64)
        for i in range(min(len(ds), 3)):
            sample = ds[i]
            if sample["cls_label"] is not None:
                assert sample["cls_label"].dtype == torch.float32


class TestCLIPDataset:
    def test_text_always_generated(self, tmp_manifest_with_masks):
        from manifest import load_manifest
        from downstream_dataset import CLIPDataset

        entries = load_manifest(tmp_manifest_with_masks)
        ds = CLIPDataset(entries, image_size=64)

        for i in range(min(len(ds), 3)):
            sample = ds[i]
            assert "image" in sample
            assert "text" in sample
            assert isinstance(sample["text"], str)
            assert len(sample["text"]) > 5

    def test_image_shape(self, tmp_manifest_with_masks):
        from manifest import load_manifest
        from downstream_dataset import CLIPDataset

        entries = load_manifest(tmp_manifest_with_masks)
        ds = CLIPDataset(entries, image_size=64)
        sample = ds[0]
        assert sample["image"].shape == (1, 64, 64)


class TestPatientLevelDataset:
    def test_grouping(self, tmp_manifest_with_masks):
        from manifest import load_manifest
        from label_spec import LabelSpec, LossType, LabelGranularity
        from downstream_dataset import PatientLevelDataset

        entries = load_manifest(tmp_manifest_with_masks)
        spec = LabelSpec(
            task_name="patient_test", loss_type=LossType.BCE,
            n_classes=1, class_names=["label"],
            granularity=LabelGranularity.PATIENT,
        )
        ds = PatientLevelDataset(entries, spec, image_size=64)
        assert len(ds) >= 1

        sample = ds[0]
        assert "images" in sample
        assert "patient_id" in sample
        assert "n_views" in sample
        assert sample["images"].ndim == 4  # (N_views, 1, H, W)


class TestRegressionDataset:
    def test_creates_empty_without_labels(self, tmp_manifest_with_masks):
        """When no regression labels exist, dataset should be empty."""
        from manifest import load_manifest
        from label_spec import get_label_spec
        from downstream_dataset import RegressionDataset

        entries = load_manifest(tmp_manifest_with_masks)
        spec = get_label_spec("HC18", "regression")
        ds = RegressionDataset(entries, spec, image_size=64)
        # No HC18 labels in our synthetic thyroid manifest → 0 samples
        assert len(ds) == 0


# ── Collator tests ────────────────────────────────────────────────────────────

class TestDownstreamCollator:
    def test_images_stacked(self, tmp_manifest_with_masks):
        import torch
        from manifest import load_manifest
        from label_spec import LabelSpec, LossType, LabelGranularity
        from downstream_dataset import SegmentationDataset, DownstreamCollator

        entries = load_manifest(tmp_manifest_with_masks)
        spec = LabelSpec(
            task_name="t", loss_type=LossType.DICE_BCE,
            n_classes=1, class_names=["c"],
            granularity=LabelGranularity.PIXEL,
        )
        ds = SegmentationDataset(entries, spec, image_size=64)
        collator = DownstreamCollator()

        samples = [ds[i] for i in range(min(2, len(ds)))]
        if not samples:
            pytest.skip("No samples available")

        batch = collator(samples)
        assert "image" in batch
        assert batch["image"].shape[0] == len(samples)

    def test_none_masks_handled(self):
        import torch
        from downstream_dataset import DownstreamCollator

        collator = DownstreamCollator()
        samples = [
            {"image": torch.zeros(1, 64, 64), "seg_mask": None,
             "sample_id": "a", "dataset_id": "X", "anatomy_family": "lung"},
            {"image": torch.zeros(1, 64, 64), "seg_mask": None,
             "sample_id": "b", "dataset_id": "X", "anatomy_family": "lung"},
        ]
        batch = collator(samples)
        assert batch["seg_masks"] is None


class TestCLIPCollator:
    def test_output_keys(self, tmp_manifest_with_masks):
        import torch
        from manifest import load_manifest
        from downstream_dataset import CLIPDataset, CLIPCollator

        entries = load_manifest(tmp_manifest_with_masks)
        ds = CLIPDataset(entries, image_size=64)
        collator = CLIPCollator()

        samples = [ds[i] for i in range(min(2, len(ds)))]
        if not samples:
            pytest.skip("No samples")

        batch = collator(samples)
        assert "images" in batch
        assert "texts" in batch
        assert isinstance(batch["texts"], list)
        assert batch["images"].shape[0] == len(samples)


# ── build_downstream_loader ────────────────────────────────────────────────────

def test_build_downstream_loader_segmentation(tmp_manifest_with_masks):
    from torch.utils.data import DataLoader
    from downstream_dataset import build_downstream_loader

    loader = build_downstream_loader(
        manifest_path=tmp_manifest_with_masks,
        task="segmentation",
        dataset_ids=["TEST_SEG"],
        split="train",
        batch_size=2,
        num_workers=0,
        image_size=64,
        label_spec_override=None,
    )
    assert isinstance(loader, DataLoader)
    # Check we can iterate at least once
    batch = next(iter(loader), None)
    if batch is not None:
        assert "image" in batch


def test_build_downstream_loader_classification(tmp_manifest_with_cls):
    from torch.utils.data import DataLoader
    from downstream_dataset import build_downstream_loader

    loader = build_downstream_loader(
        manifest_path=tmp_manifest_with_cls,
        task="classification",
        dataset_ids=["COVIDx-US"],
        split="train",
        batch_size=2,
        num_workers=0,
        image_size=64,
    )
    assert isinstance(loader, DataLoader)


def test_build_downstream_loader_clip(tmp_manifest_with_masks):
    from torch.utils.data import DataLoader
    from downstream_dataset import build_downstream_loader

    loader = build_downstream_loader(
        manifest_path=tmp_manifest_with_masks,
        task="clip",
        split="train",
        batch_size=2,
        num_workers=0,
        image_size=64,
    )
    assert isinstance(loader, DataLoader)
    batch = next(iter(loader), None)
    if batch is not None:
        assert "images" in batch
        assert "texts" in batch


def test_build_downstream_loader_raises_on_empty(tmp_manifest_with_masks):
    from downstream_dataset import build_downstream_loader

    with pytest.raises(ValueError, match="No entries found"):
        build_downstream_loader(
            manifest_path=tmp_manifest_with_masks,
            task="segmentation",
            dataset_ids=["NONEXISTENT"],
            split="train",
            num_workers=0,
        )
