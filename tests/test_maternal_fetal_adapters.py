"""
tests/test_maternal_fetal_adapters.py  ·  Unit tests for maternal/fetal adapters

Run with:
    pytest tests/test_maternal_fetal_adapters.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.adapters.maternal_fetal.acouslic                   import ACOUSLICAIAdapter
from data.adapters.maternal_fetal.fetal_abdominal_structures import FASSAdapter
from data.adapters.maternal_fetal.fetal_planes_db            import FetalPlanesDBAdapter
from data.adapters.maternal_fetal.focus                      import FOCUSAdapter
from data.adapters.maternal_fetal.fpus23                     import FPUS23Adapter
from data.adapters.maternal_fetal.fh_ps_aop                  import FHPSAOPAdapter
from data.adapters.maternal_fetal.hc18                       import HC18Adapter
from data.adapters.maternal_fetal.iugc2024                   import IUGC2024Adapter
from data.adapters.maternal_fetal.jnu_ifm                    import JNUIFMAdapter
from data.adapters.maternal_fetal.psfhs                      import PSFHSAdapter


# ══════════════════════════════════════════════════════════════════════════════
# ACOUSLIC-AI
# ══════════════════════════════════════════════════════════════════════════════

def test_acouslic_yields_all_sweeps(acouslic_root: Path):
    entries = list(ACOUSLICAIAdapter(acouslic_root).iter_entries())
    # All 4 .mha files must be yielded, including the mask-less one.
    assert len(entries) == 4


def test_acouslic_schema(acouslic_root: Path):
    entries = list(ACOUSLICAIAdapter(acouslic_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "ACOUSLIC-AI"
        assert e.anatomy_family == "fetal_abdomen"
        assert e.modality_type == "volume"
        assert e.is_3d is True
        assert e.num_frames == 840
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)
        assert len(e.image_paths) == 1


def test_acouslic_segmentation_entries(acouslic_root: Path):
    entries = {e.series_id: e for e in ACOUSLICAIAdapter(acouslic_root).iter_entries()}

    for uuid in ("sweep-aaa", "sweep-bbb", "sweep-ccc"):
        e = entries[uuid]
        assert e.has_mask is True
        assert e.task_type == "segmentation"
        assert e.is_promptable is True
        assert len(e.instances) == 1
        assert e.instances[0].mask_path is not None
        assert e.instances[0].label_ontology == "fetal_abdomen"


def test_acouslic_ssl_only_entry(acouslic_root: Path):
    entries = {e.series_id: e for e in ACOUSLICAIAdapter(acouslic_root).iter_entries()}
    e = entries["sweep-ddd"]
    assert e.has_mask is False
    assert e.task_type == "ssl_only"
    assert e.is_promptable is False
    assert e.instances[0].mask_path is None


def test_acouslic_measurement_mm(acouslic_root: Path):
    entries = {e.series_id: e for e in ACOUSLICAIAdapter(acouslic_root).iter_entries()}

    assert entries["sweep-aaa"].instances[0].measurement_mm == pytest.approx(250.0)
    assert entries["sweep-bbb"].instances[0].measurement_mm == pytest.approx(252.0)
    assert entries["sweep-ccc"].instances[0].measurement_mm == pytest.approx(270.0)
    assert entries["sweep-ddd"].instances[0].measurement_mm is None


def test_acouslic_subject_level_splitting(acouslic_root: Path):
    entries = {e.series_id: e for e in ACOUSLICAIAdapter(acouslic_root).iter_entries()}

    # Both sweeps from subject 01 must receive the same split.
    assert entries["sweep-aaa"].split == entries["sweep-bbb"].split

    # study_id should be the subject, series_id the sweep uuid.
    assert entries["sweep-aaa"].study_id == "1"   # leading zero stripped
    assert entries["sweep-bbb"].study_id == "1"
    assert entries["sweep-ccc"].study_id == "2"


def test_acouslic_source_meta(acouslic_root: Path):
    entries = {e.series_id: e for e in ACOUSLICAIAdapter(acouslic_root).iter_entries()}
    e = entries["sweep-aaa"]
    assert e.source_meta["uuid"] == "sweep-aaa"
    assert e.source_meta["subject_id"] == "1"
    assert e.source_meta["ac_mm"] == pytest.approx(250.0)


def test_acouslic_resolve_root_direct(acouslic_root: Path):
    # Adapter must also accept the inner acouslic-ai-train-set/ path directly.
    inner = acouslic_root / "acouslic-ai-train-set"
    entries = list(ACOUSLICAIAdapter(inner).iter_entries())
    assert len(entries) == 4


def test_acouslic_split_override(acouslic_root: Path):
    entries = list(ACOUSLICAIAdapter(acouslic_root, split_override="val").iter_entries())
    assert all(e.split == "val" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# FASS
# ══════════════════════════════════════════════════════════════════════════════

def test_fass_yields_all_images(fass_root: Path):
    entries = list(FASSAdapter(fass_root).iter_entries())
    # All 6 images must be yielded, including the mask-less one.
    assert len(entries) == 6


def test_fass_schema(fass_root: Path):
    entries = list(FASSAdapter(fass_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "FASS"
        assert e.anatomy_family == "fetal_abdomen"
        assert e.modality_type == "image"
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)
        assert len(e.image_paths) == 1


def test_fass_multi_structure_instances(fass_root: Path):
    entries = {e.series_id: e for e in FASSAdapter(fass_root).iter_entries()}

    for stem in ("P01_IMG1", "P01_IMG2", "P02_IMG1", "P02_IMG2", "P03_IMG1"):
        e = entries[stem]
        assert e.has_mask is True
        assert e.task_type == "segmentation"
        assert e.is_promptable is True
        assert len(e.instances) == 4  # one per structure

        ontologies = {inst.label_ontology for inst in e.instances}
        assert ontologies == {
            "fetal_abdominal_artery", "fetal_liver", "fetal_stomach", "fetal_abdominal_vein"
        }

        channels = {inst.mask_channel for inst in e.instances}
        assert channels == {0, 1, 2, 3}

        # All instances share the same NPY mask_path
        mask_paths = {inst.mask_path for inst in e.instances}
        assert len(mask_paths) == 1
        assert mask_paths.pop().endswith(".npy")


def test_fass_ssl_only_entry(fass_root: Path):
    entries = {e.series_id: e for e in FASSAdapter(fass_root).iter_entries()}
    e = entries["P03_IMG2"]
    assert e.has_mask is False
    assert e.task_type == "ssl_only"
    assert e.is_promptable is False
    assert len(e.instances) == 0


def test_fass_patient_level_splitting(fass_root: Path):
    entries = {e.series_id: e for e in FASSAdapter(fass_root).iter_entries()}

    # Both images of the same patient must share the same split.
    assert entries["P01_IMG1"].split == entries["P01_IMG2"].split
    assert entries["P02_IMG1"].split == entries["P02_IMG2"].split

    # study_id is the patient number string, series_id is the full stem.
    assert entries["P01_IMG1"].study_id == "01"
    assert entries["P01_IMG1"].series_id == "P01_IMG1"


def test_fass_resolve_root_direct(fass_root: Path):
    # Adapter must also accept the inner long-named subdirectory directly.
    inner = fass_root / "Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images"
    entries = list(FASSAdapter(inner).iter_entries())
    assert len(entries) == 6


def test_fass_split_override(fass_root: Path):
    entries = list(FASSAdapter(fass_root, split_override="test").iter_entries())
    assert all(e.split == "test" for e in entries)


def test_fass_grayscale_flag(tmp_path: Path):
    # Build a minimal dataset with one RGB and one grayscale PNG.
    inner = tmp_path / "Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images"
    img_dir  = inner / "IMAGES"
    mask_dir = inner / "ARRAY_FORMAT"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    from PIL import Image as PILImage
    import numpy as np

    PILImage.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(img_dir / "P99_IMG1.png")
    PILImage.fromarray(np.zeros((4, 4),    dtype=np.uint8), mode="L").save(img_dir / "P99_IMG2.png")

    entries = {e.series_id: e for e in FASSAdapter(tmp_path).iter_entries()}
    assert entries["P99_IMG1"].source_meta["is_grayscale"] is False
    assert entries["P99_IMG2"].source_meta["is_grayscale"] is True


# ══════════════════════════════════════════════════════════════════════════════
# FH-PS-AOP
# ══════════════════════════════════════════════════════════════════════════════

def test_fh_ps_aop_yields_all_images(fh_ps_aop_root: Path):
    entries = list(FHPSAOPAdapter(fh_ps_aop_root).iter_entries())
    assert len(entries) == 6


def test_fh_ps_aop_schema(fh_ps_aop_root: Path):
    entries = list(FHPSAOPAdapter(fh_ps_aop_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "FH-PS-AOP"
        assert e.anatomy_family == "intrapartum"
        assert e.modality_type == "image"
        assert e.view_type == "intrapartum_transperineal"
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)


def test_fh_ps_aop_segmentation_entries(fh_ps_aop_root: Path):
    entries = {e.study_id: e for e in FHPSAOPAdapter(fh_ps_aop_root).iter_entries()
               if e.has_mask}
    assert len(entries) == 5

    for e in entries.values():
        assert e.task_type == "segmentation"
        assert e.is_promptable is True
        assert len(e.instances) == 2

        ontologies = {inst.label_ontology for inst in e.instances}
        assert ontologies == {"pubic_symphysis", "fetal_head"}

        channels = {inst.mask_channel for inst in e.instances}
        assert channels == {1, 2}

        # Both instances share the same mask_path
        mask_paths = {inst.mask_path for inst in e.instances}
        assert len(mask_paths) == 1
        assert mask_paths.pop().endswith(".mha")


def test_fh_ps_aop_ssl_only_entry(fh_ps_aop_root: Path):
    entries = {e.study_id: e for e in FHPSAOPAdapter(fh_ps_aop_root).iter_entries()}
    e = entries["00006"]
    assert e.has_mask is False
    assert e.task_type == "ssl_only"
    assert e.is_promptable is False
    assert len(e.instances) == 0


def test_fh_ps_aop_resolve_root_direct(fh_ps_aop_root: Path):
    inner = fh_ps_aop_root / "Pubic Symphysis-Fetal Head Segmentation and Angle of Progression"
    entries = list(FHPSAOPAdapter(inner).iter_entries())
    assert len(entries) == 6


def test_fh_ps_aop_split_override(fh_ps_aop_root: Path):
    entries = list(FHPSAOPAdapter(fh_ps_aop_root, split_override="val").iter_entries())
    assert all(e.split == "val" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# HC18
# ══════════════════════════════════════════════════════════════════════════════

def test_hc18_yields_all_images(hc18_root: Path):
    entries = list(HC18Adapter(hc18_root).iter_entries())
    # 4 training + 2 test
    assert len(entries) == 6


def test_hc18_schema(hc18_root: Path):
    entries = list(HC18Adapter(hc18_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "HC18"
        assert e.anatomy_family == "fetal_head"
        assert e.modality_type == "image"
        assert e.view_type == "fetal_head_standard_plane"
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)


def test_hc18_training_segmentation_entries(hc18_root: Path):
    entries = {e.series_id: e for e in HC18Adapter(hc18_root).iter_entries()}

    for stem in ("001_HC", "001_2HC", "002_HC"):
        e = entries[stem]
        assert e.has_mask is True
        assert e.task_type == "regression"
        assert e.is_promptable is True
        assert len(e.instances) == 1
        assert e.instances[0].label_ontology == "head_circumference"
        assert e.instances[0].mask_path is not None
        assert e.source_meta["annotation_type"] == "contour"


def test_hc18_measurement_mm(hc18_root: Path):
    entries = {e.series_id: e for e in HC18Adapter(hc18_root).iter_entries()}
    assert entries["001_HC"].instances[0].measurement_mm  == pytest.approx(178.3)
    assert entries["001_2HC"].instances[0].measurement_mm == pytest.approx(179.1)
    assert entries["002_HC"].instances[0].measurement_mm  == pytest.approx(185.0)


def test_hc18_pixel_size_in_meta(hc18_root: Path):
    entries = {e.series_id: e for e in HC18Adapter(hc18_root).iter_entries()}
    assert entries["001_HC"].source_meta["pixel_size_mm"] == pytest.approx(0.154)
    assert entries["004_HC"].source_meta["pixel_size_mm"] == pytest.approx(0.158)


def test_hc18_patient_level_splitting(hc18_root: Path):
    entries = {e.series_id: e for e in HC18Adapter(hc18_root).iter_entries()}
    # Both sweeps of patient 001 must share the same split.
    assert entries["001_HC"].split == entries["001_2HC"].split
    # study_id is the patient prefix; series_id is the full stem.
    assert entries["001_HC"].study_id  == "001"
    assert entries["001_2HC"].study_id == "001"
    assert entries["002_HC"].study_id  == "002"


def test_hc18_test_entries(hc18_root: Path):
    entries = {e.series_id: e for e in HC18Adapter(hc18_root).iter_entries()}
    for stem in ("004_HC", "005_HC"):
        e = entries[stem]
        assert e.split == "test"
        assert e.task_type == "ssl_only"
        assert e.has_mask is False
        assert len(e.instances) == 0


def test_hc18_ssl_only_training_entry(hc18_root: Path):
    entries = {e.series_id: e for e in HC18Adapter(hc18_root).iter_entries()}
    e = entries["003_HC"]
    assert e.has_mask is False
    assert e.task_type == "ssl_only"


def test_hc18_split_override(hc18_root: Path):
    entries = list(HC18Adapter(hc18_root, split_override="val").iter_entries())
    assert all(e.split == "val" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# FETAL-PLANES-DB
# ══════════════════════════════════════════════════════════════════════════════

def test_fetal_planes_db_yields_existing_images(fetal_planes_root: Path):
    entries = list(FetalPlanesDBAdapter(fetal_planes_root).iter_entries())
    # 6 image files exist; one extra metadata row points to a missing image.
    assert len(entries) == 6


def test_fetal_planes_db_schema(fetal_planes_root: Path):
    entries = list(FetalPlanesDBAdapter(fetal_planes_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "FETAL_PLANES_DB"
        assert e.anatomy_family == "fetal_planes"
        assert e.modality_type == "image"
        assert e.task_type == "classification"
        assert e.has_mask is False
        assert e.is_promptable is False
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)
        assert len(e.instances) == 1


def test_fetal_planes_db_classification_labels(fetal_planes_root: Path):
    entries = {e.instances[0].label_raw: e for e in FetalPlanesDBAdapter(fetal_planes_root).iter_entries()}

    assert entries["Other"].instances[0].classification_label == 0
    assert entries["Fetal brain"].instances[0].classification_label == 1
    assert entries["Fetal abdomen"].instances[0].classification_label == 2
    assert entries["Fetal femur"].instances[0].classification_label == 3
    assert entries["Fetal thorax"].instances[0].classification_label == 4
    assert entries["Maternal cervix"].instances[0].classification_label == 5


def test_fetal_planes_db_uses_official_split_column(fetal_planes_root: Path):
    entries = {e.series_id: e for e in FetalPlanesDBAdapter(fetal_planes_root).iter_entries()}

    assert entries["Patient00001_Plane1_1_of_2"].split == "train"
    assert entries["Patient00001_Plane2_2_of_2"].split == "train"
    assert entries["Patient00004_Plane5_1_of_1"].split == "test"
    assert entries["Patient00005_Plane6_1_of_1"].split == "test"


def test_fetal_planes_db_patient_metadata(fetal_planes_root: Path):
    entries = {e.series_id: e for e in FetalPlanesDBAdapter(fetal_planes_root).iter_entries()}

    brain = entries["Patient00001_Plane2_2_of_2"]
    assert brain.study_id == "1"
    assert brain.view_type == "fetal_brain"
    assert brain.source_meta["brain_plane"] == "Trans-thalamic"
    assert brain.source_meta["operator"] == "Op. 1"
    assert brain.source_meta["us_machine"] == "Voluson E6"
    assert brain.source_meta["train_flag"] == "1"
    assert brain.source_meta["image_format"] == "rgba_png"


def test_fetal_planes_db_resolve_root_from_parent(fetal_planes_root: Path):
    entries = list(FetalPlanesDBAdapter(fetal_planes_root.parent).iter_entries())
    assert len(entries) == 6


def test_fetal_planes_db_split_override(fetal_planes_root: Path):
    entries = list(FetalPlanesDBAdapter(fetal_planes_root, split_override="val").iter_entries())
    assert all(e.split == "val" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# FPUS23
# ══════════════════════════════════════════════════════════════════════════════

def test_fpus23_yields_pose_frames_and_plane_images(fpus23_root: Path):
    entries = list(FPUS23Adapter(fpus23_root).iter_entries())
    pose_entries = [e for e in entries if e.source_meta["sub_dataset"] == "Dataset"]
    plane_entries = [e for e in entries if e.source_meta["sub_dataset"] == "Dataset_Plane"]

    assert len(entries) == 14
    assert len(pose_entries) == 6
    assert len(plane_entries) == 8


def test_fpus23_schema(fpus23_root: Path):
    entries = list(FPUS23Adapter(fpus23_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "FPUS23"
        assert e.anatomy_family == "fetal_planes"
        assert e.modality_type == "image"
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)
        assert len(e.image_paths) == 1


def test_fpus23_pose_detection_entries(fpus23_root: Path):
    entries = {
        (e.source_meta.get("stream_name"), e.source_meta.get("frame_name")): e
        for e in FPUS23Adapter(fpus23_root).iter_entries()
        if e.source_meta["sub_dataset"] == "Dataset"
    }

    e = entries[("stream_hdvb_aroundabd_h", "frame_000000.png")]
    assert e.task_type == "detection"
    assert e.has_box is True
    assert e.is_promptable is True
    assert e.study_id == "stream_hdvb_aroundabd_h"
    assert e.series_id == "stream_hdvb_aroundabd_h"
    assert e.frame_indices == [0]

    orientation = [inst for inst in e.instances if inst.label_ontology == "fetal_pose_orientation"]
    boxes = [inst for inst in e.instances if inst.bbox_xyxy is not None]
    assert orientation[0].label_raw == "hdvb"
    assert orientation[0].classification_label == 0
    assert {box.label_raw for box in boxes} == {"abdomen", "arm"}
    assert boxes[0].bbox_xyxy == pytest.approx([1.0, 1.0, 5.0, 6.0])
    assert e.source_meta["probe_orientation"] == "h"
    assert e.source_meta["view_fetus"] == "abdomen"


def test_fpus23_pose_tag_only_entries(fpus23_root: Path):
    entries = {
        (e.source_meta.get("stream_name"), e.source_meta.get("frame_name")): e
        for e in FPUS23Adapter(fpus23_root).iter_entries()
        if e.source_meta["sub_dataset"] == "Dataset"
    }

    e = entries[("stream_huvb_aroundhead_v", "frame_000001.png")]
    assert e.task_type == "classification"
    assert e.has_box is False
    assert e.is_promptable is False
    assert len(e.instances) == 1
    assert e.instances[0].label_raw == "huvb"
    assert e.instances[0].classification_label == 1
    assert e.source_meta["probe_orientation"] == "v"


def test_fpus23_stream_level_splitting(fpus23_root: Path):
    entries = [
        e for e in FPUS23Adapter(fpus23_root).iter_entries()
        if e.source_meta["sub_dataset"] == "Dataset"
    ]
    by_stream = {}
    for e in entries:
        by_stream.setdefault(e.study_id, set()).add(e.split)

    assert by_stream
    assert all(len(splits) == 1 for splits in by_stream.values())


def test_fpus23_plane_classification_entries(fpus23_root: Path):
    plane_entries = [
        e for e in FPUS23Adapter(fpus23_root).iter_entries()
        if e.source_meta["sub_dataset"] == "Dataset_Plane"
    ]

    assert {e.source_meta["plane_class"] for e in plane_entries} == {
        "AC_PLANE", "BPD_PLANE", "FL_PLANE", "NO_PLANE"
    }
    assert {e.instances[0].classification_label for e in plane_entries} == {0, 1, 2, 3}
    assert all(e.task_type == "classification" for e in plane_entries)
    assert all(e.has_box is False for e in plane_entries)


def test_fpus23_resolve_root_direct_archive(fpus23_root: Path):
    entries = list(FPUS23Adapter(fpus23_root / "archive").iter_entries())
    assert len(entries) == 14


def test_fpus23_split_override(fpus23_root: Path):
    entries = list(FPUS23Adapter(fpus23_root, split_override="val").iter_entries())
    assert all(e.split == "val" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# FOCUS
# ══════════════════════════════════════════════════════════════════════════════

def test_focus_yields_all_split_images(focus_root: Path):
    entries = list(FOCUSAdapter(focus_root).iter_entries())
    assert len(entries) == 4
    assert {e.split for e in entries} == {"train", "val", "test"}


def test_focus_schema(focus_root: Path):
    entries = list(FOCUSAdapter(focus_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "FOCUS"
        assert e.anatomy_family == "fetal_cardiac"
        assert e.modality_type == "image"
        assert e.view_type == "fetal_cardiothoracic"
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)


def test_focus_multi_channel_mask_instances(focus_root: Path):
    entries = {e.series_id: e for e in FOCUSAdapter(focus_root).iter_entries()}
    e = entries["001"]

    assert e.has_mask is True
    assert e.task_type == "segmentation"
    assert e.is_promptable is True
    assert len(e.instances) == 2
    assert {inst.label_raw for inst in e.instances} == {"cardiac", "thorax"}
    assert {inst.label_ontology for inst in e.instances} == {"fetal_cardiac", "fetal_thorax"}
    assert {inst.mask_channel for inst in e.instances} == {0, 1}

    mask_paths = {inst.mask_path for inst in e.instances}
    assert len(mask_paths) == 1
    mask_path = Path(mask_paths.pop())
    assert mask_path.exists()
    assert mask_path.suffix == ".npy"

    import numpy as np
    stack = np.load(mask_path)
    assert stack.shape == (2, 8, 8)
    assert stack[0].sum() > 0
    assert stack[1].sum() > 0


def test_focus_missing_class_mask_is_zero_channel(focus_root: Path):
    entries = {e.series_id: e for e in FOCUSAdapter(focus_root).iter_entries()}
    e = entries["002"]

    assert e.has_mask is True
    assert e.source_meta["missing_mask_classes"] == ["thorax"]

    import numpy as np
    stack = np.load(e.instances[0].mask_path)
    assert stack.shape == (2, 8, 8)
    assert stack[0].sum() > 0
    assert stack[1].sum() == 0


def test_focus_rectangle_annotations_are_axis_aligned_boxes(focus_root: Path):
    entries = {e.series_id: e for e in FOCUSAdapter(focus_root).iter_entries()}
    e = entries["001"]

    cardiac = next(inst for inst in e.instances if inst.label_raw == "cardiac")
    thorax = next(inst for inst in e.instances if inst.label_raw == "thorax")
    assert cardiac.bbox_xyxy == pytest.approx([1.0, 1.0, 4.0, 5.0])
    assert thorax.bbox_xyxy == pytest.approx([3.0, 3.0, 7.0, 7.0])
    assert e.has_box is True
    assert e.source_meta["rectangles"][0]["points"] == [
        [1.0, 1.0], [4.0, 1.0], [4.0, 5.0], [1.0, 5.0]
    ]


def test_focus_ellipse_annotations_in_source_meta(focus_root: Path):
    entries = {e.series_id: e for e in FOCUSAdapter(focus_root).iter_entries()}
    ellipses = entries["001"].source_meta["ellipses"]

    assert len(ellipses) == 2
    assert ellipses[0]["label"] == "cardiac"
    assert ellipses[0]["center_x"] == pytest.approx(4.0)
    assert ellipses[0]["axis_a"] == pytest.approx(2.0)


def test_focus_grayscale_flag(focus_root: Path):
    entries = {e.series_id: e for e in FOCUSAdapter(focus_root).iter_entries()}
    assert entries["001"].source_meta["is_grayscale"] is True
    assert entries["004"].source_meta["is_grayscale"] is False


def test_focus_no_mask_entry_keeps_geometry(focus_root: Path):
    entries = {e.series_id: e for e in FOCUSAdapter(focus_root).iter_entries()}
    e = entries["004"]

    assert e.has_mask is False
    assert e.has_box is True
    assert e.task_type == "detection"
    assert e.is_promptable is True
    assert len(e.instances) == 0
    assert e.source_meta["missing_mask_classes"] == ["cardiac", "thorax"]


def test_focus_split_override(focus_root: Path):
    entries = list(FOCUSAdapter(focus_root, split_override="test").iter_entries())
    assert all(e.split == "test" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# PSFHS
# ══════════════════════════════════════════════════════════════════════════════

def test_psfhs_yields_all_images(psfhs_root: Path):
    entries = list(PSFHSAdapter(psfhs_root).iter_entries())
    assert len(entries) == 3


def test_psfhs_schema(psfhs_root: Path):
    entries = list(PSFHSAdapter(psfhs_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "PSFHS"
        assert e.anatomy_family == "intrapartum"
        assert e.modality_type == "image"
        assert e.view_type == "intrapartum_transperineal"
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)


def test_psfhs_segmentation_instances(psfhs_root: Path):
    entries = {e.series_id: e for e in PSFHSAdapter(psfhs_root).iter_entries()}

    for stem in ("00001", "00002"):
        e = entries[stem]
        assert e.has_mask is True
        assert e.task_type == "segmentation"
        assert e.is_promptable is True
        assert len(e.instances) == 2
        assert {inst.label_ontology for inst in e.instances} == {"pubic_symphysis", "fetal_head"}
        assert {inst.mask_channel for inst in e.instances} == {1, 2}
        assert len({inst.mask_path for inst in e.instances}) == 1
        assert e.instances[0].mask_path.endswith(".mha")


def test_psfhs_ssl_only_missing_label(psfhs_root: Path):
    entries = {e.series_id: e for e in PSFHSAdapter(psfhs_root).iter_entries()}
    e = entries["00003"]

    assert e.has_mask is False
    assert e.task_type == "ssl_only"
    assert e.instances == []


def test_psfhs_mha_header_metadata(psfhs_root: Path):
    entries = {e.series_id: e for e in PSFHSAdapter(psfhs_root).iter_entries()}
    e = entries["00001"]

    assert e.source_meta["image_dim_size"] == "8 8"
    assert e.source_meta["image_channels"] == "3"
    assert e.source_meta["image_spacing"] == "1 1"
    assert e.source_meta["image_origin"] == "0 0"
    assert e.source_meta["label_dim_size"] == "8 8"
    assert e.source_meta["label_element_type"] == "MET_UCHAR"


def test_psfhs_resolve_root_direct(psfhs_root: Path):
    entries = list(PSFHSAdapter(psfhs_root / "PSFHS").iter_entries())
    assert len(entries) == 3


def test_psfhs_split_override(psfhs_root: Path):
    entries = list(PSFHSAdapter(psfhs_root, split_override="val").iter_entries())
    assert all(e.split == "val" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# JNU-IFM
# ══════════════════════════════════════════════════════════════════════════════

def test_jnu_ifm_uses_csv_frame_list(jnu_ifm_root: Path):
    entries = list(JNUIFMAdapter(jnu_ifm_root).iter_entries())
    assert len(entries) == 5
    assert "20190830T115515_999" not in {e.series_id for e in entries}


def test_jnu_ifm_schema(jnu_ifm_root: Path):
    entries = list(JNUIFMAdapter(jnu_ifm_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "JNU-IFM"
        assert e.anatomy_family == "intrapartum"
        assert e.modality_type == "image"
        assert e.view_type == "intrapartum_transperineal"
        assert e.has_mask is True
        assert e.task_type == "segmentation"
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)


def test_jnu_ifm_video_level_splitting(jnu_ifm_root: Path):
    entries = list(JNUIFMAdapter(jnu_ifm_root).iter_entries())
    by_video = {}
    for e in entries:
        by_video.setdefault(e.study_id, set()).add(e.split)

    assert by_video
    assert all(len(splits) == 1 for splits in by_video.values())


def test_jnu_ifm_frame_label_mapping(jnu_ifm_root: Path):
    entries = {e.series_id: e for e in JNUIFMAdapter(jnu_ifm_root).iter_entries()}

    expected = {
        "20190830T115515_169": ("none", 0),
        "20190830T115515_170": ("only_sp", 1),
        "20190918T123342_10": ("only_head", 2),
        "20190830T115515_171": ("sp_head", 3),
    }
    for series_id, (label_raw, label_idx) in expected.items():
        cls_inst = next(inst for inst in entries[series_id].instances
                        if inst.label_ontology == "jnu_ifm_frame_visibility")
        assert cls_inst.label_raw == label_raw
        assert cls_inst.classification_label == label_idx


def test_jnu_ifm_segmentation_instances(jnu_ifm_root: Path):
    entries = {e.series_id: e for e in JNUIFMAdapter(jnu_ifm_root).iter_entries()}
    e = entries["20190830T115515_171"]

    seg_instances = [
        inst for inst in e.instances
        if inst.label_ontology in {"pubic_symphysis", "fetal_head"}
    ]
    assert len(seg_instances) == 2
    assert {inst.mask_channel for inst in seg_instances} == {1, 2}
    assert all(inst.mask_path.endswith(".npy") for inst in seg_instances)
    assert all(inst.is_promptable is True for inst in seg_instances)


def test_jnu_ifm_promptability_tracks_frame_label(jnu_ifm_root: Path):
    entries = {e.series_id: e for e in JNUIFMAdapter(jnu_ifm_root).iter_entries()}

    none_entry = entries["20190830T115515_169"]
    only_sp_entry = entries["20190830T115515_170"]
    only_head_entry = entries["20190918T123342_10"]

    def flags(entry):
        return {
            inst.label_ontology: inst.is_promptable
            for inst in entry.instances
            if inst.label_ontology in {"pubic_symphysis", "fetal_head"}
        }

    assert flags(none_entry) == {"pubic_symphysis": False, "fetal_head": False}
    assert flags(only_sp_entry) == {"pubic_symphysis": True, "fetal_head": False}
    assert flags(only_head_entry) == {"pubic_symphysis": False, "fetal_head": True}


def test_jnu_ifm_remaps_raw_mask_values(jnu_ifm_root: Path):
    import numpy as np

    entries = {e.series_id: e for e in JNUIFMAdapter(jnu_ifm_root).iter_entries()}
    e = entries["20190830T115515_171"]
    mask_path = next(inst.mask_path for inst in e.instances if inst.mask_path)
    mapped = np.load(mask_path)

    assert set(np.unique(mapped).tolist()) == {0, 1, 2}
    assert mapped[1:4, 1:4].max() == 1
    assert mapped[4:7, 4:7].max() == 2
    assert e.source_meta["mask_value_mapping"] == {"7": 1, "8": 2}


def test_jnu_ifm_source_meta(jnu_ifm_root: Path):
    entries = {e.series_id: e for e in JNUIFMAdapter(jnu_ifm_root).iter_entries()}
    e = entries["20190830T115515_170"]

    assert e.study_id == "20190830T115515"
    assert e.frame_indices == [170]
    assert e.source_meta["video_id"] == "20190830T115515"
    assert e.source_meta["frame_id"] == 170
    assert e.source_meta["frame_label_raw"] == 4
    assert e.source_meta["frame_label"] == "only_sp"
    assert e.source_meta["frame_label_index"] == 1
    assert e.source_meta["mask_enhance_ignored"] is True
    assert e.source_meta["is_grayscale"] is True


def test_jnu_ifm_resolve_us_data_direct(jnu_ifm_root: Path):
    entries = list(JNUIFMAdapter(jnu_ifm_root / "us_data").iter_entries())
    assert len(entries) == 5


def test_jnu_ifm_split_override(jnu_ifm_root: Path):
    entries = list(JNUIFMAdapter(jnu_ifm_root, split_override="val").iter_entries())
    assert all(e.split == "val" for e in entries)


# ══════════════════════════════════════════════════════════════════════════════
# IUGC2024
# ══════════════════════════════════════════════════════════════════════════════

def test_iugc2024_yields_one_entry_per_video(iugc2024_root: Path):
    entries = list(IUGC2024Adapter(iugc2024_root).iter_entries())
    assert len(entries) == 3
    assert {e.split for e in entries} == {"train", "val", "test"}


def test_iugc2024_schema(iugc2024_root: Path):
    entries = list(IUGC2024Adapter(iugc2024_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "IUGC2024"
        assert e.anatomy_family == "intrapartum"
        assert e.modality_type == "video"
        assert e.view_type == "intrapartum_transperineal"
        assert e.has_temporal_order is True
        assert e.is_cine is True
        assert e.ssl_stream == "both"
        assert e.curriculum_tier in (1, 2, 3)


def test_iugc2024_train_mask_paths_and_instances(iugc2024_root: Path):
    entries = {e.series_id: e for e in IUGC2024Adapter(iugc2024_root).iter_entries()}
    e = entries["trainvid"]

    assert e.split == "train"
    assert e.num_frames == 80
    assert e.frame_indices == [0, 9]
    assert e.has_mask is True
    assert e.task_type == "segmentation"
    assert len(e.instances) == 4  # 2 structures x 2 labelled frames
    assert {inst.mask_channel for inst in e.instances} == {1, 2}
    assert all("/train/seg/trainvid/mask/" in inst.mask_path for inst in e.instances)
    assert e.source_meta["mask_names"] == ["trainvid_0_6.png", "trainvid_9_6.png"]


def test_iugc2024_val_and_test_mask_naming(iugc2024_root: Path):
    entries = {e.series_id: e for e in IUGC2024Adapter(iugc2024_root).iter_entries()}

    val = entries["valvid"]
    test = entries["test_10_80"]
    assert val.frame_indices == [6]
    assert test.frame_indices == [80]
    assert val.source_meta["mask_names"] == ["valvid_6.png"]
    assert test.source_meta["mask_names"] == ["test_10_80.png"]
    assert val.source_meta["mask_paths"][0].endswith("/val/seg/valvid_6.png")
    assert test.source_meta["mask_paths"][0].endswith("/test/seg/test_10_80.png")


def test_iugc2024_classification_indices(iugc2024_root: Path):
    entries = {e.series_id: e for e in IUGC2024Adapter(iugc2024_root).iter_entries()}

    assert entries["trainvid"].source_meta["pos_indices"] == [0, 9]
    assert entries["trainvid"].source_meta["neg_indices"] == "NONE"
    assert entries["valvid"].source_meta["pos_indices"] == [6, 7]
    assert entries["valvid"].source_meta["neg_indices"] == [0]
    assert entries["test_10_80"].source_meta["pos_indices"] == list(range(81))
    assert entries["test_10_80"].source_meta["neg_indices"] == "NONE"


def test_iugc2024_landmarks_convert_yx_to_xy(iugc2024_root: Path):
    entries = {e.series_id: e for e in IUGC2024Adapter(iugc2024_root).iter_entries()}
    e = entries["trainvid"]

    ps = next(inst for inst in e.instances
              if inst.instance_id == "trainvid_0_pubic_symphysis")
    head = next(inst for inst in e.instances
                if inst.instance_id == "trainvid_0_fetal_head")

    assert ps.keypoints == [[198.0, 69.0], [299.0, 84.0]]
    assert ps.measurement_mm == pytest.approx(122.4)
    assert head.keypoints == [[294.0, 147.0], [339.0, 174.0]]
    assert head.measurement_mm == pytest.approx(63.2)


def test_iugc2024_split_level_info_metadata(iugc2024_root: Path):
    entries = {e.series_id: e for e in IUGC2024Adapter(iugc2024_root).iter_entries()}

    train = entries["trainvid"]
    val = entries["valvid"]
    assert train.source_meta["standard_plane"] is True
    assert val.source_meta["sp_count"] == 2
    assert val.source_meta["nsp_count"] == 1
    assert val.source_meta["sp_indices"] == [6, 7]
    assert val.source_meta["nsp_indices"] == [0]


def test_iugc2024_resolve_new_root_direct(iugc2024_root: Path):
    entries = list(IUGC2024Adapter(iugc2024_root / "new").iter_entries())
    assert len(entries) == 3


def test_iugc2024_split_override(iugc2024_root: Path):
    entries = list(IUGC2024Adapter(iugc2024_root, split_override="test").iter_entries())
    assert all(e.split == "test" for e in entries)
