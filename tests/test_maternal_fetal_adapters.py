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
from data.adapters.maternal_fetal.fh_ps_aop                  import FHPSAOPAdapter
from data.adapters.maternal_fetal.hc18                       import HC18Adapter


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
