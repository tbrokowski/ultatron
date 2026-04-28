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

from data.adapters.maternal_fetal.acouslic import ACOUSLICAIAdapter


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
