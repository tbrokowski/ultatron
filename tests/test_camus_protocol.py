"""Tests for CAMUS official protocol helpers."""
from __future__ import annotations

import pytest

from finetune.datasets.camus_io import (
    OFFICIAL_CAMUS_SPLIT,
    aggregate_camus_metrics,
    encode_lv_mask,
    fold_for_patient,
    get_camus_variant_spec,
    official_split_for_patient,
    quality_is_acceptable,
)


@pytest.mark.parametrize(
    "patient_id,expected_fold",
    [
        (1, 1),
        (50, 1),
        (51, 2),
        (400, 8),
        (401, 9),
        (450, 9),
        (451, 10),
        (500, 10),
    ],
)
def test_fold_for_patient(patient_id, expected_fold):
    assert fold_for_patient(patient_id) == expected_fold


@pytest.mark.parametrize(
    "patient_id,expected_split",
    [
        (1, "train"),
        (400, "train"),
        (401, "val"),
        (450, "val"),
        (451, "test"),
        (500, "test"),
    ],
)
def test_official_split_matches_fold_blocks(patient_id, expected_split):
    assert official_split_for_patient(patient_id) == expected_split


def test_official_split_ranges_consistent_with_folds():
    for split, (lo, hi) in OFFICIAL_CAMUS_SPLIT.items():
        for pid in (lo, hi, (lo + hi) // 2):
            assert official_split_for_patient(pid) == split


def test_encode_lv_masks_match_tmi_structures():
    labels = __import__("numpy").array(
        [[0, 1, 2], [3, 1, 0]], dtype=__import__("numpy").int64,
    )
    endo = encode_lv_mask(labels, "lv_cavity")
    epi = encode_lv_mask(labels, "lv_structures")
    la = encode_lv_mask(labels, "lv_la")
    assert endo.tolist() == [[0, 1, 0], [0, 1, 0]]
    assert epi.tolist() == [[0, 1, 1], [0, 1, 0]]
    assert la.tolist() == [[0, 0, 0], [1, 0, 0]]


@pytest.mark.parametrize(
    "quality,ok",
    [
        ("Good", True),
        ("Medium", True),
        ("Poor", False),
        (None, True),
    ],
)
def test_quality_filter(quality, ok):
    assert quality_is_acceptable(quality) is ok


def test_aggregate_camus_binary_lv_epi_only_emits_epi_keys():
    per_sample = [
        {"view": "2CH", "phase": "ED", "dice_lv_epi": 0.9, "iou": 0.8, "hd95": 1.0},
        {"view": "4CH", "phase": "ES", "dice_lv_epi": 0.8, "iou": 0.7, "hd95": 2.0},
    ]
    out = aggregate_camus_metrics(per_sample, "lv_epi")
    assert "dice_lv_epi_mean" in out
    assert "dice_lv_endo_mean" not in out
    assert out["dice_mean"] == out["dice_lv_epi_mean"]


def test_aggregate_camus_multiclass_emits_class_keys():
    per_sample = [
        {
            "view": "2CH", "phase": "ED",
            "dice_class_1": 0.9, "dice_class_2": 0.8, "dice_class_3": 0.7,
            "iou": 0.8, "hd95": 1.0,
        },
    ]
    out = aggregate_camus_metrics(per_sample, "multiclass")
    assert out["dice_class_1_mean"] == 0.9
    assert out["dice_macro_fg_mean"] == pytest.approx(0.8, rel=1e-3)


def test_get_camus_variant_spec_lv_la():
    spec = get_camus_variant_spec("la")
    assert spec["lv_target"] == "lv_la"
    assert spec["dice_key"] == "dice_la"
