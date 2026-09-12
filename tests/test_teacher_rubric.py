"""Tests for the frozen-teacher rubric reward (WP5 R3/R4)."""
from __future__ import annotations

from vlm.rewards.teacher_rubric import (
    DEFAULT_WEIGHTS,
    TeacherRubricReward,
    combine_reward,
    exact_match_organ_diagnosis,
    parse_teacher_scores,
)


def test_parse_teacher_json():
    scores, ok = parse_teacher_scores(
        '{"organ": 1, "diagnosis": 0.5, "attribute": 0.2, "hallucinated": 0.0, "format": 1}'
    )
    assert ok
    assert scores["organ"] == 1.0
    assert scores["diagnosis"] == 0.5


def test_parse_failure():
    scores, ok = parse_teacher_scores("not json at all")
    assert ok is False
    assert scores is None


def test_exact_match_only_when_attributes_exist():
    em = exact_match_organ_diagnosis(
        {"organ": "kidney", "diagnosis": "cyst"},
        {"organ": "kidney", "diagnosis": "cyst"},
    )
    assert em["applied"] == 1.0
    assert em["organ_em"] == 1.0
    assert em["diagnosis_em"] == 1.0
    empty = exact_match_organ_diagnosis({"organ": "x"}, {})
    assert empty["applied"] == 0.0


def test_weighted_sum_defaults():
    scores = {k: 1.0 for k in DEFAULT_WEIGHTS}
    # hallucinated weight is negative, so perfect 1.0 there *decreases* reward
    out = combine_reward(scores, {"applied": 0.0}, parse_ok=True)
    expected = sum(DEFAULT_WEIGHTS.values())  # 0.25+0.35+0.25-0.15+0.15 = 0.85
    assert abs(out["reward"] - expected) < 1e-9


def test_reward_function_end_to_end():
    fn = TeacherRubricReward()
    out = fn.compute(
        trajectory={
            "teacher_text": '{"organ":1,"diagnosis":1,"attribute":1,"hallucinated":0,"format":1}',
            "text": '{"organ":"kidney","diagnosis":"cyst"}',
        },
        ground_truth={"organ": "kidney", "diagnosis": "cyst"},
        task_type="weak_label",
    )
    assert out.score > 1.0  # teacher weighted + exact-match
    assert out.meta["parse_ok"] is True
