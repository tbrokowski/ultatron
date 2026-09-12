"""
vlm/rewards/teacher_rubric.py  ·  Frozen VLM teacher reward (WP5 R3/R4)
=======================================================================

The teacher sees the image, caption/attributes, the stored reference answer
and the student answer.  It returns 0–1 scores for:

  organ, diagnosis, attribute correctness, hallucinated findings, format

Reward = weighted sum (defaults 0.25 / 0.35 / 0.25 / −0.15 / 0.15) plus an
in-process exact-match term on organ and diagnosis when attributes exist.

This module is the in-process half of the NeMo-Gym resource server: it parses
the teacher's JSON (or applies exact-match when no teacher JSON is present).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from vlm.rewards.base import RewardFunction, RewardOutput


DEFAULT_WEIGHTS = {
    "organ": 0.25,
    "diagnosis": 0.35,
    "attribute": 0.25,
    "hallucinated": -0.15,
    "format": 0.15,
}

TEACHER_SCHEMA = {
    "organ": "0-1 correctness of named organ / body system",
    "diagnosis": "0-1 correctness of diagnosis or primary finding",
    "attribute": "0-1 correctness of remaining structured attributes",
    "hallucinated": "0-1 presence of findings not supported by image or metadata",
    "format": "0-1 valid JSON / required keys",
}

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _norm(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _parse_json_blob(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None


def parse_teacher_scores(text: str) -> Tuple[Optional[dict], bool]:
    """Return (scores dict, parse_ok). Missing keys default to 0."""
    obj = _parse_json_blob(text)
    if obj is None:
        return None, False
    scores = {}
    for k in DEFAULT_WEIGHTS:
        try:
            scores[k] = float(obj.get(k, 0.0))
        except (TypeError, ValueError):
            scores[k] = 0.0
        scores[k] = max(0.0, min(1.0, scores[k]))
    return scores, True


def exact_match_organ_diagnosis(
    student: dict,
    reference: dict,
) -> Dict[str, float]:
    """In-process exact-match term when structured attributes exist."""
    out = {"organ_em": 0.0, "diagnosis_em": 0.0, "applied": 0.0}
    organ_ref = _norm(reference.get("organ") or reference.get("body_system"))
    diag_ref = _norm(reference.get("diagnosis") or reference.get("finding"))
    if not organ_ref and not diag_ref:
        return out
    out["applied"] = 1.0
    organ_st = _norm(student.get("organ") or student.get("body_system"))
    diag_st = _norm(student.get("diagnosis") or student.get("finding"))
    if organ_ref:
        out["organ_em"] = 1.0 if organ_st == organ_ref else 0.0
    if diag_ref:
        out["diagnosis_em"] = 1.0 if diag_st == diag_ref else 0.0
    return out


def combine_reward(
    teacher_scores: Optional[Dict[str, float]],
    exact: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
    parse_ok: bool = True,
) -> Dict[str, Any]:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    scores = dict(teacher_scores or {k: 0.0 for k in DEFAULT_WEIGHTS})
    if not parse_ok:
        scores["format"] = 0.0
    weighted = sum(w[k] * scores.get(k, 0.0) for k in w)
    em = 0.0
    if exact.get("applied"):
        em = 0.5 * (exact.get("organ_em", 0.0) + exact.get("diagnosis_em", 0.0))
    total = weighted + em
    return {
        "reward": total,
        "weighted_teacher": weighted,
        "exact_match": em,
        "scores": scores,
        "exact": exact,
        "parse_ok": parse_ok,
        "weights": w,
    }


class TeacherRubricReward(RewardFunction):
    """
    Combine frozen-teacher JSON scores with optional exact-match.

    ``trajectory`` may contain:
      * teacher_text / teacher_json — raw teacher output
      * text / answer — student answer (JSON or free text)
    ``ground_truth`` is the reference dict (caption, organ, diagnosis, …).
    """

    def __init__(self, weight: float = 1.0, weights: Optional[Dict[str, float]] = None):
        super().__init__(weight)
        self.component_weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.component_weights.update(weights)

    def compute(
        self,
        trajectory: Dict[str, Any],
        ground_truth: Any,
        task_type: str,
        dataset_id: Optional[str] = None,
        image: Optional[Any] = None,
    ) -> RewardOutput:
        teacher_text = (
            trajectory.get("teacher_text")
            or trajectory.get("teacher_json")
            or ""
        )
        if isinstance(teacher_text, dict):
            scores, parse_ok = teacher_text, True
        else:
            scores, parse_ok = parse_teacher_scores(str(teacher_text))
        student_blob = _parse_json_blob(
            trajectory.get("text") or trajectory.get("answer") or ""
        ) or {}
        ref = ground_truth if isinstance(ground_truth, dict) else {}
        exact = exact_match_organ_diagnosis(student_blob, ref)
        combined = combine_reward(
            scores, exact, self.component_weights, parse_ok=parse_ok,
        )
        return RewardOutput(
            score=combined["reward"],
            breakdown={
                "weighted_teacher": combined["weighted_teacher"],
                "exact_match": combined["exact_match"],
                **combined["scores"],
            },
            meta={
                "parse_ok": parse_ok,
                "exact": exact,
                "dataset_id": dataset_id,
                "task_type": task_type,
            },
        )
