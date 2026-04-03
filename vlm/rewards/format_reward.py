"""
vlm/rewards/format_reward.py  ·  FormatReward
===============================================

Checks whether the model's output follows the required structured format.

Expected output structure (per task type)
------------------------------------------
  All tasks:
    <think>…</think><answer>JSON or plain text</answer>

  regression:
    <answer>{"value": <float>}</answer>

  classification:
    <answer>{"label": "<str>"}</answer>
    or
    <answer>{"label": <int>}</answer>

  segmentation:
    <answer>{"label": "<str>", "bbox": [x1,y1,x2,y2]}</answer>

  open_ended / weak_label:
    <answer>{"findings": "<str>", "impression": "<str>"}</answer>

Scoring
-------
  1.0  — all required tags present and answer is valid JSON of the right shape
  0.5  — think/answer tags present but JSON malformed or wrong keys
  0.0  — no answer tag found
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from vlm.rewards.base import RewardFunction, RewardOutput

log = logging.getLogger(__name__)

_ANSWER_RE  = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE   = re.compile(r"<think>(.*?)</think>",   re.DOTALL)

# Required JSON keys per task type (any of the sets is acceptable)
_REQUIRED_KEYS: Dict[str, list] = {
    "regression":      [["value"], ["ef", "ejection_fraction"], ["measurement"]],
    "classification":  [["label"], ["view"], ["diagnosis"]],
    "segmentation":    [["label"], ["bbox"], ["findings"]],
    "weak_label":      [["findings"], ["impression"], ["label"], ["value"]],
    "ssl_only":        [[]],   # no specific key required
    "measurement":     [["value"], ["measurement"]],
    "sequence":        [["label"]],
    "detection":       [["bbox"], ["label"]],
}


class FormatReward(RewardFunction):
    """
    Structural format compliance reward.

    weight : contribution to composite reward (plan default: w_fmt in config)
    """

    def __init__(self, weight: float = 0.1):
        super().__init__(weight)

    def compute(
        self,
        trajectory:   Dict[str, Any],
        ground_truth: Any,
        task_type:    str,
        dataset_id:   Optional[str] = None,
        image:        Optional[Any] = None,
    ) -> RewardOutput:
        text = trajectory.get("text", "") or trajectory.get("answer", "")

        has_think  = bool(_THINK_RE.search(text))
        ans_match  = _ANSWER_RE.search(text)
        has_answer = bool(ans_match)

        if not has_answer:
            return RewardOutput(
                score=0.0,
                breakdown={"has_think": float(has_think), "has_answer": 0.0, "valid_json": 0.0},
            )

        ans_text = ans_match.group(1).strip()
        parsed, valid_json = self._try_parse_json(ans_text)

        if not valid_json:
            return RewardOutput(
                score=0.5,
                breakdown={"has_think": float(has_think), "has_answer": 1.0, "valid_json": 0.0},
            )

        # Check required keys
        key_ok = self._check_keys(parsed, task_type)
        score  = 1.0 if key_ok else 0.75

        return RewardOutput(
            score=score,
            breakdown={
                "has_think":  float(has_think),
                "has_answer": 1.0,
                "valid_json": 1.0,
                "key_ok":     float(key_ok),
            },
        )

    @staticmethod
    def _try_parse_json(text: str):
        """Try to parse as JSON; return (parsed, success)."""
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            return None, False

    @staticmethod
    def _check_keys(parsed: Any, task_type: str) -> bool:
        """Check that at least one of the required key sets is present."""
        if not isinstance(parsed, dict):
            return False
        required_sets = _REQUIRED_KEYS.get(task_type, [[]])
        parsed_keys   = set(parsed.keys())
        for key_set in required_sets:
            if not key_set or parsed_keys.intersection(key_set):
                return True
        return False
