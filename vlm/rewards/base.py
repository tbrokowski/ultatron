"""
vlm/rewards/base.py  ·  RewardFunction ABC
===========================================

All reward functions return a RewardOutput so the composite can combine them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RewardOutput:
    """
    Per-trajectory reward breakdown.

    Attributes
    ----------
    score     : scalar reward for this function (before weighting)
    breakdown : optional sub-component scores for logging
    meta      : arbitrary extra info (e.g. judge response text)
    """
    score:     float
    breakdown: Dict[str, float] = field(default_factory=dict)
    meta:      Dict[str, Any]   = field(default_factory=dict)

    def __add__(self, other: "RewardOutput") -> "RewardOutput":
        return RewardOutput(
            score     = self.score + other.score,
            breakdown = {**self.breakdown, **other.breakdown},
            meta      = {**self.meta, **other.meta},
        )


class RewardFunction(ABC):
    """
    Abstract reward function.

    Subclasses implement compute() which maps a trajectory + ground truth to
    a RewardOutput.

    Parameters
    ----------
    weight : multiplicative weight applied before summing into composite reward
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @abstractmethod
    def compute(
        self,
        trajectory:     Dict[str, Any],   # output of StudentModel.generate_with_tools
        ground_truth:   Any,              # task-specific label
        task_type:      str,              # e.g. "regression", "classification", "segmentation"
        dataset_id:     Optional[str] = None,
        image:          Optional[Any] = None,
    ) -> RewardOutput:
        """Compute the reward for a single trajectory."""
        ...

    def weighted(self, out: RewardOutput) -> float:
        """Return weighted scalar reward."""
        return self.weight * out.score

    def __call__(self, *args, **kwargs) -> RewardOutput:
        return self.compute(*args, **kwargs)
