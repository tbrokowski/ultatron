"""
vlm/rewards/composite.py  ·  CompositeReward
=============================================

Combines all reward components into a single scalar per trajectory.

Formula (DeepEyes Eq. 2 + MedGRPO cross-dataset normalization)
---------------------------------------------------------------
  R(τ) = w_acc  * R_acc(τ)
       + w_fmt  * R_format(τ)
       + I[R_acc > 0] * w_tool * R_tool(τ)    # conditional SAM2 bonus
       + w_seg  * R_seg(τ)                     # Dice when GT mask available

Cross-dataset normalization (MedGRPO, arxiv 2512.06581):
  Within a batch, normalize each dataset's rewards by its running median/std
  to prevent reward scale collapse when training on cardiac + lung + breast
  simultaneously.

Conditional tool bonus (DeepEyes, arxiv 2505.14362):
  R_tool is granted ONLY when R_acc > 0 AND the trajectory includes at least
  one SAM2 tool call.  This aligns tool use with correct outcomes.

Parameters (read from configs/vlm/base_vlm.yaml → reward section)
------------------------------------------------------------------
  w_acc   : float  (default 1.0)
  w_fmt   : float  (default 0.1)
  w_tool  : float  (default 0.2)
  w_seg   : float  (default 0.3)
  use_cross_dataset_norm : bool (default true)
  norm_window : int  — rolling window for dataset-level median/std (default 200)
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from vlm.rewards.base import RewardFunction, RewardOutput

log = logging.getLogger(__name__)


@dataclass
class RewardWeights:
    w_acc:   float = 1.0
    w_fmt:   float = 0.1
    w_tool:  float = 0.2
    w_seg:   float = 0.3

    @classmethod
    def from_dict(cls, d: dict) -> "RewardWeights":
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


class CompositeReward:
    """
    Combines accuracy, format, tool, and segmentation rewards.

    Parameters
    ----------
    acc_reward  : MedGeminiReward instance
    fmt_reward  : FormatReward instance
    seg_reward  : SegmentationReward instance
    weights     : RewardWeights
    use_norm    : enable cross-dataset reward normalization (MedGRPO)
    norm_window : rolling window size for per-dataset statistics
    """

    def __init__(
        self,
        acc_reward:  RewardFunction,
        fmt_reward:  RewardFunction,
        seg_reward:  RewardFunction,
        weights:     Optional[RewardWeights] = None,
        use_norm:    bool = True,
        norm_window: int  = 200,
    ):
        self.acc_reward  = acc_reward
        self.fmt_reward  = fmt_reward
        self.seg_reward  = seg_reward
        self.weights     = weights if weights is not None else RewardWeights()
        self.use_norm    = use_norm

        # Per-dataset rolling buffers for normalization
        self._norm_buf: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=norm_window)
        )

    # ── Compute single trajectory ─────────────────────────────────────────────

    def compute(
        self,
        trajectory:   Dict[str, Any],
        ground_truth: Any,
        task_type:    str,
        dataset_id:   Optional[str] = None,
        image:        Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Compute all reward components for a single trajectory.

        Returns a dict with:
          total      : composite scalar (after normalization)
          r_acc      : raw accuracy reward
          r_fmt      : raw format reward
          r_tool     : raw tool bonus (0 if not triggered)
          r_seg      : raw segmentation reward
          normalized : bool — whether cross-dataset norm was applied
        """
        w = self.weights

        # ── Accuracy ─────────────────────────────────────────────────────────
        acc_out = self.acc_reward.compute(trajectory, ground_truth, task_type,
                                          dataset_id=dataset_id, image=image)
        r_acc = acc_out.score

        # ── Format ───────────────────────────────────────────────────────────
        fmt_out = self.fmt_reward.compute(trajectory, ground_truth, task_type)
        r_fmt = fmt_out.score

        # ── Conditional tool bonus ────────────────────────────────────────────
        # Granted iff R_acc > 0 AND at least one SAM2 tool call was made
        tool_calls = trajectory.get("tool_calls", [])
        n_tool     = len([tc for tc in tool_calls if not tc.get("error")])
        r_tool = 1.0 if (r_acc > 0 and n_tool > 0) else 0.0

        # ── Segmentation ─────────────────────────────────────────────────────
        seg_out = self.seg_reward.compute(trajectory, ground_truth, task_type,
                                           dataset_id=dataset_id, image=image)
        r_seg = seg_out.score

        # ── Composite ────────────────────────────────────────────────────────
        raw_total = (
            w.w_acc  * r_acc
            + w.w_fmt  * r_fmt
            + w.w_tool * r_tool
            + w.w_seg  * r_seg
        )

        # ── Cross-dataset normalization ───────────────────────────────────────
        normalized = False
        if self.use_norm and dataset_id:
            raw_total, normalized = self._normalize(raw_total, dataset_id)

        return {
            "total":      raw_total,
            "r_acc":      r_acc,
            "r_fmt":      r_fmt,
            "r_tool":     r_tool,
            "r_seg":      r_seg,
            "n_tool":     n_tool,
            "normalized": normalized,
            "acc_breakdown": acc_out.breakdown,
            "fmt_breakdown": fmt_out.breakdown,
            "seg_breakdown": seg_out.breakdown,
        }

    def _normalize(self, score: float, dataset_id: str) -> tuple:
        """
        Cross-dataset normalization per MedGRPO:
          normalized = (score - median_ds) / (std_ds + eps)

        The buffer is updated AFTER normalization so the current score
        participates in future normalizations.
        """
        buf = self._norm_buf[dataset_id]
        if len(buf) >= 2:
            median = float(np.median(list(buf)))
            std    = float(np.std(list(buf))) + 1e-6
            score_norm = (score - median) / std
        else:
            score_norm = score
        buf.append(score)  # update after use
        return score_norm, True

    # ── Batch compute (for GRPO rollout) ─────────────────────────────────────

    def compute_batch(
        self,
        trajectories: List[Dict[str, Any]],
        ground_truths: List[Any],
        task_types:    List[str],
        dataset_ids:   Optional[List[Optional[str]]] = None,
        images:        Optional[List[Any]] = None,
    ) -> List[Dict[str, float]]:
        """
        Compute rewards for a batch of trajectories (one per rollout).
        """
        n = len(trajectories)
        if dataset_ids is None:
            dataset_ids = [None] * n
        if images is None:
            images = [None] * n

        return [
            self.compute(
                trajectory   = trajectories[i],
                ground_truth = ground_truths[i],
                task_type    = task_types[i],
                dataset_id   = dataset_ids[i],
                image        = images[i],
            )
            for i in range(n)
        ]

    # ── Factory ─────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "CompositeReward":
        """
        Build from the reward section of configs/vlm/base_vlm.yaml.

        Expected structure:
          reward:
            weights:
              w_acc:  1.0
              w_fmt:  0.1
              w_tool: 0.2
              w_seg:  0.3
            use_cross_dataset_norm: true
            norm_window: 200
            medgemini:
              mode: auto
              api_model: gemini-2.0-flash
              local_model: null
            format:
              weight: 0.1
            seg:
              weight: 0.3
              use_iou: true
        """
        from vlm.rewards.medgemini   import MedGeminiReward
        from vlm.rewards.format_reward import FormatReward
        from vlm.rewards.seg_reward  import SegmentationReward

        r_cfg    = cfg.get("reward", cfg)
        weights  = RewardWeights.from_dict(r_cfg.get("weights", {}))

        mg_cfg   = r_cfg.get("medgemini", {})
        fmt_cfg  = r_cfg.get("format",    {})
        seg_cfg  = r_cfg.get("seg",       {})

        acc_rew = MedGeminiReward(weight=weights.w_acc, **mg_cfg)
        fmt_rew = FormatReward(weight=weights.w_fmt, **{k: v for k, v in fmt_cfg.items()
                                                         if k != "weight"})
        seg_rew = SegmentationReward(weight=weights.w_seg, **{k: v for k, v in seg_cfg.items()
                                                               if k != "weight"})

        return cls(
            acc_reward  = acc_rew,
            fmt_reward  = fmt_rew,
            seg_reward  = seg_rew,
            weights     = weights,
            use_norm    = r_cfg.get("use_cross_dataset_norm", True),
            norm_window = r_cfg.get("norm_window", 200),
        )
