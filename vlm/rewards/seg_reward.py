"""
vlm/rewards/seg_reward.py  ·  SegmentationReward
==================================================

Dice / IoU reward for segmentation tasks.

When ground-truth masks are available in the manifest (instance.mask_path),
this reward computes the Dice coefficient between the SAM2 mask(s) produced
during the trajectory and the GT mask.

For non-segmentation tasks this reward returns 0.0 gracefully.

References
----------
  SAM-R1 (arxiv 2505.22596) — SAM IoU/confidence as RL reward
  Dr.Seg (arxiv 2603.00152) — GRPO for segmentation with distribution-ranked reward
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vlm.rewards.base import RewardFunction, RewardOutput

log = logging.getLogger(__name__)


def _load_mask(mask_path: str) -> Optional[np.ndarray]:
    """Load a binary mask from a PNG path."""
    try:
        from PIL import Image
        img = Image.open(mask_path).convert("L")
        return np.array(img) > 128
    except Exception as e:
        log.warning(f"Could not load mask {mask_path}: {e}")
        return None


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient between two binary arrays."""
    pred_bool = pred.astype(bool)
    gt_bool   = gt.astype(bool)
    intersection = (pred_bool & gt_bool).sum()
    denom = pred_bool.sum() + gt_bool.sum()
    if denom == 0:
        return 1.0 if pred_bool.sum() == 0 else 0.0
    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two binary arrays."""
    pred_bool = pred.astype(bool)
    gt_bool   = gt.astype(bool)
    intersection = (pred_bool & gt_bool).sum()
    union = (pred_bool | gt_bool).sum()
    return float(intersection / union) if union > 0 else (1.0 if pred_bool.sum() == 0 else 0.0)


def _resize_mask(mask: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize mask to target (H, W) using nearest-neighbour."""
    if mask.shape == target_shape:
        return mask
    try:
        from PIL import Image
        m_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        m_pil = m_pil.resize((target_shape[1], target_shape[0]), Image.NEAREST)
        return np.array(m_pil) > 128
    except Exception:
        return mask


class SegmentationReward(RewardFunction):
    """
    Dice-based segmentation reward using SAM2 masks from the trajectory.

    Parameters
    ----------
    weight        : contribution weight in composite reward
    use_iou       : if True, average Dice and IoU; else Dice only
    min_area_frac : minimum mask area fraction to count as a valid prediction
                    (avoids rewarding empty masks on non-segmentation tasks)
    """

    def __init__(
        self,
        weight:        float = 0.3,
        use_iou:       bool  = True,
        min_area_frac: float = 0.001,
    ):
        super().__init__(weight)
        self.use_iou       = use_iou
        self.min_area_frac = min_area_frac

    def compute(
        self,
        trajectory:   Dict[str, Any],
        ground_truth: Any,
        task_type:    str,
        dataset_id:   Optional[str] = None,
        image:        Optional[Any] = None,
    ) -> RewardOutput:
        """
        ground_truth for segmentation: a mask_path str, np.ndarray, or
        a dict {"mask_path": str} / {"mask": np.ndarray}.
        """
        # Only active for segmentation tasks
        if task_type not in ("segmentation", "weak_label") or ground_truth is None:
            return RewardOutput(score=0.0, breakdown={"seg_reward_active": 0.0})

        gt_mask = self._resolve_gt_mask(ground_truth)
        if gt_mask is None:
            return RewardOutput(score=0.0, breakdown={"gt_mask_available": 0.0})

        # Collect all SAM2 masks from trajectory tool calls
        tool_calls: List[dict] = trajectory.get("tool_calls", [])
        pred_masks = [
            tc["raw_mask"] for tc in tool_calls
            if isinstance(tc.get("raw_mask"), np.ndarray)
        ]

        if not pred_masks:
            return RewardOutput(score=0.0, breakdown={"gt_mask_available": 1.0, "n_masks": 0})

        # Use the best mask (highest Dice with GT)
        best_dice = 0.0
        best_iou  = 0.0
        for pred_mask in pred_masks:
            area_frac = float(pred_mask.sum()) / max(pred_mask.size, 1)
            if area_frac < self.min_area_frac:
                continue
            pred_resized = _resize_mask(pred_mask, gt_mask.shape)
            d = dice_score(pred_resized, gt_mask)
            u = iou_score(pred_resized, gt_mask)
            if d > best_dice:
                best_dice = d
                best_iou  = u

        score = (best_dice + best_iou) / 2.0 if self.use_iou else best_dice

        return RewardOutput(
            score=score,
            breakdown={
                "gt_mask_available": 1.0,
                "n_masks":           len(pred_masks),
                "best_dice":         best_dice,
                "best_iou":          best_iou,
            },
        )

    @staticmethod
    def _resolve_gt_mask(ground_truth: Any) -> Optional[np.ndarray]:
        """Resolve various GT mask formats to a numpy bool array."""
        if isinstance(ground_truth, np.ndarray):
            return ground_truth.astype(bool)
        if isinstance(ground_truth, str):
            return _load_mask(ground_truth)
        if isinstance(ground_truth, dict):
            if "mask" in ground_truth and isinstance(ground_truth["mask"], np.ndarray):
                return ground_truth["mask"].astype(bool)
            if "mask_path" in ground_truth:
                return _load_mask(ground_truth["mask_path"])
        return None
