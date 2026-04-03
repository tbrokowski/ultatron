"""
eval/metrics.py  ·  Evaluation metric functions
=====================================================

All inputs are numpy
arrays or torch tensors.  Safe to call from any eval script, test, or
the AutoResearch agent loop.

Metrics
-------
Segmentation
    dice_score(pred, target)          Dice / F1 coefficient
    iou_score(pred, target)           Intersection over Union (Jaccard)
    hausdorff_95(pred, target)        95th-percentile Hausdorff distance (mm)

Classification
    auc_roc(y_true, y_score)          Area Under ROC Curve
    average_precision(y_true, y_score)AP (area under PR curve)

Regression / Measurement
    mae(pred, target)                 Mean Absolute Error
    rmse(pred, target)                Root Mean Squared Error
    pearson_r(pred, target)           Pearson correlation coefficient

Cardiac-specific
    ef_mae(pred_ef, true_ef)          EF MAE (EchoNet-Dynamic benchmark)
    ef_r2(pred_ef, true_ef)           R² for ejection fraction

All functions return Python floats unless noted.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional, Union

import numpy as np


# ── Segmentation ──────────────────────────────────────────────────────────────

def dice_score(
    pred: np.ndarray,    # (H, W) or (B, H, W) float or bool, already thresholded
    target: np.ndarray,  # same shape, binary
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Dice coefficient: 2*|P∩T| / (|P|+|T|)

    Accepts continuous predictions (thresholded at `threshold`) or binary masks.
    Averages over batch dimension if inputs are 3D.
    """
    if pred.dtype != bool:
        pred = pred >= threshold
    if target.dtype != bool:
        target = target >= threshold

    if pred.ndim == 3:
        scores = [dice_score(pred[i], target[i], eps=eps) for i in range(pred.shape[0])]
        return float(np.mean(scores))

    intersection = (pred & target).sum()
    union_sum    = pred.sum() + target.sum()
    if union_sum == 0:
        return 1.0   # both empty → perfect match
    return float(2.0 * intersection / (union_sum + eps))


def iou_score(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Intersection over Union (Jaccard index): |P∩T| / |P∪T|
    """
    if pred.dtype != bool:
        pred = pred >= threshold
    if target.dtype != bool:
        target = target >= threshold

    if pred.ndim == 3:
        scores = [iou_score(pred[i], target[i], eps=eps) for i in range(pred.shape[0])]
        return float(np.mean(scores))

    inter = (pred & target).sum()
    union = (pred | target).sum()
    if union == 0:
        return 1.0
    return float(inter / (union + eps))


def hausdorff_95(
    pred: np.ndarray,    # (H, W) binary
    target: np.ndarray,  # (H, W) binary
    spacing_mm: float = 1.0,  # isotropic pixel spacing in mm
) -> float:
    """
    95th-percentile Hausdorff distance in mm.

    Falls back to scipy.ndimage if available; returns NaN otherwise.
    Requires non-empty pred and target.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        warnings.warn("scipy not available — hausdorff_95 returns NaN")
        return float("nan")

    if not pred.any() or not target.any():
        return float("nan")

    pred_b   = pred.astype(bool)
    target_b = target.astype(bool)

    # Distance from each pred point to nearest target point and vice versa
    dt_target = distance_transform_edt(~target_b) * spacing_mm
    dt_pred   = distance_transform_edt(~pred_b)   * spacing_mm

    d_pt = dt_target[pred_b]
    d_tp = dt_pred[target_b]

    all_d = np.concatenate([d_pt, d_tp])
    return float(np.percentile(all_d, 95))


# ── Classification ────────────────────────────────────────────────────────────

def auc_roc(
    y_true: np.ndarray,   # (N,) int or binary
    y_score: np.ndarray,  # (N,) float probabilities, or (N, C) for multiclass
    average: str = "macro",
) -> float:
    """
    Area Under the ROC Curve.
    Multiclass: OvR average (average="macro" or "weighted").
    Returns NaN if only one class present.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        warnings.warn("sklearn not available — auc_roc returns NaN")
        return float("nan")

    if len(np.unique(y_true)) < 2:
        return float("nan")

    try:
        if y_score.ndim == 1:
            return float(roc_auc_score(y_true, y_score))
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average=average))
    except Exception as e:
        warnings.warn(f"auc_roc failed: {e}")
        return float("nan")


def average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Area under Precision-Recall curve."""
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


# ── Regression / Measurement ──────────────────────────────────────────────────

def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.abs(pred - target).mean())


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(math.sqrt(((pred - target) ** 2).mean()))


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation coefficient. Returns NaN if std is 0."""
    if pred.std() < 1e-8 or target.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(pred.flatten(), target.flatten())[0, 1])


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Coefficient of determination R².
    R² = 1 - SS_res / SS_tot
    """
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


# ── Cardiac-specific ──────────────────────────────────────────────────────────

def ef_mae(pred_ef: np.ndarray, true_ef: np.ndarray) -> float:
    """
    Ejection fraction MAE — the primary EchoNet-Dynamic benchmark metric.
    pred_ef and true_ef in percentage points [0, 100].
    """
    return mae(pred_ef, true_ef)


def ef_r2(pred_ef: np.ndarray, true_ef: np.ndarray) -> float:
    """R² for ejection fraction prediction."""
    return r2_score(pred_ef, true_ef)


# ── Binary classification convenience aliases ─────────────────────────────────

def binary_auc(y_score: np.ndarray, y_true: np.ndarray) -> float:
    """AUC-ROC for binary classification. Signature: (scores, labels)."""
    return auc_roc(y_true, y_score)


def binary_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of correct binary predictions."""
    return float(np.mean(np.asarray(y_pred) == np.asarray(y_true)))


def binary_f1(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """F1 score for binary classification (positive class = 1)."""
    pred = np.asarray(y_pred, dtype=int)
    true = np.asarray(y_true, dtype=int)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


# ── Anatomy-stratified aggregation ────────────────────────────────────────────

def stratified_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    anatomy_labels: list[str],
    min_samples: int = 5,
) -> dict:
    """
    Compute AUC per anatomy family and macro-average.

    Parameters
    ----------
    y_true         : (N,) int labels
    y_score        : (N,) or (N, C) float scores
    anatomy_labels : list of N anatomy family strings
    min_samples    : minimum samples per family to compute AUC

    Returns
    -------
    dict with keys:
        "macro"            : float  macro-average AUC
        "per_anatomy"      : {family: auc}
        "n_families"       : int
        "n_labelled"       : int
    """
    families = sorted(set(anatomy_labels))
    per_anatomy = {}

    for fam in families:
        idx = [i for i, a in enumerate(anatomy_labels) if a == fam]
        if len(idx) < min_samples:
            continue
        y_f = y_true[idx]
        s_f = y_score[idx] if y_score.ndim == 1 else y_score[idx]
        auc = auc_roc(y_f, s_f)
        if not math.isnan(auc):
            per_anatomy[fam] = round(auc, 4)

    macro = float(np.mean(list(per_anatomy.values()))) if per_anatomy else float("nan")

    return {
        "macro":       round(macro, 4) if not math.isnan(macro) else macro,
        "per_anatomy": per_anatomy,
        "n_families":  len(per_anatomy),
        "n_labelled":  int((y_true >= 0).sum()),
    }
