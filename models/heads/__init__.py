"""
models/heads/__init__.py
Downstream task heads.  receive patch_tokens (B, N, D) or cls (B, D) from any backbone via ImageBranch.
"""
from .segmentation_head import (
    LinearSegHead,
    DPTSegHead,
    build_seg_head,
)
from .classification_head import (
    LinearClsHead,
    MLPClsHead,
    AttentivePoolClsHead,
    build_cls_head,
)
from .concept_detection_head import ConceptDetectionHead
from .regression_head import RegressionHead, MeasurementHead

__all__ = [
    # Segmentation
    "LinearSegHead", "DPTSegHead", "build_seg_head",
    # Classification
    "LinearClsHead", "MLPClsHead", "AttentivePoolClsHead", "build_cls_head",
    # Concept detection
    "ConceptDetectionHead",
    # Regression / measurement
    "RegressionHead", "MeasurementHead",
]
