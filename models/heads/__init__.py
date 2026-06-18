"""
models/heads/__init__.py
Downstream task heads.  receive patch_tokens (B, N, D) or cls (B, D) from any backbone via ImageBranch.
"""
from .segmentation_head import (
    LinearSegHead,
    DPTSegHead,
    EnhancedDPTSegHead,
    build_seg_head,
)
from .hierarchical_seg import UPerNetDecoder, build_hierarchical_seg_head
from .finetune_seg import (
    build_finetune_seg_head,
    forward_seg_head,
    is_hierarchical_seg_head,
    encoder_has_hierarchical_features,
    head_type_requires_hierarchy,
)
from .classification_head import (
    LinearClsHead,
    MLPClsHead,
    AttentivePoolClsHead,
    build_cls_head,
)
from .concept_detection_head import ConceptDetectionHead
from .regression_head import RegressionHead, MeasurementHead, VideoRegressionHead, build_video_regression_head
from .mil_head import GatedAttentionMILPool, PatientMILClsHead, MIL_HIDDEN_DIM
from .multifinding_head import MultiFindingBinaryHead, build_multifinding_head

__all__ = [
    # Segmentation
    "LinearSegHead", "DPTSegHead", "EnhancedDPTSegHead", "build_seg_head",
    "UPerNetDecoder", "build_hierarchical_seg_head",
    "build_finetune_seg_head", "forward_seg_head",
    "is_hierarchical_seg_head", "encoder_has_hierarchical_features",
    "head_type_requires_hierarchy",
    # Classification
    "LinearClsHead", "MLPClsHead", "AttentivePoolClsHead", "build_cls_head",
    # Concept detection
    "ConceptDetectionHead",
    # Regression / measurement
    "RegressionHead", "MeasurementHead", "VideoRegressionHead", "build_video_regression_head",
    # MIL / multi-finding
    "GatedAttentionMILPool", "PatientMILClsHead", "MIL_HIDDEN_DIM",
    "MultiFindingBinaryHead", "build_multifinding_head",
]
