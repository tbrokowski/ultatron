"""
label_interface.py  ·  Universal label interface for all downstream heads
=========================================================================

This module defines the complete label contract between the dataset layer
and every downstream head (segmentation, binary/multi-class classification,
detection, regression, CLIP pretraining, patient-level, etc.).

Design goals
------------
* One LabelTarget per (sample × head) pair — a sample can have multiple targets
  (e.g., a thyroid scan with a segmentation mask AND a TIRADS classification)
* Each LabelTarget carries everything the loss function needs: the tensor,
  the loss type, the number of classes, class names, and normalization info
* Head registry is declarative — adding a new dataset means registering a
  HeadSpec, not modifying training code
* Supports all current and anticipated future head types:
    - Binary segmentation (BCE)
    - Multi-class segmentation (CE / Dice)
    - Binary classification (BCE)
    - Multi-class / ordinal classification (CE / ordinal CE)
    - Regression / measurement (L1 / MSE)
    - CLIP image-text contrastive (InfoNCE)
    - Patient-level aggregation (MIL / mean-pooling)
    - Dense detection (FCOS / anchor-free)
    - Promptable SAM-style (point / box / mask prompt)

Anatomy-specific label spaces
------------------------------
Each anatomy family has its own canonical label space registered at module
level.  Datasets map their raw labels into this space during manifest
construction, so the training code never sees dataset-specific integers.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Enumerated head / loss types
# ─────────────────────────────────────────────────────────────────────────────

class HeadType(str, enum.Enum):
    # Pixel-level
    BINARY_SEG          = "binary_seg"          # 1 output channel, BCE
    MULTICLASS_SEG      = "multiclass_seg"       # C output channels, CE/Dice
    INSTANCE_SEG        = "instance_seg"         # mask + class per instance

    # Image-level classification
    BINARY_CLS          = "binary_cls"           # 1 logit, BCE
    MULTICLASS_CLS      = "multiclass_cls"       # C logits, CE
    MULTILABEL_CLS      = "multilabel_cls"       # C logits, independent BCE
    ORDINAL_CLS         = "ordinal_cls"          # CORAL / cumulative logits

    # Regression / measurement
    REGRESSION          = "regression"           # scalar or vector, L1/MSE
    MEASUREMENT         = "measurement"          # physical quantity (mm), L1

    # Detection
    DETECTION           = "detection"            # boxes + class + score
    KEYPOINT            = "keypoint"             # landmark coordinates

    # Cross-modal
    CLIP_PRETRAIN       = "clip_pretrain"        # image–text InfoNCE
    CROSS_MODAL         = "cross_modal"          # image–video alignment

    # Patient / study level
    PATIENT_CLS         = "patient_cls"          # aggregated over study frames
    PATIENT_REGRESSION  = "patient_regression"   # e.g., EF from echo

    # Prompt-based
    PROMPTABLE_SEG      = "promptable_seg"       # SAM-style point/box prompting


class LossType(str, enum.Enum):
    BCE                 = "bce"                  # binary cross-entropy
    BCE_WITH_LOGITS     = "bce_with_logits"
    CE                  = "ce"                   # categorical cross-entropy
    FOCAL               = "focal"                # focal loss
    DICE                = "dice"
    DICE_CE             = "dice_ce"              # combined
    L1                  = "l1"
    MSE                 = "mse"
    SMOOTH_L1           = "smooth_l1"
    INFO_NCE            = "info_nce"             # CLIP contrastive
    CORAL               = "coral"                # ordinal regression
    MIL_BCE             = "mil_bce"              # multiple-instance learning


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Canonical label spaces per anatomy
#     Maps anatomy_family → ordered list of class names.
#     Index = integer class id used everywhere in the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

ANATOMY_LABEL_SPACES: Dict[str, List[str]] = {
    # ── Cardiac ──────────────────────────────────────────────────────────────
    "cardiac": [
        "background",
        "left_ventricle_endo",   # 1
        "left_ventricle_epi",    # 2
        "myocardium",            # 3
        "left_atrium",           # 4
        "right_ventricle_endo",  # 5
        "right_ventricle_epi",   # 6
        "right_atrium",          # 7
        "aortic_root",           # 8
    ],

    # ── Lung (LUS) ────────────────────────────────────────────────────────────
    "lung": [
        "normal",                # 0  (a-line pattern)
        "b_line",                # 1
        "consolidation",         # 2
        "pleural_effusion",      # 3
        "covid",                 # 4 – may overlap with b_line / consolidation
        "pneumonia",             # 5
    ],

    # ── Breast ────────────────────────────────────────────────────────────────
    "breast": [
        "background",
        "benign",                # 1
        "malignant",             # 2
        "normal",                # 3
    ],

    # ── Thyroid ───────────────────────────────────────────────────────────────
    "thyroid": [
        "background",
        "nodule",                # 1 – generic; refined by TIRADS below
        "whole_thyroid",         # 2
    ],
    "thyroid_tirads": [          # TIRADS risk stratification (ordinal)
        "TR1_benign",            # 0
        "TR2_not_suspicious",    # 1
        "TR3_mildly_suspicious", # 2
        "TR4_moderately_suspicious", # 3
        "TR5_highly_suspicious", # 4
    ],

    # ── Fetal ─────────────────────────────────────────────────────────────────
    "fetal_head": [
        "background",
        "skull",                 # 1 – outer boundary
        "brain",                 # 2
        "head_circumference",    # 3 – for regression task use measurement_mm
    ],
    "fetal_planes": [            # fetal plane classification (FETAL_PLANES_DB)
        "Other",                 # 0
        "Fetal brain",           # 1
        "Fetal abdomen",         # 2
        "Fetal femur",           # 3
        "Fetal thorax",          # 4
        "Maternal cervix",       # 5
    ],

    # ── Kidney ────────────────────────────────────────────────────────────────
    "kidney": [
        "background",
        "kidney",                # 1
        "cyst",                  # 2
        "tumor",                 # 3
    ],

    # ── Liver ─────────────────────────────────────────────────────────────────
    "liver": [
        "background",
        "liver",                 # 1
        "lesion",                # 2
        "fatty",                 # 3
    ],

    # ── Ovarian ───────────────────────────────────────────────────────────────
    "ovarian": [
        "background",
        "ovary",                 # 1
        "cyst",                  # 2
        "pcos_follicle",         # 3
    ],

    # ── Gallbladder ───────────────────────────────────────────────────────────
    "gallbladder": [
        "background",
        "gallbladder",           # 1
        "polyp",                 # 2
        "cancer",                # 3
    ],

    # ── Musculoskeletal ───────────────────────────────────────────────────────
    "musculoskeletal": [
        "background",
        "muscle_fascicle",       # 1
        "tendon",                # 2
        "bone_surface",          # 3
        "nerve",                 # 4
    ],

    # ── Vascular ─────────────────────────────────────────────────────────────
    "vascular": [
        "background",
        "vessel_lumen",          # 1
        "intima_media",          # 2
        "plaque",                # 3
    ],

    # ── Multi-organ / other ───────────────────────────────────────────────────
    "multi": [
        "background",
        "organ",                 # generic
    ],
}

# Shorthand: maps anatomy_family → HeadType for the *primary* task
# This is a default; samples can override with their own HeadSpec.
ANATOMY_DEFAULT_HEAD: Dict[str, HeadType] = {
    "cardiac":      HeadType.MULTICLASS_SEG,
    "lung":         HeadType.MULTICLASS_CLS,   # classification + seg
    "breast":       HeadType.BINARY_SEG,
    "thyroid":      HeadType.BINARY_SEG,
    "fetal_head":   HeadType.BINARY_SEG,       # measurement via regression
    "fetal_planes": HeadType.MULTICLASS_CLS,
    "kidney":       HeadType.BINARY_SEG,
    "liver":        HeadType.BINARY_SEG,
    "ovarian":      HeadType.BINARY_SEG,
    "gallbladder":  HeadType.BINARY_SEG,
    "musculoskeletal": HeadType.BINARY_SEG,
    "vascular":     HeadType.BINARY_SEG,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  HeadSpec: declarative config for each head
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadSpec:
    """
    Declarative specification for a downstream head.

    The training code uses this to:
      - Allocate the correct output layer (n_classes)
      - Choose the correct loss function (loss_type)
      - Retrieve the correct label from the batch dict (label_key)
      - Optionally normalise targets (mean/std for regression)

    Multiple HeadSpecs can be active simultaneously.
    """
    head_id: str                        # unique name, e.g. "cardiac_seg"
    head_type: HeadType
    loss_type: LossType
    anatomy_family: str                 # which anatomy this head applies to
    label_key: str                      # key in the batch dict to retrieve target
    n_classes: int = 1                  # output channels
    class_names: List[str] = field(default_factory=list)

    # Loss hyperparameters
    loss_weight: float = 1.0
    class_weights: Optional[List[float]] = None   # for imbalanced CE
    ignore_index: int = -1                         # for CE segmentation
    pos_weight: Optional[float] = None             # for BCE imbalance

    # Regression normalisation (applied at dataset level)
    target_mean: Optional[float] = None
    target_std: Optional[float] = None
    target_min: Optional[float] = None
    target_max: Optional[float] = None

    # CLIP / text
    text_template: Optional[str] = None  # e.g. "An ultrasound of {label}"
    tokenizer_name: str = "openai/clip-vit-base-patch32"

    # Patient-level aggregation
    aggregation: str = "mean"           # "mean" | "max" | "attention" | "mil"

    # Metadata
    dataset_ids: List[str] = field(default_factory=list)   # which datasets feed this head
    enabled: bool = True

    @property
    def is_segmentation(self) -> bool:
        return self.head_type in (
            HeadType.BINARY_SEG, HeadType.MULTICLASS_SEG,
            HeadType.INSTANCE_SEG, HeadType.PROMPTABLE_SEG
        )

    @property
    def is_classification(self) -> bool:
        return self.head_type in (
            HeadType.BINARY_CLS, HeadType.MULTICLASS_CLS,
            HeadType.MULTILABEL_CLS, HeadType.ORDINAL_CLS,
            HeadType.PATIENT_CLS,
        )

    @property
    def is_regression(self) -> bool:
        return self.head_type in (
            HeadType.REGRESSION, HeadType.MEASUREMENT, HeadType.PATIENT_REGRESSION
        )

    @property
    def is_clip(self) -> bool:
        return self.head_type in (HeadType.CLIP_PRETRAIN, HeadType.CROSS_MODAL)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LabelTarget: the actual label tensor delivered by __getitem__
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LabelTarget:
    """
    A single supervision signal attached to a sample.

    The dataset's __getitem__ returns a list[LabelTarget] under the key
    "label_targets".  The collator stacks compatible targets and the trainer
    routes each to the appropriate head using head_id.

    Examples
    --------
    Binary segmentation of a breast lesion:
        LabelTarget(head_id="breast_seg", head_type=BINARY_SEG,
                    loss_type=BCE_WITH_LOGITS,
                    value=Tensor[1, H, W], anatomy="breast")

    Multi-class LUS classification:
        LabelTarget(head_id="lung_cls", head_type=MULTICLASS_CLS,
                    loss_type=CE,
                    value=Tensor(scalar int), anatomy="lung")

    CLIP pretraining:
        LabelTarget(head_id="clip_pretrain", head_type=CLIP_PRETRAIN,
                    loss_type=INFO_NCE,
                    value=Tensor[D] (text embedding), anatomy="lung",
                    text="An ultrasound of b-line artefacts in the lung")

    HC18 head circumference measurement:
        LabelTarget(head_id="fetal_hc_measurement", head_type=MEASUREMENT,
                    loss_type=L1,
                    value=Tensor(scalar mm), anatomy="fetal_head",
                    text=None)
    """
    head_id: str
    head_type: HeadType
    loss_type: LossType
    anatomy: str
    value: Union[Tensor, float, int, None]

    # Optional enrichment
    text: Optional[str] = None               # for CLIP targets
    class_name: Optional[str] = None         # human-readable class
    instance_id: Optional[str] = None        # which lesion/structure
    is_valid: bool = True                    # False = should be masked in loss
    weight: float = 1.0                      # per-sample loss weight

    def to_tensor(self) -> Optional[Tensor]:
        if self.value is None:
            return None
        if isinstance(self.value, Tensor):
            return self.value
        return torch.tensor(self.value)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Head registry: the single place to declare all heads in the system
# ─────────────────────────────────────────────────────────────────────────────

class HeadRegistry:
    """
    Central registry of all downstream heads.

    Usage
    -----
    # Register a custom head
    registry.register(HeadSpec(
        head_id="lus_aline_binary",
        head_type=HeadType.BINARY_CLS,
        loss_type=LossType.BCE_WITH_LOGITS,
        anatomy_family="lung",
        label_key="lung_aline_label",
        n_classes=1,
        class_names=["a_line"],
        dataset_ids=["LUS-multicenter-2025", "COVIDx-US"],
    ))

    # Look up the head for a given anatomy + task
    spec = registry.get("lus_aline_binary")
    specs = registry.for_anatomy("lung")
    """

    def __init__(self):
        self._registry: Dict[str, HeadSpec] = {}

    def register(self, spec: HeadSpec) -> None:
        self._registry[spec.head_id] = spec

    def get(self, head_id: str) -> Optional[HeadSpec]:
        return self._registry.get(head_id)

    def for_anatomy(self, anatomy: str) -> List[HeadSpec]:
        return [s for s in self._registry.values() if s.anatomy_family == anatomy and s.enabled]

    def for_head_type(self, head_type: HeadType) -> List[HeadSpec]:
        return [s for s in self._registry.values() if s.head_type == head_type and s.enabled]

    def all_enabled(self) -> List[HeadSpec]:
        return [s for s in self._registry.values() if s.enabled]

    def disable(self, head_id: str) -> None:
        if head_id in self._registry:
            self._registry[head_id].enabled = False

    def enable(self, head_id: str) -> None:
        if head_id in self._registry:
            self._registry[head_id].enabled = True

    def summary(self) -> str:
        lines = [f"HeadRegistry: {len(self._registry)} heads registered"]
        for spec in sorted(self._registry.values(), key=lambda s: s.anatomy_family):
            status = "✓" if spec.enabled else "✗"
            lines.append(
                f"  {status} [{spec.anatomy_family:20s}] {spec.head_id:40s} "
                f"{spec.head_type.value:25s} {spec.loss_type.value:20s} n={spec.n_classes}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Default head registry — populated at import time
#     All heads in the system are registered here.
# ─────────────────────────────────────────────────────────────────────────────

HEAD_REGISTRY = HeadRegistry()

def _register_default_heads():
    R = HEAD_REGISTRY

    # ── Cardiac ───────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="cardiac_seg_multiclass",
        head_type=HeadType.MULTICLASS_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="cardiac",
        label_key="seg_mask",
        n_classes=len(ANATOMY_LABEL_SPACES["cardiac"]),
        class_names=ANATOMY_LABEL_SPACES["cardiac"],
        dataset_ids=["CAMUS", "EchoNet-Dynamic", "EchoNet-LVH", "MIMIC-IV-ECHO",
                     "MITEA", "CardiacUDC", "EchoNet-Pediatric"],
    ))
    R.register(HeadSpec(
        head_id="cardiac_ef_regression",
        head_type=HeadType.PATIENT_REGRESSION,
        loss_type=LossType.SMOOTH_L1,
        anatomy_family="cardiac",
        label_key="ef_value",
        n_classes=1,
        target_mean=55.0, target_std=12.0,   # EF% normalisation
        dataset_ids=["EchoNet-Dynamic", "MIMIC-IV-ECHO"],
        aggregation="mean",
    ))
    R.register(HeadSpec(
        head_id="cardiac_clip_pretrain",
        head_type=HeadType.CLIP_PRETRAIN,
        loss_type=LossType.INFO_NCE,
        anatomy_family="cardiac",
        label_key="clip_text",
        n_classes=512,
        text_template="An echocardiogram showing {label}",
        dataset_ids=["CAMUS", "EchoNet-Dynamic", "MIMIC-IV-ECHO"],
    ))

    # ── Lung / LUS ────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="lus_aline_binary",
        head_type=HeadType.BINARY_CLS,
        loss_type=LossType.BCE_WITH_LOGITS,
        anatomy_family="lung",
        label_key="lus_aline_label",
        n_classes=1,
        class_names=["a_line"],
        pos_weight=2.0,   # a-lines less common in pathological datasets
        dataset_ids=["LUS-multicenter-2025", "COVIDx-US", "POCUS-LUS"],
    ))
    R.register(HeadSpec(
        head_id="lus_multiclass_cls",
        head_type=HeadType.MULTICLASS_CLS,
        loss_type=LossType.CE,
        anatomy_family="lung",
        label_key="cls_label",
        n_classes=len(ANATOMY_LABEL_SPACES["lung"]),
        class_names=ANATOMY_LABEL_SPACES["lung"],
        dataset_ids=["COVIDx-US", "POCUS-LUS", "LUS-multicenter-2025"],
    ))
    R.register(HeadSpec(
        head_id="lus_seg_pleural",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.BCE_WITH_LOGITS,
        anatomy_family="lung",
        label_key="seg_mask",
        n_classes=1,
        class_names=["pleural_line"],
        dataset_ids=["LUS-multicenter-2025"],
    ))
    R.register(HeadSpec(
        head_id="lung_clip_pretrain",
        head_type=HeadType.CLIP_PRETRAIN,
        loss_type=LossType.INFO_NCE,
        anatomy_family="lung",
        label_key="clip_text",
        n_classes=512,
        text_template="A lung ultrasound showing {label}",
        dataset_ids=["COVIDx-US", "LUS-multicenter-2025"],
    ))

    # ── Breast ────────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="breast_lesion_seg",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="breast",
        label_key="seg_mask",
        n_classes=1,
        class_names=["lesion"],
        dataset_ids=["BUSI", "BUS-BRA", "BUS-UC", "BrEaST", "BUID"],
    ))
    R.register(HeadSpec(
        head_id="breast_malignancy_cls",
        head_type=HeadType.BINARY_CLS,
        loss_type=LossType.BCE_WITH_LOGITS,
        anatomy_family="breast",
        label_key="malignancy_label",
        n_classes=1,
        class_names=["malignant"],
        dataset_ids=["BUSI", "BUS-BRA", "BrEaST"],
    ))
    R.register(HeadSpec(
        head_id="breast_birads_ordinal",
        head_type=HeadType.ORDINAL_CLS,
        loss_type=LossType.CORAL,
        anatomy_family="breast",
        label_key="birads_label",
        n_classes=6,      # BI-RADS 0-5
        class_names=["BIRADS0","BIRADS1","BIRADS2","BIRADS3","BIRADS4","BIRADS5"],
        dataset_ids=["BUS-BRA"],
    ))

    # ── Thyroid ───────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="thyroid_nodule_seg",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="thyroid",
        label_key="seg_mask",
        n_classes=1,
        class_names=["nodule"],
        dataset_ids=["TN3K", "TN5000", "DDTI", "TNSCUI", "TG3K"],
    ))
    R.register(HeadSpec(
        head_id="thyroid_tirads_ordinal",
        head_type=HeadType.ORDINAL_CLS,
        loss_type=LossType.CORAL,
        anatomy_family="thyroid",
        label_key="tirads_label",
        n_classes=5,
        class_names=ANATOMY_LABEL_SPACES["thyroid_tirads"],
        dataset_ids=["TN5000", "DDTI"],
    ))

    # ── Fetal ─────────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="fetal_head_seg",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="fetal_head",
        label_key="seg_mask",
        n_classes=1,
        class_names=["fetal_head"],
        dataset_ids=["HC18", "ACOUSLIC-AI"],
    ))
    R.register(HeadSpec(
        head_id="fetal_hc_measurement",
        head_type=HeadType.MEASUREMENT,
        loss_type=LossType.SMOOTH_L1,
        anatomy_family="fetal_head",
        label_key="measurement_mm",
        n_classes=1,
        target_mean=250.0, target_std=50.0,    # HC in mm (approx. range 100-400)
        dataset_ids=["HC18"],
    ))
    R.register(HeadSpec(
        head_id="fetal_plane_cls",
        head_type=HeadType.MULTICLASS_CLS,
        loss_type=LossType.CE,
        anatomy_family="fetal_planes",
        label_key="cls_label",
        n_classes=len(ANATOMY_LABEL_SPACES["fetal_planes"]),
        class_names=ANATOMY_LABEL_SPACES["fetal_planes"],
        dataset_ids=["FETAL_PLANES_DB"],
    ))

    # ── Kidney ────────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="kidney_seg",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="kidney",
        label_key="seg_mask",
        n_classes=1,
        class_names=["kidney"],
        dataset_ids=["KidneyUS"],
    ))

    # ── Liver ─────────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="liver_seg",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="liver",
        label_key="seg_mask",
        n_classes=1,
        class_names=["liver"],
        dataset_ids=["AUL", "BEHSOF"],
    ))

    # ── Ovarian ───────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="ovarian_seg",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="ovarian",
        label_key="seg_mask",
        n_classes=1,
        class_names=["ovary_structure"],
        dataset_ids=["MMOTU-2D", "PCOSGen"],
    ))
    R.register(HeadSpec(
        head_id="pcos_binary_cls",
        head_type=HeadType.BINARY_CLS,
        loss_type=LossType.BCE_WITH_LOGITS,
        anatomy_family="ovarian",
        label_key="pcos_label",
        n_classes=1,
        class_names=["pcos"],
        dataset_ids=["PCOSGen"],
    ))

    # ── Gallbladder ───────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="gallbladder_seg",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="gallbladder",
        label_key="seg_mask",
        n_classes=1,
        class_names=["gallbladder"],
        dataset_ids=["GBCU"],
    ))
    R.register(HeadSpec(
        head_id="gallbladder_cancer_cls",
        head_type=HeadType.BINARY_CLS,
        loss_type=LossType.FOCAL,
        anatomy_family="gallbladder",
        label_key="cancer_label",
        n_classes=1,
        class_names=["malignant"],
        pos_weight=10.0,    # heavily imbalanced
        dataset_ids=["GBCU"],
    ))

    # ── Musculoskeletal ───────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="msk_fascicle_seg",
        head_type=HeadType.BINARY_SEG,
        loss_type=LossType.DICE_CE,
        anatomy_family="musculoskeletal",
        label_key="seg_mask",
        n_classes=1,
        class_names=["fascicle"],
        dataset_ids=["FALLMUD", "STMUS-NDA"],
    ))

    # ── Vascular ─────────────────────────────────────────────────────────────
    R.register(HeadSpec(
        head_id="carotid_imt_regression",
        head_type=HeadType.MEASUREMENT,
        loss_type=LossType.SMOOTH_L1,
        anatomy_family="vascular",
        label_key="measurement_mm",
        n_classes=1,
        target_mean=0.7, target_std=0.2,     # IMT in mm (normal ~0.5-1.0)
        dataset_ids=["CCA-US"],
    ))

    # ── Universal CLIP pretraining ────────────────────────────────────────────
    # This is active when training_mode == "clip" regardless of anatomy
    R.register(HeadSpec(
        head_id="us_clip_universal",
        head_type=HeadType.CLIP_PRETRAIN,
        loss_type=LossType.INFO_NCE,
        anatomy_family="multi",
        label_key="clip_text",
        n_classes=512,
        text_template="An ultrasound image of {anatomy} showing {label}",
        dataset_ids=[],    # receives all labelled samples
        enabled=False,     # enabled only when training_mode == "clip"
    ))

_register_default_heads()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Label builder helpers
#     Called by dataset __getitem__ to construct LabelTarget lists
# ─────────────────────────────────────────────────────────────────────────────

def build_seg_target(
    mask_tensor: Optional[Tensor],
    head_id: str,
    anatomy: str,
    instance_id: Optional[str] = None,
) -> Optional[LabelTarget]:
    """Construct a segmentation LabelTarget from a mask tensor."""
    spec = HEAD_REGISTRY.get(head_id)
    if spec is None or mask_tensor is None:
        return None
    return LabelTarget(
        head_id=head_id,
        head_type=spec.head_type,
        loss_type=spec.loss_type,
        anatomy=anatomy,
        value=mask_tensor,
        instance_id=instance_id,
        is_valid=True,
    )


def build_cls_target(
    label_int: Optional[int],
    head_id: str,
    anatomy: str,
    class_name: Optional[str] = None,
) -> Optional[LabelTarget]:
    """Construct a classification LabelTarget from an integer label."""
    spec = HEAD_REGISTRY.get(head_id)
    if spec is None or label_int is None or label_int < 0:
        return None
    return LabelTarget(
        head_id=head_id,
        head_type=spec.head_type,
        loss_type=spec.loss_type,
        anatomy=anatomy,
        value=torch.tensor(label_int, dtype=torch.long),
        class_name=class_name,
        is_valid=True,
    )


def build_regression_target(
    value_mm: Optional[float],
    head_id: str,
    anatomy: str,
) -> Optional[LabelTarget]:
    """Construct a regression/measurement LabelTarget."""
    spec = HEAD_REGISTRY.get(head_id)
    if spec is None or value_mm is None:
        return None
    # Normalise
    v = value_mm
    if spec.target_mean is not None and spec.target_std is not None:
        v = (v - spec.target_mean) / (spec.target_std + 1e-8)
    return LabelTarget(
        head_id=head_id,
        head_type=spec.head_type,
        loss_type=spec.loss_type,
        anatomy=anatomy,
        value=torch.tensor(v, dtype=torch.float32),
        is_valid=True,
    )


def build_clip_target(
    label_text: str,
    head_id: str,
    anatomy: str,
) -> LabelTarget:
    """Construct a CLIP text LabelTarget (text not yet tokenized)."""
    spec = HEAD_REGISTRY.get(head_id)
    loss = spec.loss_type if spec else LossType.INFO_NCE
    head = spec.head_type if spec else HeadType.CLIP_PRETRAIN
    return LabelTarget(
        head_id=head_id,
        head_type=head,
        loss_type=loss,
        anatomy=anatomy,
        value=None,          # tokenization happens in collator
        text=label_text,
        is_valid=True,
    )


def resolve_clip_text(entry_dict: dict, label_ontology: Optional[str]) -> Optional[str]:
    """
    Build a natural-language description for CLIP pretraining from a manifest entry.
    Returns None if insufficient label information is available.
    """
    anatomy = entry_dict.get("anatomy_family", "unknown")
    label   = label_ontology or entry_dict.get("task_type", "structure")

    spec = HEAD_REGISTRY.get(f"{anatomy}_clip_pretrain") or \
           HEAD_REGISTRY.get("us_clip_universal")
    if spec is None:
        return None

    template = spec.text_template or "An ultrasound of {anatomy} showing {label}"
    try:
        return template.format(anatomy=anatomy.replace("_", " "), label=label.replace("_", " "))
    except KeyError:
        return f"An ultrasound image of {anatomy.replace('_', ' ')}"
