"""
label_spec.py  ·  label specification system
==================================================

This module defines the complete label interface for all downstream heads.
The LabelSpec is the single, unified description of what a sample's annotation
means and how it should be used in training or evaluation.

Design principle: ONE label specification per Instance, carrying everything
the training code needs to know without any implicit conventions.

Supported task heads
--------------------
  segmentation      : pixel-level mask (binary or multiclass)
  binary_cls        : single sigmoid output (present / absent)
  multiclass_cls    : softmax over N classes (mutually exclusive)
  multilabel_cls    : N independent sigmoid outputs (co-occurring)
  detection         : bounding boxes + class ids
  clip              : image ↔ text contrastive (CLIP / SigLIP style)
  regression        : continuous scalar target (EF %, HC measurement)
  measurement       : physical measurement in mm
  patient_cls       : study/patient-level classification (pooled frames)
  sequence_cls      : temporal sequence-level classification
  grounding         : text → spatial region (like SAM with text prompt)

Loss types
----------
  bce       : BinaryCrossEntropy  (binary_cls, binary segmentation)
  ce        : CrossEntropy        (multiclass_cls, multiclass segmentation)
  dice      : Dice loss           (segmentation, often combined with bce/ce)
  focal     : Focal loss          (class-imbalanced segmentation)
  mse       : Mean Squared Error  (regression)
  mae       : Mean Absolute Error (measurement)
  contrastive: SimCLR / NT-Xent   (SSL)
  clip_nce  : InfoNCE / CLIP loss (cross-modal)
  combo     : sum of weighted sub-losses (specified via loss_weights)

Anatomy-standardised label vocabularies
-----------------------------------------
Each anatomy family has a canonical set of class names.  When an adapter
emits an instance, it must map its raw labels to these canonical names.
The LabelSpec carries both the raw and canonical names for traceability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


# ── Enumerations ──────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """Task head type — what does this label train?"""
    SEGMENTATION     = "segmentation"
    BINARY_CLS       = "binary_cls"
    MULTICLASS_CLS   = "multiclass_cls"
    MULTILABEL_CLS   = "multilabel_cls"
    DETECTION        = "detection"
    CLIP             = "clip"
    REGRESSION       = "regression"
    MEASUREMENT      = "measurement"
    PATIENT_CLS      = "patient_cls"
    SEQUENCE_CLS     = "sequence_cls"
    GROUNDING        = "grounding"
    SSL_ONLY         = "ssl_only"


class LossType(str, Enum):
    BCE          = "bce"
    CE           = "ce"
    DICE         = "dice"
    FOCAL        = "focal"
    MSE          = "mse"
    MAE          = "mae"
    CONTRASTIVE  = "contrastive"
    CLIP_NCE     = "clip_nce"
    COMBO        = "combo"
    NONE         = "none"


class MaskEncoding(str, Enum):
    """How the segmentation mask is encoded."""
    BINARY      = "binary"      # 0/1 single channel
    MULTICLASS  = "multiclass"  # integer-encoded class map, shape H×W
    MULTILABEL  = "multilabel"  # C×H×W binary channels
    PROBABILITY = "probability" # float [0,1] for soft labels


# ── Per-anatomy canonical label vocabularies ──────────────────────────────────
# These are the canonical class names for each anatomy family.
# Adapters MUST map to these names so all downstream heads share a vocabulary.

ANATOMY_LABEL_VOCAB: Dict[str, List[str]] = {
    "cardiac": [
        "left_ventricle", "myocardium", "left_atrium",
        "right_ventricle", "right_atrium", "aorta", "pericardium",
        "lv_outflow_tract", "mitral_valve", "aortic_valve",
    ],
    "lung": [
        "normal", "covid", "pneumonia", "consolidation",
        "pleural_effusion", "b_lines", "a_lines",
        "lung_sliding_present", "lung_sliding_absent",
    ],
    "breast": [
        "benign_mass", "malignant_mass", "cyst", "background",
        "birads_0", "birads_1", "birads_2", "birads_3",
        "birads_4", "birads_5",
    ],
    "thyroid": [
        "nodule", "background", "benign", "malignant",
        "follicular", "papillary", "tirads_1", "tirads_2",
        "tirads_3", "tirads_4", "tirads_5",
    ],
    "fetal_head": [
        "fetal_head", "head_circumference_contour",
        "cranium", "cavum_septi_pellucidi", "cerebellum",
    ],
    "fetal_abdomen": [
        "fetal_abdomen", "fetal_brain", "fetal_femur",
        "fetal_thorax", "fetal_face", "placenta",
        "maternal_cervix", "other_fetal_plane",
    ],
    "liver": [
        "liver", "hepatocellular_carcinoma", "hemangioma",
        "metastasis", "portal_vein", "bile_duct",
    ],
    "kidney": [
        "kidney", "renal_cortex", "renal_sinus",
        "cyst", "stone", "hydronephrosis",
    ],
    "gallbladder": [
        "gallbladder", "gallstone", "polyp",
        "cholecystitis", "wall_thickening",
    ],
    "ovarian": [
        "ovary", "follicle", "corpus_luteum",
        "simple_cyst", "complex_cyst", "pcos",
        "endometrioma", "dermoid",
    ],
    "prostate": [
        "prostate", "peripheral_zone", "transitional_zone",
        "seminal_vesicle", "benign_hyperplasia", "carcinoma",
    ],
    "muscle": [
        "muscle", "tendon", "fascia",
        "intramuscular_fat", "tear", "hematoma",
    ],
    "nerve": [
        "nerve", "brachial_plexus", "median_nerve",
        "ulnar_nerve", "sciatic_nerve",
    ],
    "vascular": [
        "artery", "vein", "carotid_artery", "jugular_vein",
        "intima_media", "plaque", "thrombus",
    ],
    "spine": [
        "vertebral_body", "disc", "spinal_canal",
        "facet_joint", "epidural_space",
    ],
    "ocular": [
        "retina", "optic_nerve", "vitreous",
        "detachment", "choroid",
    ],
    "multi": [],   # heterogeneous — use per-instance labels
    "other": [],
}


# ── Loss configuration ────────────────────────────────────────────────────────

@dataclass
class LossConfig:
    """
    Describes how to compute the loss for a given LabelSpec.
    Supports composite losses (e.g., BCE + Dice).
    """
    primary: LossType = LossType.BCE
    auxiliary: Optional[LossType] = None
    # Weights for composite loss: primary_weight * L_primary + aux_weight * L_aux
    primary_weight: float = 1.0
    auxiliary_weight: float = 0.5
    # Class weights for imbalanced problems (length = num_classes)
    class_weights: Optional[List[float]] = None
    # Focal loss gamma parameter
    focal_gamma: float = 2.0
    # CLIP temperature parameter
    clip_temperature: float = 0.07
    # Ignore index for CE/segmentation (padding or unlabeled regions)
    ignore_index: int = -1
    # Label smoothing for CE
    label_smoothing: float = 0.0

    def to_dict(self) -> dict:
        return {
            "primary": self.primary.value,
            "auxiliary": self.auxiliary.value if self.auxiliary else None,
            "primary_weight": self.primary_weight,
            "auxiliary_weight": self.auxiliary_weight,
            "class_weights": self.class_weights,
            "focal_gamma": self.focal_gamma,
            "clip_temperature": self.clip_temperature,
            "ignore_index": self.ignore_index,
            "label_smoothing": self.label_smoothing,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LossConfig":
        d = d.copy()
        d["primary"] = LossType(d["primary"])
        if d.get("auxiliary"): d["auxiliary"] = LossType(d["auxiliary"])
        return cls(**d)


# ── LabelSpec ─────────────────────────────────────────────────────────────────

@dataclass
class LabelSpec:
    """
    Complete specification of what a label/annotation means and how to train with it.

    One LabelSpec is attached to each Instance.  The same instance can carry
    multiple LabelSpecs (e.g., a mask for segmentation AND a class for
    classification), which are stored as a list on the Instance.

    Fields
    ------
    task        : TaskType enum – what head this label targets
    loss        : LossConfig  – how to compute the loss
    num_classes : int – 1 for binary/regression, N for multiclass
    class_names : list of canonical class names (from ANATOMY_LABEL_VOCAB)
    anatomy_family : which anatomy family this label belongs to (for
                     standardizing class indices across datasets)
    is_binary   : shortcut flag — True if exactly 1 output neuron + sigmoid
    label_value : for classification: integer class index (0-based)
                  for regression: float value
                  for multi-label: list of 0/1 for each class
    text_label  : natural language description of this instance/label
                  (used for CLIP/text-grounded training)
    mask_encoding : how the segmentation mask is encoded (when task=segmentation)
    # Traceability
    label_raw   : original string label from the dataset (before ontology mapping)
    label_source: which dataset this label came from
    # Extensions
    report_text : longer clinical text (radiology report excerpt)
    attributes  : arbitrary key-value metadata (future extensions)
    """
    task: TaskType
    loss: LossConfig
    num_classes: int
    class_names: List[str]
    anatomy_family: str

    # Value fields
    label_value: Optional[Union[int, float, List[int]]] = None
    text_label: Optional[str] = None
    report_text: Optional[str] = None
    mask_encoding: MaskEncoding = MaskEncoding.BINARY
    is_binary: bool = False

    # Traceability
    label_raw: Optional[str] = None
    label_source: Optional[str] = None

    # Future extensions
    attributes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task.value,
            "loss": self.loss.to_dict(),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "anatomy_family": self.anatomy_family,
            "label_value": self.label_value,
            "text_label": self.text_label,
            "report_text": self.report_text,
            "mask_encoding": self.mask_encoding.value,
            "is_binary": self.is_binary,
            "label_raw": self.label_raw,
            "label_source": self.label_source,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelSpec":
        d = d.copy()
        d["task"] = TaskType(d["task"])
        d["loss"] = LossConfig.from_dict(d["loss"])
        d["mask_encoding"] = MaskEncoding(d.get("mask_encoding", "binary"))
        return cls(**d)

    @classmethod
    def from_task_type(
        cls,
        task: TaskType,
        anatomy_family: str = "other",
        num_classes: int = 1,
        class_names: Optional[List[str]] = None,
        label_raw: str = "",
        label_source: str = "",
    ) -> "LabelSpec":
        """
        Convenience constructor: build a sensible LabelSpec from a TaskType.

        Selects default loss configuration based on task type:
          SEGMENTATION       → Dice + BCE
          BINARY_CLS         → BCE
          MULTICLASS_CLS     → CE
          MULTILABEL_CLS     → BCE
          REGRESSION /
          MEASUREMENT        → MSE
          DETECTION          → BCE (box)
          CLIP               → CLIP_NCE
          SSL_ONLY           → NONE

        Parameters
        ----------
        task           : TaskType enum value
        anatomy_family : anatomy family string (used to look up class_names if None)
        num_classes    : number of output classes / heads
        class_names    : optional explicit class name list; falls back to
                         ANATOMY_LABEL_VOCAB[anatomy_family] or generic names
        label_raw      : original label string before ontology mapping
        label_source   : dataset the label came from
        """
        if class_names is None:
            vocab = ANATOMY_LABEL_VOCAB.get(anatomy_family, [])
            class_names = vocab[:num_classes] if vocab else [
                f"class_{i}" for i in range(num_classes)
            ]

        _loss_map: Dict[TaskType, LossConfig] = {
            TaskType.SEGMENTATION:   LossConfig(primary=LossType.DICE,     auxiliary=LossType.BCE),
            TaskType.BINARY_CLS:     LossConfig(primary=LossType.BCE),
            TaskType.MULTICLASS_CLS: LossConfig(primary=LossType.CE),
            TaskType.MULTILABEL_CLS: LossConfig(primary=LossType.BCE),
            TaskType.DETECTION:      LossConfig(primary=LossType.BCE),
            TaskType.REGRESSION:     LossConfig(primary=LossType.MSE),
            TaskType.MEASUREMENT:    LossConfig(primary=LossType.MSE),
            TaskType.PATIENT_CLS:    LossConfig(primary=LossType.CE),
            TaskType.SEQUENCE_CLS:   LossConfig(primary=LossType.CE),
            TaskType.GROUNDING:      LossConfig(primary=LossType.BCE),
            TaskType.CLIP:           LossConfig(primary=LossType.CLIP_NCE),
            TaskType.SSL_ONLY:       LossConfig(primary=LossType.NONE),
        }
        loss      = _loss_map.get(task, LossConfig(primary=LossType.BCE))
        is_binary = (num_classes == 1 and task in (
            TaskType.SEGMENTATION, TaskType.BINARY_CLS, TaskType.REGRESSION,
            TaskType.MEASUREMENT,
        ))

        return cls(
            task=task,
            loss=loss,
            num_classes=num_classes,
            class_names=class_names,
            anatomy_family=anatomy_family,
            is_binary=is_binary,
            label_raw=label_raw,
            label_source=label_source,
        )


# ── Preset label spec factories ───────────────────────────────────────────────
# These are convenience constructors for the most common configurations.
# Adapters should use these rather than constructing LabelSpec from scratch.

def binary_segmentation_spec(
    anatomy_family: str,
    class_name: str,
    label_raw: str = "",
    label_source: str = "",
    use_dice: bool = True,
) -> LabelSpec:
    """Binary (0/1) segmentation mask. E.g., fetal head contour."""
    loss = LossConfig(
        primary=LossType.DICE if use_dice else LossType.BCE,
        auxiliary=LossType.BCE if use_dice else None,
        primary_weight=1.0,
        auxiliary_weight=0.5,
    )
    return LabelSpec(
        task=TaskType.SEGMENTATION,
        loss=loss,
        num_classes=1,
        class_names=[class_name],
        anatomy_family=anatomy_family,
        mask_encoding=MaskEncoding.BINARY,
        is_binary=True,
        label_raw=label_raw,
        label_source=label_source,
    )


def multiclass_segmentation_spec(
    anatomy_family: str,
    class_names: Optional[List[str]] = None,
    label_raw: str = "",
    label_source: str = "",
) -> LabelSpec:
    """Multi-class segmentation mask. E.g., cardiac structures."""
    if class_names is None:
        class_names = ANATOMY_LABEL_VOCAB.get(anatomy_family, ["structure"])
    n = len(class_names)
    return LabelSpec(
        task=TaskType.SEGMENTATION,
        loss=LossConfig(primary=LossType.CE, auxiliary=LossType.DICE,
                        primary_weight=1.0, auxiliary_weight=0.5, ignore_index=0),
        num_classes=n,
        class_names=class_names,
        anatomy_family=anatomy_family,
        mask_encoding=MaskEncoding.MULTICLASS,
        is_binary=False,
        label_raw=label_raw,
        label_source=label_source,
    )


def binary_classification_spec(
    anatomy_family: str,
    class_name: str,
    label_value: int,
    label_raw: str = "",
    label_source: str = "",
    text_label: Optional[str] = None,
) -> LabelSpec:
    """Binary image-level classification. E.g., A-lines present/absent."""
    return LabelSpec(
        task=TaskType.BINARY_CLS,
        loss=LossConfig(primary=LossType.BCE),
        num_classes=1,
        class_names=[class_name],
        anatomy_family=anatomy_family,
        is_binary=True,
        label_value=label_value,
        text_label=text_label or class_name,
        label_raw=label_raw,
        label_source=label_source,
    )


def multiclass_classification_spec(
    anatomy_family: str,
    class_names: List[str],
    label_value: int,
    label_raw: str = "",
    label_source: str = "",
    text_label: Optional[str] = None,
    class_weights: Optional[List[float]] = None,
) -> LabelSpec:
    """Multiclass (softmax) image-level classification."""
    return LabelSpec(
        task=TaskType.MULTICLASS_CLS,
        loss=LossConfig(
            primary=LossType.CE,
            class_weights=class_weights,
            label_smoothing=0.1,
        ),
        num_classes=len(class_names),
        class_names=class_names,
        anatomy_family=anatomy_family,
        is_binary=False,
        label_value=label_value,
        text_label=text_label or class_names[label_value],
        label_raw=label_raw,
        label_source=label_source,
    )


def multilabel_classification_spec(
    anatomy_family: str,
    class_names: List[str],
    label_values: List[int],
    label_raw: str = "",
    label_source: str = "",
) -> LabelSpec:
    """Multi-label (independent sigmoids) classification."""
    return LabelSpec(
        task=TaskType.MULTILABEL_CLS,
        loss=LossConfig(primary=LossType.BCE),
        num_classes=len(class_names),
        class_names=class_names,
        anatomy_family=anatomy_family,
        is_binary=False,
        label_value=label_values,
        label_raw=label_raw,
        label_source=label_source,
    )


def clip_spec(
    anatomy_family: str,
    text_label: str,
    class_names: Optional[List[str]] = None,
    label_raw: str = "",
    label_source: str = "",
) -> LabelSpec:
    """CLIP-style image↔text contrastive training spec."""
    return LabelSpec(
        task=TaskType.CLIP,
        loss=LossConfig(primary=LossType.CLIP_NCE, clip_temperature=0.07),
        num_classes=0,
        class_names=class_names or [],
        anatomy_family=anatomy_family,
        text_label=text_label,
        label_raw=label_raw,
        label_source=label_source,
    )


def regression_spec(
    anatomy_family: str,
    target_name: str,
    label_value: float,
    label_raw: str = "",
    label_source: str = "",
) -> LabelSpec:
    """Regression (continuous output). E.g., ejection fraction, HC diameter."""
    return LabelSpec(
        task=TaskType.REGRESSION,
        loss=LossConfig(primary=LossType.MSE),
        num_classes=1,
        class_names=[target_name],
        anatomy_family=anatomy_family,
        is_binary=False,
        label_value=label_value,
        label_raw=label_raw,
        label_source=label_source,
    )


def patient_classification_spec(
    anatomy_family: str,
    class_names: List[str],
    label_value: int,
    label_raw: str = "",
    label_source: str = "",
) -> LabelSpec:
    """
    Patient-level (study-level) classification.
    Labels are aggregated across all frames for a patient.
    """
    return LabelSpec(
        task=TaskType.PATIENT_CLS,
        loss=LossConfig(primary=LossType.CE),
        num_classes=len(class_names),
        class_names=class_names,
        anatomy_family=anatomy_family,
        label_value=label_value,
        label_raw=label_raw,
        label_source=label_source,
    )


# ── Task configuration (dataset-level) ───────────────────────────────────────

@dataclass
class TaskConfig:
    """
    Dataset-level task configuration.
    Specifies which task/head to activate when loading this dataset
    for a specific training or evaluation run.

    This is the 'mode' you pass to the DownstreamDataset to tell it
    what labels to return. Different configs can be used for:
    - Pre-training with CLIP labels on all data
    - Fine-tuning the segmentation head on cardiac data
    - Evaluating the classification head on lung B-lines
    - Patient-level prognosis classification

    active_tasks      : which task types to return labels for
    anatomy_filter    : only load samples from these anatomies (None = all)
    split_filter      : which split(s) to include
    require_text      : only include samples that have a text_label
    require_mask      : only include samples that have a segmentation mask
    patient_aggregate : if True, group by study_id for patient-level tasks
    """
    active_tasks: List[TaskType]
    anatomy_filter: Optional[List[str]] = None
    split_filter: Optional[List[str]] = None
    require_text: bool = False
    require_mask: bool = False
    require_box: bool = False
    patient_aggregate: bool = False

    # Convenience constructors
    @classmethod
    def image_ssl(cls) -> "TaskConfig":
        return cls(active_tasks=[TaskType.SSL_ONLY])

    @classmethod
    def segmentation(cls, anatomy: Optional[str] = None) -> "TaskConfig":
        f = [anatomy] if anatomy else None
        return cls(active_tasks=[TaskType.SEGMENTATION], anatomy_filter=f,
                   require_mask=True)

    @classmethod
    def binary_classification(cls, anatomy: Optional[str] = None) -> "TaskConfig":
        f = [anatomy] if anatomy else None
        return cls(active_tasks=[TaskType.BINARY_CLS], anatomy_filter=f)

    @classmethod
    def multiclass_classification(cls, anatomy: Optional[str] = None) -> "TaskConfig":
        f = [anatomy] if anatomy else None
        return cls(active_tasks=[TaskType.MULTICLASS_CLS], anatomy_filter=f)

    @classmethod
    def clip_pretraining(cls) -> "TaskConfig":
        return cls(active_tasks=[TaskType.CLIP], require_text=True)

    @classmethod
    def patient_level(cls, anatomy: Optional[str] = None) -> "TaskConfig":
        f = [anatomy] if anatomy else None
        return cls(active_tasks=[TaskType.PATIENT_CLS], anatomy_filter=f,
                   patient_aggregate=True)

    @classmethod
    def all_supervised(cls) -> "TaskConfig":
        return cls(active_tasks=[
            TaskType.SEGMENTATION, TaskType.BINARY_CLS,
            TaskType.MULTICLASS_CLS, TaskType.MULTILABEL_CLS,
            TaskType.DETECTION, TaskType.REGRESSION,
        ])
