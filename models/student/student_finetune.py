"""
models/student/student_finetune.py  ·  Task-head integration for the student encoder
=====================================================================================

StudentFinetuneModel
--------------------
Wires the pretrained HieraStudentBackbone to the existing task-specific head
library.  All head instances live inside this module; the backbone is imported
as a dependency.

Backbone ↔ head output mapping
-------------------------------
HieraStudentBackbone produces:
  global : (B, D4)             ← replaces CLS token
  F1     : (B, T, N1, D1)      ← stride-4   dense tokens
  F2     : (B, T, N2, D2)      ← stride-8
  F3     : (B, T, N3, D3)      ← stride-16
  F4     : (B, T, N4, D4)      ← stride-32  ← replaces "patch_tokens"

Legacy heads expect (cls, patch_tokens, padding_mask).
The `StudentBackboneAdapter` translates:
  cls          → global           (B, D4)
  patch_tokens → F4[:, t_idx]     (B, N4, D4)  — selected frame

Head registry
-------------
Built from TaskConfig + anatomy family:
  SEGMENTATION    → UPerNetDecoder  (hierarchical, all 4 scales)
  BINARY_CLS      → MLPClsHead
  MULTICLASS_CLS  → MLPClsHead
  MULTILABEL_CLS  → MLPClsHead (sigmoid in loss)
  REGRESSION      → RegressionHead
  MEASUREMENT     → MeasurementHead
  PATIENT_CLS     → MLPClsHead + temporal pool
  SEQUENCE_CLS    → TemporalConceptHead
  DETECTION       → ConceptDetectionHead (presence logits per concept)

All losses are computed in forward() when labels are provided.
Inference mode: forward(x, padding_mask) → {head_name: logits/masks}

Usage
-----
  model = StudentFinetuneModel.from_config(cfg, backbone_checkpoint)
  output = model(batch)
  loss   = output["loss"]
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from data.labels.label_spec import TaskType, LossType
from models.heads.classification_head import MLPClsHead, AttentivePoolClsHead, LinearClsHead
from models.heads.mil_head import PatientMILClsHead, MIL_HIDDEN_DIM
from models.heads.concept_detection_head import ConceptDetectionHead
from models.heads.regression_head import RegressionHead, MeasurementHead
from models.heads.hierarchical_seg import UPerNetDecoder, build_hierarchical_seg_head

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backbone adapter: translate student dict → legacy head contract
# ---------------------------------------------------------------------------

class StudentBackboneAdapter(nn.Module):
    """
    Thin adapter between HieraStudentBackbone outputs and legacy head inputs.

    Exposes:
      cls(features, t=0)           → (B, D4) from global token
      patch_tokens(features, t=0)  → (B, N4, D4) from F4[:, t]
      dense_tokens(features, t=0)  → (B, N1, D1) from F1[:, t]  (highest res)

    Also provides a per-scale projection if a head's embed_dim != D4.
    The optional per-scale projections are 1×1 linears, initialised as identity
    (they zero-init if the dims differ).
    """

    def __init__(self, embed_dims: List[int]):
        super().__init__()
        self.embed_dims = embed_dims   # [D1, D2, D3, D4]
        self.D4 = embed_dims[-1]
        self.D1 = embed_dims[0]

    def cls(self, features: dict) -> Tensor:
        """(B, D4) — global semantic token."""
        return features["global"]

    def patch_tokens(self, features: dict, t_idx: int = 0) -> Tensor:
        """(B, N4, D4) — stage-4 tokens for frame t_idx."""
        f4 = features["F4"]   # (B, T, N4, D4)
        t = min(t_idx, f4.shape[1] - 1)
        return f4[:, t]

    def dense_tokens(self, features: dict, t_idx: int = 0) -> Tensor:
        """(B, N1, D1) — stage-1 tokens (highest resolution) for frame t_idx."""
        f1 = features["F1"]
        t = min(t_idx, f1.shape[1] - 1)
        return f1[:, t]

    def temporal_tokens(self, features: dict) -> Tensor:
        """(B, T, D4) — per-frame global tokens from F4 mean pooling."""
        f4 = features["F4"]   # (B, T, N4, D4)
        return f4.mean(dim=2)  # (B, T, D4)

    def padding_mask_at_scale(
        self,
        features: dict,
        scale: int = 4,   # 1=stride-4, 2=stride-8, 3=stride-16, 4=stride-32
    ) -> Optional[Tensor]:
        """(B, ph_i, pw_i) padding mask at scale i (1-indexed)."""
        pmasks = features.get("pmasks", [None, None, None, None])
        return pmasks[min(scale - 1, 3)]


# ---------------------------------------------------------------------------
# Per-task head wrappers (normalise loss computation per task type)
# ---------------------------------------------------------------------------

class _SegHead(nn.Module):
    """UPerNet segmentation — handles BCE + Dice combo."""

    def __init__(self, decoder: UPerNetDecoder, n_classes: int, use_dice: bool = True):
        super().__init__()
        self.decoder   = decoder
        self.n_classes = n_classes
        self.use_dice  = use_dice

    def forward(
        self,
        features: dict,
        padding_mask: Optional[Tensor] = None,
        frame_indices: Optional[List[int]] = None,
    ) -> Tensor:
        return self.decoder(features, padding_mask=padding_mask,
                            frame_indices=frame_indices)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float().expand_as(logits)
        )
        if self.use_dice:
            p = torch.sigmoid(logits).flatten(2)
            t = targets.float().flatten(2)
            inter = (p * t).sum(2)
            union = p.sum(2) + t.sum(2)
            dice = 1.0 - (2 * inter + 1) / (union + 1)
            return bce + dice.mean()
        return bce


class _ClsHead(nn.Module):
    """Wraps MLPClsHead / LinearClsHead with loss routing."""

    def __init__(self, head: nn.Module, task_type: TaskType, n_classes: int):
        super().__init__()
        self.head      = head
        self.task_type = task_type
        self.n_classes = n_classes

    def forward(self, cls_token: Tensor) -> Tensor:
        return self.head(cls_token)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.task_type == TaskType.MULTILABEL_CLS:
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        elif self.task_type == TaskType.BINARY_CLS:
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float())
        else:
            return F.cross_entropy(logits, targets.long())


class _AttentiveClsHead(nn.Module):
    """Attentive-pool classification head (for localised findings)."""

    def __init__(self, head: AttentivePoolClsHead, task_type: TaskType):
        super().__init__()
        self.head      = head
        self.task_type = task_type

    def forward(self, patch_tokens: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.head(patch_tokens, padding_mask=padding_mask)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.task_type == TaskType.MULTILABEL_CLS:
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        elif self.task_type == TaskType.BINARY_CLS:
            return F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets.float())
        return F.cross_entropy(logits, targets.long())


class _ConceptHead(nn.Module):
    """ConceptDetectionHead with BCE loss."""

    def __init__(self, head: ConceptDetectionHead):
        super().__init__()
        self.head = head

    def forward(self, patch_tokens: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.head(patch_tokens, padding_mask=padding_mask)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets.float())


class _RegHead(nn.Module):
    """RegressionHead with MSE or MAE loss."""

    def __init__(self, head: nn.Module, loss_type: LossType = LossType.MSE):
        super().__init__()
        self.head      = head
        self.loss_type = loss_type

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)

    def loss(self, pred: Tensor, targets: Tensor) -> Tensor:
        if self.loss_type == LossType.MSE:
            return F.mse_loss(pred, targets.float())
        return F.l1_loss(pred, targets.float())


class _TemporalClsHead(nn.Module):
    """MLPClsHead over temporally pooled features for patient/sequence CLS."""

    def __init__(self, head: MLPClsHead, task_type: TaskType):
        super().__init__()
        self.head      = head
        self.task_type = task_type

    def forward(self, temporal_tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # temporal_tokens: (B, T, D)
        if mask is not None:
            weights = mask.float().unsqueeze(-1)
            pooled = (temporal_tokens * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        else:
            pooled = temporal_tokens.mean(1)
        return self.head(pooled)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        if self.task_type in (TaskType.BINARY_CLS, TaskType.PATIENT_CLS):
            if logits.dim() > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return F.binary_cross_entropy_with_logits(logits, targets.float())
        return F.cross_entropy(logits, targets.long())


class _MILClsHead(nn.Module):
    """Gated attention MIL pool + MLP for patient-level binary classification."""

    def __init__(
        self,
        embed_dim: int,
        task_type: TaskType,
        mil_hidden_dim: int = MIL_HIDDEN_DIM,
        n_classes: int = 1,
    ):
        super().__init__()
        self.head = PatientMILClsHead(
            embed_dim=embed_dim,
            hidden_dim=mil_hidden_dim,
            n_classes=n_classes,
        )
        self.task_type = task_type

    def forward(self, temporal_tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.head(temporal_tokens, mask=mask)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.head.loss(logits, targets)


# ---------------------------------------------------------------------------
# Head factory
# ---------------------------------------------------------------------------

def _build_head_for_task(
    task_type: TaskType,
    embed_dims: List[int],
    n_classes: int,
    class_names: Optional[List[str]] = None,
    loss_type: LossType = LossType.BCE,
    head_style: str = "mlp",         # "mlp" | "linear" | "attentive" | "concept"
    aggregation: str = "mean",       # "mean" | "mil"
) -> nn.Module:
    """
    Build and return the appropriate wrapped head for a task type.

    Parameters
    ----------
    task_type   : TaskType from label spec
    embed_dims  : [D1, D2, D3, D4] from backbone
    n_classes   : output dimension
    class_names : concept names (for ConceptDetectionHead logging)
    loss_type   : primary loss type
    head_style  : preferred head architecture variant
    """
    D4 = embed_dims[-1]
    D1 = embed_dims[0]
    hidden = D4 // 2

    if task_type == TaskType.SEGMENTATION:
        decoder = build_hierarchical_seg_head(
            embed_dims=embed_dims,
            n_classes=max(1, n_classes),
            fpn_channels=256,
        )
        return _SegHead(decoder, n_classes=n_classes, use_dice=(loss_type == LossType.DICE))

    elif task_type in (TaskType.BINARY_CLS, TaskType.MULTICLASS_CLS, TaskType.MULTILABEL_CLS):
        if head_style == "attentive":
            inner = AttentivePoolClsHead(embed_dim=D4, n_classes=n_classes)
            return _AttentiveClsHead(inner, task_type)
        elif head_style == "linear":
            inner = LinearClsHead(embed_dim=D4, n_classes=n_classes)
            return _ClsHead(inner, task_type, n_classes)
        else:
            inner = MLPClsHead(embed_dim=D4, n_classes=n_classes, hidden_dim=hidden)
            return _ClsHead(inner, task_type, n_classes)

    elif task_type in (TaskType.REGRESSION, TaskType.MEASUREMENT):
        if task_type == TaskType.MEASUREMENT:
            inner = MeasurementHead(embed_dim=D4, hidden_dim=hidden)
        else:
            inner = RegressionHead(embed_dim=D4, hidden_dim=hidden)
        return _RegHead(inner, loss_type)

    elif task_type in (TaskType.PATIENT_CLS, TaskType.SEQUENCE_CLS):
        if aggregation == "mil":
            return _MILClsHead(
                embed_dim=D4,
                task_type=task_type,
                n_classes=n_classes,
            )
        inner = MLPClsHead(embed_dim=D4, n_classes=n_classes, hidden_dim=hidden)
        return _TemporalClsHead(inner, task_type)

    elif task_type == TaskType.DETECTION:
        concept_names = class_names or [f"concept_{i}" for i in range(n_classes)]
        inner = ConceptDetectionHead(
            embed_dim=D4, n_concepts=n_classes, concept_names=concept_names
        )
        return _ConceptHead(inner)

    else:
        # CLIP and SSL_ONLY tasks don't need a supervised head
        log.warning(f"No supervised head built for task_type={task_type!r} (unsupervised/CLIP)")
        return None


# ---------------------------------------------------------------------------
# Head spec — describes one task head
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field as dc_field


@dataclass
class HeadSpec:
    """
    Configuration for one downstream task head.

    Fields
    ------
    name        : unique name for this head (used as key in output dict)
    task_type   : TaskType from label spec
    n_classes   : number of output classes
    class_names : optional list of class names
    loss_type   : primary loss type
    loss_weight : contribution to total loss (default 1.0)
    head_style  : "mlp" | "linear" | "attentive" | "concept"
    aggregation : "mean" | "mil"  (patient-level only)
    batch_key   : which key in the batch dict to read labels from
                  (defaults to f"{name}_labels")
    anatomy     : anatomy family tag (used for logging/routing)
    """
    name:        str
    task_type:   TaskType
    n_classes:   int
    class_names: List[str]                  = dc_field(default_factory=list)
    loss_type:   LossType                   = LossType.BCE
    loss_weight: float                      = 1.0
    head_style:  str                        = "mlp"
    aggregation: str                        = "mean"
    batch_key:   Optional[str]              = None
    anatomy:     str                        = "other"

    @property
    def label_key(self) -> str:
        return self.batch_key or f"{self.name}_labels"


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class StudentFinetuneModel(nn.Module):
    """
    Student backbone + all downstream task heads.

    The backbone is always present and optionally frozen.
    Each task head is registered with HeadSpec and built lazily.

    forward(batch) → {
        "loss"              : scalar total loss (if any labels present)
        "<head_name>"       : head output tensor
        "loss_<head_name>"  : per-head loss scalar
        "features"          : raw backbone output dict (for probing / debug)
    }

    Parameters
    ----------
    backbone      : HieraStudentBackbone (or any compatible backbone)
    head_specs    : list of HeadSpec defining which heads to attach
    backbone_frozen : whether to stop gradients through the backbone
    anchor_frame  : which temporal frame index to use for image-like heads (default 0)
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_specs: List[HeadSpec],
        backbone_frozen: bool = False,
        anchor_frame: int = 0,
    ):
        super().__init__()
        self.backbone        = backbone
        self.backbone_frozen = backbone_frozen
        self.anchor_frame    = anchor_frame

        # Derive embed_dims from backbone attribute
        embed_dims: List[int] = getattr(backbone, "embed_dims", [768, 768, 768, 768])
        self.embed_dims       = embed_dims
        self.adapter          = StudentBackboneAdapter(embed_dims)

        # Build and register all heads
        self.heads = nn.ModuleDict()
        self.head_specs: Dict[str, HeadSpec] = {}
        for spec in head_specs:
            head = _build_head_for_task(
                task_type   = spec.task_type,
                embed_dims  = embed_dims,
                n_classes   = spec.n_classes,
                class_names = spec.class_names or None,
                loss_type   = spec.loss_type,
                head_style  = spec.head_style,
                aggregation = spec.aggregation,
            )
            if head is not None:
                self.heads[spec.name] = head
                self.head_specs[spec.name] = spec
                log.info(
                    f"  Head '{spec.name}': {spec.task_type.value} "
                    f"({spec.n_classes} classes, {spec.head_style}, w={spec.loss_weight})"
                )

    # ------------------------------------------------------------------
    # Backbone forward
    # ------------------------------------------------------------------

    def _encode(self, pixel_values: Tensor, padding_mask: Optional[Tensor] = None) -> dict:
        if self.backbone_frozen:
            with torch.no_grad():
                return self.backbone(pixel_values, padding_mask=padding_mask)
        return self.backbone(pixel_values, padding_mask=padding_mask)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        batch : dict containing at minimum:
          "pixel_values" : (B, T, 3, H, W)  (T=1 for images)
                           or "global_crops" (B, N_crops, 3, H, W) from ImageSSLDataset
          "padding_mask" : (B, ph, pw) optional
          "<name>_labels": task-specific labels for each registered head

        Returns
        -------
        dict with "loss", per-head logits, and "features"
        """
        # Resolve pixel values — support both image + video batch formats
        if "pixel_values" in batch:
            x = batch["pixel_values"]
        elif "global_crops" in batch:
            crops = batch["global_crops"]   # (B, N, 3, H, W)
            x = crops[:, :1]               # use first global crop, keep T dim
        elif "full_clips" in batch:
            x = batch["full_clips"]
        else:
            raise KeyError("batch must have 'pixel_values', 'global_crops', or 'full_clips'")

        if x.dim() == 4:
            x = x.unsqueeze(1)   # (B, 3, H, W) → (B, 1, 3, H, W)

        # Resolve padding masks — prefer stride-4 granularity for Hiera
        pmask = (
            batch.get("padding_mask")
            or batch.get("padding_masks")
            or _first_pmask(batch, "global_pmasks")
        )

        # Backbone
        features = self._encode(x, padding_mask=pmask)

        # Shared representations
        cls_token    = self.adapter.cls(features)           # (B, D4)
        patch_tokens = self.adapter.patch_tokens(features, t_idx=self.anchor_frame)
        pm_stage4    = self.adapter.padding_mask_at_scale(features, scale=4)

        out: Dict[str, Any] = {"features": features}
        total_loss = cls_token.new_tensor(0.0)
        has_loss   = False

        for head_name, head in self.heads.items():
            spec = self.head_specs[head_name]

            # ── Run head forward ──────────────────────────────────────
            if spec.task_type == TaskType.SEGMENTATION:
                logits = head(
                    features,
                    padding_mask=pm_stage4,
                    frame_indices=_frame_indices(x),
                )

            elif spec.task_type in (TaskType.PATIENT_CLS, TaskType.SEQUENCE_CLS):
                temporal_tok = self.adapter.temporal_tokens(features)  # (B, T, D4)
                logits = head(temporal_tok, mask=batch.get("frame_mask"))

            elif spec.task_type == TaskType.DETECTION:
                logits = head(patch_tokens, padding_mask=pm_stage4)

            elif spec.head_style == "attentive":
                logits = head(patch_tokens, padding_mask=pm_stage4)

            elif spec.task_type in (TaskType.REGRESSION, TaskType.MEASUREMENT):
                if spec.task_type == TaskType.MEASUREMENT:
                    logits = head(patch_tokens)
                else:
                    logits = head(cls_token)

            else:
                logits = head(cls_token)

            out[head_name] = logits

            # ── Compute loss if labels present ────────────────────────
            labels = batch.get(spec.label_key)
            if labels is not None:
                labels = labels.to(logits.device)
                head_loss = head.loss(logits, labels)
                out[f"loss_{head_name}"] = head_loss.item()
                total_loss = total_loss + spec.loss_weight * head_loss
                has_loss   = True

        if has_loss:
            out["loss"] = total_loss

        return out

    # ------------------------------------------------------------------
    # Convenience: load backbone from checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        head_specs: List[HeadSpec],
        hiera_variant: str = "hiera_large_video_mae_k400",
        backbone_frozen: bool = False,
        device: str = "cuda",
    ) -> "StudentFinetuneModel":
        from models.student.student_config import StudentModelConfig, build_student_encoder
        cfg = StudentModelConfig(hiera_variant=hiera_variant)
        backbone = build_student_encoder(cfg, device=device)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("student_state_dict") or ckpt.get("state_dict") or ckpt
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            log.warning(f"Missing keys in checkpoint: {len(missing)}")
        if unexpected:
            log.warning(f"Unexpected keys in checkpoint: {len(unexpected)}")

        return cls(backbone, head_specs, backbone_frozen=backbone_frozen)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_pmask(batch: dict, key: str) -> Optional[Tensor]:
    pm = batch.get(key)
    if pm is None:
        return None
    if pm.dim() == 4:   # (B, N_crops, ph, pw)
        return pm[:, 0]
    return pm


def _frame_indices(x: Tensor) -> Optional[List[int]]:
    T = x.shape[1]
    return list(range(T))


# ---------------------------------------------------------------------------
# Canonical head spec builders for each anatomy + task
# ---------------------------------------------------------------------------

def cardiac_head_specs(embed_dims: List[int]) -> List[HeadSpec]:
    """Standard head suite for cardiac echo datasets (CAMUS, EchoNet, MIMIC)."""
    return [
        HeadSpec(
            name="cardiac_seg",
            task_type=TaskType.SEGMENTATION,
            n_classes=1,
            class_names=["left_ventricle"],
            loss_type=LossType.DICE,
            loss_weight=1.0,
            batch_key="seg_masks",
            anatomy="cardiac",
        ),
        HeadSpec(
            name="ef_regression",
            task_type=TaskType.REGRESSION,
            n_classes=1,
            class_names=["ejection_fraction"],
            loss_type=LossType.MSE,
            loss_weight=0.5,
            batch_key="ef_labels",
            anatomy="cardiac",
        ),
        HeadSpec(
            name="view_cls",
            task_type=TaskType.MULTICLASS_CLS,
            n_classes=4,
            class_names=["2CH", "4CH", "PLAX", "PSAX"],
            loss_type=LossType.CE,
            loss_weight=0.5,
            batch_key="view_labels",
            anatomy="cardiac",
        ),
    ]


def lung_head_specs(embed_dims: List[int]) -> List[HeadSpec]:
    """LUS / COVID-US classification + B-lines concept detection + patient TB."""
    return [
        HeadSpec(
            name="lus_patient_tb",
            task_type=TaskType.PATIENT_CLS,
            n_classes=1,
            class_names=["tb"],
            loss_type=LossType.BCE,
            loss_weight=1.0,
            aggregation="mil",
            batch_key="lus_patient_tb_labels",
            anatomy="lung",
        ),
        HeadSpec(
            name="lung_cls",
            task_type=TaskType.MULTICLASS_CLS,
            n_classes=3,
            class_names=["normal", "covid", "pneumonia"],
            loss_type=LossType.CE,
            loss_weight=1.0,
            batch_key="lung_cls_labels",
            anatomy="lung",
        ),
        HeadSpec(
            name="bline_cls",
            task_type=TaskType.BINARY_CLS,
            n_classes=1,
            class_names=["b_lines"],
            loss_type=LossType.BCE,
            loss_weight=0.5,
            batch_key="bline_labels",
            anatomy="lung",
        ),
    ]


def breast_head_specs(embed_dims: List[int]) -> List[HeadSpec]:
    """BUSI / BUS-BRA: segmentation + BI-RADS classification."""
    return [
        HeadSpec(
            name="breast_seg",
            task_type=TaskType.SEGMENTATION,
            n_classes=1,
            class_names=["mass"],
            loss_type=LossType.DICE,
            loss_weight=1.0,
            batch_key="seg_masks",
            anatomy="breast",
        ),
        HeadSpec(
            name="birads_cls",
            task_type=TaskType.MULTICLASS_CLS,
            n_classes=3,
            class_names=["benign", "malignant", "normal"],
            loss_type=LossType.CE,
            loss_weight=0.5,
            batch_key="birads_labels",
            anatomy="breast",
            head_style="attentive",
        ),
    ]


def thyroid_head_specs(embed_dims: List[int]) -> List[HeadSpec]:
    """TN3K / DDTI: nodule segmentation + TIRADS ordinal classification."""
    return [
        HeadSpec(
            name="thyroid_seg",
            task_type=TaskType.SEGMENTATION,
            n_classes=1,
            class_names=["nodule"],
            loss_type=LossType.DICE,
            loss_weight=1.0,
            batch_key="seg_masks",
            anatomy="thyroid",
        ),
        HeadSpec(
            name="tirads_cls",
            task_type=TaskType.MULTICLASS_CLS,
            n_classes=5,
            class_names=["tirads_1", "tirads_2", "tirads_3", "tirads_4", "tirads_5"],
            loss_type=LossType.CE,
            loss_weight=0.5,
            batch_key="tirads_labels",
            anatomy="thyroid",
        ),
    ]


def fetal_head_specs(embed_dims: List[int]) -> List[HeadSpec]:
    """HC18 / ACOUSLIC: plane classification + HC measurement."""
    return [
        HeadSpec(
            name="fetal_plane_cls",
            task_type=TaskType.MULTICLASS_CLS,
            n_classes=6,
            class_names=["brain", "abdomen", "femur", "thorax", "cervix", "other"],
            loss_type=LossType.CE,
            loss_weight=1.0,
            batch_key="plane_labels",
            anatomy="fetal",
        ),
        HeadSpec(
            name="hc_measurement",
            task_type=TaskType.MEASUREMENT,
            n_classes=1,
            class_names=["head_circumference_mm"],
            loss_type=LossType.MSE,
            loss_weight=0.5,
            batch_key="hc_labels",
            anatomy="fetal",
        ),
    ]


def all_head_specs_for_anatomy(anatomy: str, embed_dims: List[int]) -> List[HeadSpec]:
    """Return canonical head specs for a given anatomy family."""
    registry = {
        "cardiac":  cardiac_head_specs,
        "lung":     lung_head_specs,
        "breast":   breast_head_specs,
        "thyroid":  thyroid_head_specs,
        "fetal":    fetal_head_specs,
    }
    builder = registry.get(anatomy)
    if builder is None:
        log.warning(f"No canonical head specs for anatomy={anatomy!r}")
        return []
    return builder(embed_dims)
