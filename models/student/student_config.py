"""
models/student/student_config.py  ·  StudentModelConfig + factory
=================================================================

StudentModelConfig
------------------
Dataclass specifying the full Hiera student encoder configuration.
Loaded from the YAML ``student:`` section.

Example YAML:

    student:
      hiera_variant:            hiera_large_video_mae_k400
      hiera_hf_cache_dir:       null
      temporal_mixing:          factorized_attn
      temporal_adapter_stages:  [2, 3]
      max_frames:               32
      trainable_stages:         null
      align_dim:                1024
      n_prototypes:             256
      ema_momentum:             0.9995
      dino_teacher_key:         dinov3_l
      vjepa_teacher_key:        vjepa2_l
      fusion_radius:            2
      fusion_init_gate:        -3.0
      dtype:                    bfloat16
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
}


@dataclass
class StudentModelConfig:
    """
    Full configuration for the Hiera-based student encoder and its teachers.

    Fields
    ------
    hiera_variant            : "hiera_large_video_mae_k400" | "sam2_hiera_large"
    hiera_hf_cache_dir       : HuggingFace cache dir (inherits global if None)
    temporal_mixing          : "factorized_attn" | "depthwise_conv"
    temporal_adapter_stages  : list of inter-stage positions to inject adapters
    max_frames               : max temporal positions for temporal pos enc
    trainable_stages         : last N Hiera stages to unfreeze (None = all)
    align_dim                : shared projection dim for teachers + prototypes
    n_prototypes             : number of prototype vectors
    ema_momentum             : EMA momentum for live student → EMA-student sync (see loss_weights lam_ema_max)
    dino_teacher_key         : image backbone registry key for DINO teacher
    vjepa_teacher_key        : video backbone registry key for V-JEPA teacher
    fusion_radius            : local neighbourhood radius for cross-attention fusion
    fusion_init_gate         : initial gate logit for LocalGatedCrossAttnFusion
    dtype                    : model dtype for teachers
    """
    hiera_variant:           str        = "hiera_large_video_mae_k400"
    hiera_hf_cache_dir:      Optional[str] = None

    temporal_mixing:         str        = "factorized_attn"
    temporal_adapter_stages: List[int]  = field(default_factory=lambda: [2, 3])
    max_frames:              int        = 32
    trainable_stages:        Optional[int] = None

    align_dim:               int        = 1024
    n_prototypes:            int        = 256
    ema_momentum:            float      = 0.9995

    dino_teacher_key:        str        = "dinov3_l"
    vjepa_teacher_key:       str        = "vjepa2_l"

    fusion_radius:           int        = 2
    fusion_init_gate:        float      = -3.0

    dtype:                   str        = "bfloat16"

    @classmethod
    def from_dict(cls, d: dict) -> "StudentModelConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid}
        return cls(**filtered)

    def torch_dtype(self) -> torch.dtype:
        return _DTYPE_MAP.get(self.dtype, torch.bfloat16)


def build_student_encoder(
    cfg: StudentModelConfig,
    device: str = "cuda",
) -> "HieraStudentBackbone":
    """
    Instantiate and return the HieraStudentBackbone from config.
    Places the model on `device` after construction.
    """
    from .hiera_backbone import HieraStudentBackbone

    log.info(f"Building HieraStudentBackbone: variant={cfg.hiera_variant}")
    model = HieraStudentBackbone(
        hiera_variant           = cfg.hiera_variant,
        temporal_mixing         = cfg.temporal_mixing,
        temporal_adapter_stages = cfg.temporal_adapter_stages,
        max_frames              = cfg.max_frames,
        trainable_stages        = cfg.trainable_stages,
        hf_cache_dir            = cfg.hiera_hf_cache_dir,
        align_dim               = cfg.align_dim,
    )
    model = model.to(device)
    return model


def build_fusion_target_builder(
    cfg: StudentModelConfig,
    device: str = "cuda",
) -> "FusionTargetBuilder":
    """
    Build the FusionTargetBuilder (training-time only).
    Reads teacher hidden dims from their loaded backbones to set d_img / d_vid.
    """
    from models.registry import build_image_backbone, build_video_backbone
    from .cross_attention_fusion import FusionTargetBuilder

    dtype = cfg.torch_dtype()
    img_bb = build_image_backbone(cfg.dino_teacher_key, dtype=dtype, hf_cache_dir=cfg.hiera_hf_cache_dir)
    vid_bb = build_video_backbone(cfg.vjepa_teacher_key, dtype=dtype, hf_cache_dir=cfg.hiera_hf_cache_dir)
    d_img = img_bb.hidden_size
    d_vid = vid_bb.hidden_size
    del img_bb, vid_bb   # free memory; teacher wrappers will reload separately

    builder = FusionTargetBuilder(
        d_img        = d_img,
        d_vid        = d_vid,
        align_dim    = cfg.align_dim,
        num_heads    = max(1, cfg.align_dim // 64),
        fusion_radius = cfg.fusion_radius,
        init_gate    = cfg.fusion_init_gate,
    )
    return builder.to(device)
