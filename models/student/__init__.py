"""
models/student/  ·  Single-student Hiera-based encoder package
==============================================================

Public API
----------
  HieraStudentBackbone   — main encoder (backbone.py)
  StudentModelConfig     — config dataclass
  build_student_encoder  — factory
  FrozenDINOTeacher      — frozen DINOv3 wrapper
  FrozenVJEPATeacher     — frozen V-JEPA2 wrapper
  FusionTargetBuilder    — Stage-3 local DINO-V-JEPA fusion (training only)
  FactorizedTemporalAttention
  TemporalDepthwiseConv
"""
from .hiera_backbone import HieraStudentBackbone                          # noqa: F401
from .student_config import StudentModelConfig, build_student_encoder     # noqa: F401
from .teacher_wrappers import FrozenDINOTeacher, FrozenVJEPATeacher       # noqa: F401
from .cross_attention_fusion import (                                      # noqa: F401
    FusionTargetBuilder,
    LocalGatedCrossAttentionFusion,
    build_tube_to_patch_map,
    gather_local_patches,
)
from .temporal_mixing import (                                             # noqa: F401
    FactorizedTemporalAttention,
    TemporalDepthwiseConv,
    build_temporal_mixer,
)
from .student_finetune import (                                            # noqa: F401
    StudentFinetuneModel,
    HeadSpec,
    cardiac_head_specs,
    breast_head_specs,
    thyroid_head_specs,
    lung_head_specs,
    fetal_head_specs,
    all_head_specs_for_anatomy,
)

# ── Register backbone variants with the central registry ─────────────────
from models.registry import register_student_backbone                     # noqa: F401


@register_student_backbone("hiera_large_video_mae_k400")
def _build_hiera_video_k400(hf_cache_dir=None, **kwargs):
    return HieraStudentBackbone(
        hiera_variant="hiera_large_video_mae_k400",
        hf_cache_dir=hf_cache_dir,
        **kwargs,
    )


@register_student_backbone("sam2_hiera_large")
def _build_sam2_hiera(hf_cache_dir=None, **kwargs):
    return HieraStudentBackbone(
        hiera_variant="sam2_hiera_large",
        hf_cache_dir=hf_cache_dir,
        **kwargs,
    )
