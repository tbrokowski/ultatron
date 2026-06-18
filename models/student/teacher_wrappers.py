"""
models/student/teacher_wrappers.py  ·  Frozen teacher wrappers
===============================================================

Wraps the existing registered DINOv3 and V-JEPA2 backbones as frozen
teachers for the single-student pretraining pipeline.

Both teachers:
  - are permanently frozen (no gradients, no EMA updates)
  - reuse the existing backbone registry — no new HuggingFace downloads
  - project their native hidden dimension to a shared align_dim

FrozenDINOTeacher
-----------------
Wraps the registered `dinov3_l` image backbone.
  forward(pixel_values, padding_mask, return_attention=False) →
    {
      "cls"                  : (B, D_dino)          raw teacher CLS token
      "patch_tokens"         : (B, N, D_dino)       raw teacher patch tokens
      "cls_proj"             : (B, align_dim)       projected CLS
      "patch_proj"           : (B, N, align_dim)    projected patch tokens
      "cls_patch_attention"  : (B, N) optional — mean CLS→patch attn (last layer)
    }

FrozenVJEPATeacher
------------------
Wraps the registered `vjepa2_l` video backbone.
  forward(pixel_values, tube_mask, padding_mask, valid_frames) →
    {
      "clip_cls"     : (B, D_vjepa)
      "tube_tokens"  : (B, T*N, D_vjepa)
      "clip_proj"    : (B, align_dim)
      "tube_proj"    : (B, T*N, align_dim)
    }
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


class FrozenDINOTeacher(nn.Module):
    """
    Permanently frozen DINOv3 image teacher.

    Loaded from the existing image backbone registry so no separate
    HuggingFace download is needed.

    Parameters
    ----------
    backbone_key : registry key, default "dinov3_l"
    align_dim    : projected output dimension
    dtype        : model dtype
    hf_cache_dir : HuggingFace cache override
    """

    def __init__(
        self,
        backbone_key: str = "dinov3_l",
        align_dim: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        hf_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        from models.registry import build_image_backbone
        self.backbone_key = backbone_key
        self.align_dim    = align_dim

        log.info(f"Loading FrozenDINOTeacher ({backbone_key}) ...")
        backbone = build_image_backbone(backbone_key, dtype=dtype, hf_cache_dir=hf_cache_dir)
        self.backbone = backbone
        self.d_model  = backbone.hidden_size

        # Freeze all parameters
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Projection: D_dino → align_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, align_dim, bias=False),
        )

        log.info(f"  FrozenDINOTeacher: D={self.d_model} → align_dim={align_dim}")

    def _project(self, tokens: Tensor) -> Tensor:
        return self.proj(tokens.float())

    @torch.no_grad()
    def forward(
        self,
        pixel_values: Tensor,                   # (B, 3, H, W)
        padding_mask: Optional[Tensor] = None,  # (B, ph, pw)
        return_attention: bool = False,
    ) -> dict:
        out = self.backbone(
            pixel_values,
            padding_mask=padding_mask,
            output_attentions=return_attention,
        )
        cls    = out["cls"]             # (B, D)
        patch  = out["patch_tokens"]    # (B, N, D)

        cls_proj   = self._project(cls)
        patch_proj = self._project(patch)

        result = {
            "cls":          cls,
            "patch_tokens": patch,
            "cls_proj":     cls_proj,
            "patch_proj":   patch_proj,
        }
        if "cls_patch_attention" in out:
            result["cls_patch_attention"] = out["cls_patch_attention"]
        return result

    def parameters_for_ema(self):
        # Only the projection head is trainable (backbone is frozen)
        return self.proj.parameters()


class FrozenVJEPATeacher(nn.Module):
    """
    Permanently frozen V-JEPA2 video teacher.

    Parameters
    ----------
    backbone_key : registry key, default "vjepa2_l"
    align_dim    : projected output dimension
    dtype        : model dtype
    hf_cache_dir : HuggingFace cache override
    """

    def __init__(
        self,
        backbone_key: str = "vjepa2_l",
        align_dim: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        hf_cache_dir: Optional[str] = None,
    ):
        super().__init__()
        from models.registry import build_video_backbone
        self.backbone_key = backbone_key
        self.align_dim    = align_dim

        log.info(f"Loading FrozenVJEPATeacher ({backbone_key}) ...")
        backbone = build_video_backbone(backbone_key, dtype=dtype, hf_cache_dir=hf_cache_dir)
        self.backbone = backbone
        self.d_model  = backbone.hidden_size

        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, align_dim, bias=False),
        )

        log.info(f"  FrozenVJEPATeacher: D={self.d_model} → align_dim={align_dim}")

    def _project(self, tokens: Tensor) -> Tensor:
        return self.proj(tokens.float())

    @torch.no_grad()
    def forward(
        self,
        pixel_values: Tensor,                    # (B, T, 3, H, W)
        tube_mask: Optional[Tensor] = None,      # (B, T, ph, pw) bool
        padding_mask: Optional[Tensor] = None,   # (B, ph, pw) bool
        valid_frames: Optional[Tensor] = None,   # (B, T) bool
    ) -> dict:
        out = self.backbone(
            pixel_values,
            tube_mask=tube_mask,
            padding_mask=padding_mask,
            valid_frames=valid_frames,
        )
        clip_cls    = out["clip_cls"]      # (B, D)
        tube_tokens = out["tube_tokens"]   # (B, T*N, D)

        clip_proj = self._project(clip_cls)
        tube_proj = self._project(tube_tokens)

        return {
            "clip_cls":    clip_cls,
            "tube_tokens": tube_tokens,
            "clip_proj":   clip_proj,
            "tube_proj":   tube_proj,
        }

    def parameters_for_ema(self):
        return self.proj.parameters()
