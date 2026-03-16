"""
models/registry.py  ·  Backbone registry
=========================================

A central dictionary mapping string keys to backbone factory functions.
Backbones register themselves with @register_image_backbone or
@register_video_backbone.  The model config uses these keys to select
which backbone to instantiate.

Registry keys (image)
---------------------
  "dinov3_s"          DINOv3-Small    (21M)
  "dinov3_splus"      DINOv3-Small+   (21M, wider MLP)
  "dinov3_b"          DINOv3-Base     (86M)
  "dinov3_l"          DINOv3-Large    (307M)   ← default
  "dinov3_hplus"      DINOv3-Huge+    (632M)
  "rad_dino"          RadDINO         (DINOv2-B fine-tuned on radiology)
  "swin_v2_l"         SwinTransformerV2-Large  (stub, future)

Registry keys (video)
---------------------
  "vjepa2_l"          V-JEPA2-Large   fpc64-256   ← default
  "vjepa2_h"          V-JEPA2-Huge    fpc64-256
  "vjepa2_g"          V-JEPA2-Giant   fpc64-256

Registry keys (frozen teacher)
-------------------------------
  "dinov3_7b"         DINOv3-7B       (permanently frozen)

Each factory has signature:
  factory(dtype, hf_cache_dir) → BackboneInstance

The caller (build_image_branch / build_video_branch in model_config.py)
is responsible for .to(device, dtype) after construction.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional
import torch

# Registry dicts: key → factory function
_IMAGE_REGISTRY:   Dict[str, Callable] = {}
_VIDEO_REGISTRY:   Dict[str, Callable] = {}
_TEACHER_REGISTRY: Dict[str, Callable] = {}


def register_image_backbone(key: str):
    """Decorator: @register_image_backbone("dinov3_l")"""
    def decorator(fn: Callable) -> Callable:
        if key in _IMAGE_REGISTRY:
            raise ValueError(f"Image backbone already registered: '{key}'")
        _IMAGE_REGISTRY[key] = fn
        return fn
    return decorator


def register_video_backbone(key: str):
    def decorator(fn: Callable) -> Callable:
        if key in _VIDEO_REGISTRY:
            raise ValueError(f"Video backbone already registered: '{key}'")
        _VIDEO_REGISTRY[key] = fn
        return fn
    return decorator


def register_frozen_teacher(key: str):
    def decorator(fn: Callable) -> Callable:
        if key in _TEACHER_REGISTRY:
            raise ValueError(f"Frozen teacher already registered: '{key}'")
        _TEACHER_REGISTRY[key] = fn
        return fn
    return decorator


def build_image_backbone(
    key: str,
    dtype: torch.dtype = torch.bfloat16,
    hf_cache_dir: Optional[str] = None,
):
    """Instantiate a registered image backbone by key."""
    if key not in _IMAGE_REGISTRY:
        available = sorted(_IMAGE_REGISTRY.keys())
        raise KeyError(
            f"Unknown image backbone '{key}'. "
            f"Available: {available}"
        )
    return _IMAGE_REGISTRY[key](dtype=dtype, hf_cache_dir=hf_cache_dir)


def build_video_backbone(
    key: str,
    dtype: torch.dtype = torch.bfloat16,
    hf_cache_dir: Optional[str] = None,
):
    if key not in _VIDEO_REGISTRY:
        available = sorted(_VIDEO_REGISTRY.keys())
        raise KeyError(
            f"Unknown video backbone '{key}'. "
            f"Available: {available}"
        )
    return _VIDEO_REGISTRY[key](dtype=dtype, hf_cache_dir=hf_cache_dir)


def build_frozen_teacher(
    key: str,
    dtype: torch.dtype = torch.bfloat16,
    hf_cache_dir: Optional[str] = None,
    device: str = "cuda",
):
    if key not in _TEACHER_REGISTRY:
        available = sorted(_TEACHER_REGISTRY.keys())
        raise KeyError(
            f"Unknown frozen teacher '{key}'. "
            f"Available: {available}"
        )
    return _TEACHER_REGISTRY[key](dtype=dtype, hf_cache_dir=hf_cache_dir, device=device)


def list_image_backbones() -> list[str]:
    return sorted(_IMAGE_REGISTRY.keys())


def list_video_backbones() -> list[str]:
    return sorted(_VIDEO_REGISTRY.keys())


def list_frozen_teachers() -> list[str]:
    return sorted(_TEACHER_REGISTRY.keys())
