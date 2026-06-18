"""
finetune/backbones/dinov3_encoder.py  ·  DINOv3 image encoder
==============================================================

Wraps registered DINOv3 ViT variants (via models.registry) for the finetune
comparison suite.  Default variant: dinov3_b (ViT-B/16, D=768).

Video tasks use per-frame CLS embeddings + TemporalAttentionPool (frame-based,
not video-native — same pattern as torchvision ViT / BioMed-CLIP).
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)

_VARIANT_DIMS = {
    "dinov3_s":     384,
    "dinov3_splus": 384,
    "dinov3_b":     768,
    "dinov3_l":     1024,
    "dinov3_hplus": 1280,
}
_IMG_SIZE = 224


class DINOv3Encoder(BackboneEncoder):
    """
    DINOv3 ViT image encoder for backbone comparison.

    Outputs
    -------
    cls          : (B, D)     CLS token
    patch_tokens : (B, N, D)  spatial patch tokens (224×224 → N=196)
    """

    def __init__(
        self,
        variant:          str = "dinov3_b",
        temporal_dropout: float = 0.0,
        hf_cache_dir:     Optional[str] = None,
    ):
        from models.registry import build_image_backbone
        from models.hf_loading import resolve_dinov3_variant

        if variant not in _VARIANT_DIMS:
            raise ValueError(
                f"Unknown DINOv3 variant {variant!r}. "
                f"Valid: {sorted(_VARIANT_DIMS)}"
            )

        variant = resolve_dinov3_variant(variant, hf_cache_dir)
        self._variant = variant
        self._d        = _VARIANT_DIMS[variant]

        log.info("[%s] Loading DINOv3 backbone", variant)
        self.backbone = build_image_backbone(
            variant, dtype=torch.float32, hf_cache_dir=hf_cache_dir,
        )
        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        log.info("[%s] Ready (embed_dim=%d)", variant, self._d)

    @property
    def name(self) -> str:
        return self._variant

    @property
    def embed_dim(self) -> int:
        return self._d

    @staticmethod
    def _resize_images(images: Tensor, size: int = _IMG_SIZE) -> Tensor:
        if images.shape[-1] == size and images.shape[-2] == size:
            return images
        return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)

    def encode_image(self, images: Tensor) -> dict:
        dev = next(self.backbone.parameters()).device
        images = self._resize_images(images.to(dev))
        with torch.no_grad():
            out = self.backbone(images)
        return {"cls": out["cls"], "patch_tokens": out["patch_tokens"]}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames = clips.reshape(B * T, C, H, W)
        enc    = self.encode_image(frames)
        frame_tokens = enc["cls"].reshape(B, T, -1)
        clip_cls     = self.temporal_pool(frame_tokens)
        return {"clip_cls": clip_cls, "frame_tokens": frame_tokens, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.temporal_pool.parameters()

    def to(self, *args, **kwargs) -> "DINOv3Encoder":
        self.backbone.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "DINOv3Encoder":
        self.backbone.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "DINOv3Encoder":
        self.backbone.eval()
        self.temporal_pool.train(mode)
        return self
