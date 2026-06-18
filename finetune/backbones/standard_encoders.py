"""
finetune/backbones/standard_encoders.py  ·  ResNet50 and ViT-B/16 encoders
===========================================================================

Torchvision-based image encoders for the backbone comparison suite.
Both attach a TemporalAttentionPool for video task compatibility.

ResNet50Encoder
    Global average pool output as cls (D=2048).
    C5 feature map reshaped to (B, N, 2048) as patch_tokens (N=49 for 224px).

ViTEncoder
    CLS token as cls (D=768).
    Spatial patch tokens as patch_tokens (N=196 for 224px).
"""
from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)


class ResNet50Encoder(BackboneEncoder):
    """
    ResNet-50 (ImageNet-1k pretrained) image encoder.

    Outputs
    -------
    cls          : (B, 2048)     global average pool
    patch_tokens : (B, 49, 2048) C5 spatial feature map flattened
    """

    def __init__(self, pretrained: bool = True, temporal_dropout: float = 0.0):
        import torchvision.models as tv

        weights = tv.ResNet50_Weights.DEFAULT if pretrained else None
        model   = tv.resnet50(weights=weights)

        # Strip final FC; keep convolutional body through layer4 (C5)
        self.backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
        )
        self._d = 2048
        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    @property
    def name(self) -> str:
        return "resnet50"

    @property
    def embed_dim(self) -> int:
        return self._d

    def encode_image(self, images: Tensor) -> dict:
        with torch.no_grad():
            feat = self.backbone(images)                           # (B, 2048, H, W)
        cls          = feat.mean(dim=(2, 3))                       # (B, 2048)
        B, C, H, W   = feat.shape
        patch_tokens = feat.reshape(B, C, H * W).permute(0, 2, 1) # (B, N, 2048)
        return {"cls": cls.detach(), "patch_tokens": patch_tokens.detach()}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames = clips.reshape(B * T, C, H, W)
        with torch.no_grad():
            feat = self.backbone(frames)                           # (BT, 2048, h, w)
        cls_per_frame = feat.mean(dim=(2, 3))                      # (BT, 2048)
        frame_tokens  = cls_per_frame.reshape(B, T, -1)            # (B, T, 2048)
        clip_cls      = self.temporal_pool(frame_tokens)           # (B, 2048)
        return {"clip_cls": clip_cls, "frame_tokens": frame_tokens, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.temporal_pool.parameters()

    def to(self, *args, **kwargs) -> "ResNet50Encoder":
        self.backbone.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "ResNet50Encoder":
        self.backbone.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "ResNet50Encoder":
        self.backbone.eval()                         # backbone stays frozen
        self.temporal_pool.train(mode)
        return self


class ViTEncoder(BackboneEncoder):
    """
    ViT-B/16 (ImageNet-21k pretrained) image encoder.

    Outputs
    -------
    cls          : (B, 768)     CLS token
    patch_tokens : (B, 196, 768) spatial patch tokens (224px input)
    """

    def __init__(self, pretrained: bool = True, temporal_dropout: float = 0.0):
        import torchvision.models as tv

        weights = tv.ViT_B_16_Weights.DEFAULT if pretrained else None
        model   = tv.vit_b_16(weights=weights)

        self._d  = 768
        self.vit = model
        # Remove classification head; we extract CLS + patch tokens directly
        self.vit.heads = nn.Identity()

        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.vit.eval()

    @property
    def name(self) -> str:
        return "vit_b_16"

    @property
    def embed_dim(self) -> int:
        return self._d

    def _forward_vit(self, images: Tensor):
        """Run torchvision ViT encoder, return (cls, patch_tokens)."""
        with torch.no_grad():
            x = self.vit._process_input(images)                         # (B, N, D)
            n = x.shape[0]
            batch_cls = self.vit.class_token.expand(n, -1, -1)
            x = torch.cat([batch_cls, x], dim=1)                        # (B, N+1, D)
            x = self.vit.encoder(x)                                     # (B, N+1, D)
        cls_token    = x[:, 0]                                          # (B, D)
        patch_tokens = x[:, 1:]                                         # (B, N, D)
        return cls_token, patch_tokens

    def encode_image(self, images: Tensor) -> dict:
        if images.shape[-1] != 224 or images.shape[-2] != 224:
            images = F.interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False,
            )
        cls, patch = self._forward_vit(images)
        return {"cls": cls, "patch_tokens": patch}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames   = clips.reshape(B * T, C, H, W)
        if frames.shape[-1] != 224 or frames.shape[-2] != 224:
            frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
        cls_bt, _ = self._forward_vit(frames)                          # (BT, D)
        frame_tokens = cls_bt.reshape(B, T, -1)                        # (B, T, D)
        clip_cls     = self.temporal_pool(frame_tokens)                 # (B, D)
        return {"clip_cls": clip_cls, "frame_tokens": frame_tokens, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.temporal_pool.parameters()

    def to(self, *args, **kwargs) -> "ViTEncoder":
        self.vit.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "ViTEncoder":
        self.vit.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "ViTEncoder":
        self.vit.eval()
        self.temporal_pool.train(mode)
        return self
