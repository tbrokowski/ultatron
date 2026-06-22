"""
finetune/backbones/unet_encoder.py  ·  UNet segmentation encoder
=================================================================

Full UNet used as a BackboneEncoder for segmentation ablations.

Unlike frozen-backbone encoders where only a lightweight head is trained,
UNet is an end-to-end encoder+decoder architecture.  The decoder output
is returned as ``patch_tokens`` so existing segmentation heads
(``LinearSegHead``) can produce the final per-pixel logits.

Set ``freeze_backbone: false`` in the finetune config so that all
UNet parameters are trainable.
"""
from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from finetune.backbones.base import BackboneEncoder

log = logging.getLogger(__name__)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _UNetModel(nn.Module):
    """Standard 4-level UNet with skip connections."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = _ConvBlock(in_channels, c)
        self.enc2 = _ConvBlock(c, c * 2)
        self.enc3 = _ConvBlock(c * 2, c * 4)
        self.enc4 = _ConvBlock(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _ConvBlock(c * 8, c * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = _ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = _ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = _ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = _ConvBlock(c * 2, c)

    def forward(self, x: Tensor) -> Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self._match_and_cat(self.up4(b), e4)], dim=1))
        d3 = self.dec3(torch.cat([self._match_and_cat(self.up3(d4), e3)], dim=1))
        d2 = self.dec2(torch.cat([self._match_and_cat(self.up2(d3), e2)], dim=1))
        d1 = self.dec1(torch.cat([self._match_and_cat(self.up1(d2), e1)], dim=1))

        return d1, b

    @staticmethod
    def _match_and_cat(upsampled: Tensor, skip: Tensor) -> Tensor:
        """Pad upsampled tensor to match skip connection spatial dims."""
        dh = skip.shape[2] - upsampled.shape[2]
        dw = skip.shape[3] - upsampled.shape[3]
        if dh != 0 or dw != 0:
            upsampled = F.pad(upsampled, [0, dw, 0, dh])
        return torch.cat([upsampled, skip], dim=1)


class UNetEncoder(BackboneEncoder):
    """
    Full UNet as a BackboneEncoder for segmentation ablations.

    The decoder output is adaptively pooled to a patch grid matching the
    input_size / 16 convention (e.g. 14×14 for 224px input), then returned
    as ``patch_tokens``.  A ``LinearSegHead(base_channels, n_classes)``
    on top produces the final segmentation logits.

    All parameters are trainable — set ``freeze_backbone: false`` in config.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        patch_grid: int = 14,
    ):
        self._model = _UNetModel(in_channels, base_channels)
        self._base_channels = base_channels
        self._patch_grid = patch_grid

    @property
    def name(self) -> str:
        return "unet"

    @property
    def embed_dim(self) -> int:
        return self._base_channels

    def encode_image(self, images: Tensor, **kwargs) -> dict:
        dec_features, bottleneck = self._model(images)

        B, C, H, W = dec_features.shape
        pg = self._patch_grid
        pooled = F.adaptive_avg_pool2d(dec_features, (pg, pg))
        patch_tokens = pooled.reshape(B, C, pg * pg).permute(0, 2, 1)

        cls_token = bottleneck.mean(dim=(2, 3))
        return {"cls": cls_token, "patch_tokens": patch_tokens}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self._model.parameters()

    def to(self, *args, **kwargs) -> "UNetEncoder":
        self._model.to(*args, **kwargs)
        return self

    def eval(self) -> "UNetEncoder":
        self._model.eval()
        return self

    def train(self, mode: bool = True) -> "UNetEncoder":
        self._model.train(mode)
        return self
