"""
finetune/backbones/echocare_encoder.py  ·  EchoCare ultrasound foundation model encoder
========================================================================================

EchoCare is a Swin Transformer pre-trained via masked image modelling on
4.5 million ultrasound images from 23 clinical centres worldwide.

Paper / repo: https://github.com/CAIR-HKISI/EchoCare

Architecture (official README):
    MONAI SwinTransformer (2-D, spatial_dims=2, use_v2=True)
    embed_dim=128, depths=[2,2,18,2], window_size=[8,8], patch_size=[2,2]
    Output: 5 hierarchical feature maps for a 256×256 input:
        [B, 128, 128, 128], [B, 256, 64, 64], [B, 512, 32, 32],
        [B, 1024, 16, 16], [B, 2048, 8, 8]
    cls representation = global-average-pool of the final stage (D=2048)

Checkpoint loading (from official README):
    state_dict = torch.load(checkpoint)
    state_dict.pop('mask_token')
    encoder.load_state_dict(state_dict, strict=True)

Setup:
    pip install monai

Checkpoint path:
    Default: /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/Ablations/echocare_encoder.pth
    Override: US_ECHOCARE_CHECKPOINT=/path/to/echocare_encoder.pth
"""
from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from finetune.backbones.paths import ablation_weight_path
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)

# Architecture config exactly as in the CAIR-HKISI/EchoCare README
_SWIN_CFG = dict(
    in_chans=3,
    embed_dim=128,
    window_size=[8, 8],
    patch_size=[2, 2],
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    mlp_ratio=4.0,
    qkv_bias=True,
    use_checkpoint=False,   # disable grad checkpointing for inference
    spatial_dims=2,
    use_v2=True,
)

# Final-stage channel count: embed_dim × 2^(n_stages) = 128 × 2^4 = 2048
_EMBED_DIM = 2048
# EchoCare Swin was pre-trained on 256×256 inputs (official README).
_IMG_SIZE  = 256


class EchoCareEncoder(BackboneEncoder):
    """
    EchoCare ultrasound foundation model encoder.

    Uses MONAI's SwinTransformer (2-D) with the exact architecture and
    pre-trained weights from CAIR-HKISI/EchoCare.

    Parameters
    ----------
    checkpoint : str
        Path to the EchoCare pre-trained weights (.pth).
    temporal_dropout : float
        Dropout for TemporalAttentionPool used in encode_video().
    """

    def __init__(
        self,
        checkpoint:       str | None = None,
        embed_dim:        int   = _EMBED_DIM,   # kept for API compat; always 2048
        is_video:         bool  = False,         # EchoCare is an image model
        temporal_dropout: float = 0.0,
    ):
        if checkpoint is None:
            checkpoint = ablation_weight_path(
                "echocare_encoder.pth", "US_ECHOCARE_CHECKPOINT"
            )
        self._d    = _EMBED_DIM
        self._ckpt = checkpoint

        self.backbone      = self._load_model(checkpoint)
        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        log.info(f"[echocare] Ready (embed_dim={self._d})")

    def _load_model(self, checkpoint: str) -> nn.Module:
        try:
            from monai.networks.nets.swin_unetr import SwinTransformer
        except ImportError as exc:
            raise ImportError(
                "EchoCareEncoder requires the MONAI library.\n"
                "Install with:  pip install monai"
            ) from exc

        model = SwinTransformer(**_SWIN_CFG)

        ckpt = torch.load(checkpoint, map_location="cpu")
        # Official loading: checkpoint IS the state_dict, pop mask_token
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        if not isinstance(state, dict):
            raise RuntimeError(
                f"Unexpected checkpoint format in {checkpoint}. "
                "Expected a state_dict or a dict with a 'state_dict' key."
            )
        state.pop("mask_token", None)
        model.load_state_dict(state, strict=True)
        log.info(f"[echocare] Loaded checkpoint from {checkpoint}")
        return model

    @property
    def name(self) -> str:
        return "echocare"

    @property
    def embed_dim(self) -> int:
        return self._d

    @staticmethod
    def _resize_images(images: Tensor, size: int = _IMG_SIZE) -> Tensor:
        if images.shape[-1] == size and images.shape[-2] == size:
            return images
        return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)

    def encode_image(self, images: Tensor) -> dict:
        """
        Encode images with EchoCare.

        Parameters
        ----------
        images : (B, C, H, W)  — recommended 256×256 input

        Returns
        -------
        dict with:
            cls          : (B, 2048)     global-avg-pool of the final stage
            patch_tokens : (B, N, 2048)  spatial tokens from the final stage
        """
        with torch.no_grad():
            dev = next(self.backbone.parameters()).device
            images = self._resize_images(images.to(dev))
            x_outs = self.backbone(images)      # list of 5 feature maps
        feat = x_outs[-1]                       # (B, 2048, H/32, W/32)
        cls  = feat.mean(dim=(2, 3))            # (B, 2048)
        B, C, H, W = feat.shape
        patch_tokens = feat.reshape(B, C, H * W).permute(0, 2, 1)   # (B, N, 2048)
        return {"cls": cls, "patch_tokens": patch_tokens}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames       = clips.reshape(B * T, C, H, W)
        enc          = self.encode_image(frames)
        frame_tokens = enc["cls"].reshape(B, T, -1)
        clip_cls     = self.temporal_pool(frame_tokens)
        return {"clip_cls": clip_cls, "frame_tokens": frame_tokens, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.temporal_pool.parameters()

    def to(self, *args, **kwargs) -> "EchoCareEncoder":
        self.backbone.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "EchoCareEncoder":
        self.backbone.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "EchoCareEncoder":
        self.backbone.eval()   # backbone always frozen
        self.temporal_pool.train(mode)
        return self
