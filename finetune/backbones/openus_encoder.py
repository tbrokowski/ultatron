"""
finetune/backbones/openus_encoder.py  ·  OpenUS ultrasound foundation model encoder
====================================================================================

Stub encoder for OpenUS (general-purpose ultrasound foundation model).
Paper / repo: https://github.com/XZheng0427/OpenUS

Setup instructions
------------------
1. Clone the OpenUS repository:
       git clone https://github.com/XZheng0427/OpenUS finetune/backbones/vendor/openus

2. Download the pretrained weights from the OpenUS release page.

3. In comparison.yaml, set:
       - key: openus
         type: openus
         checkpoint: /path/to/openus_weights.pth
         embed_dim: 768          # check OpenUS docs
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)

_VENDOR_PATH = Path(__file__).parent / "vendor" / "openus"


class OpenUSEncoder(BackboneEncoder):
    """
    OpenUS general-purpose ultrasound foundation model encoder stub.

    Parameters
    ----------
    checkpoint : str
        Path to the OpenUS pretrained weights file.
    embed_dim : int
        Feature dimension (check OpenUS docs; typically 768 or 1024).
    temporal_dropout : float
        Dropout for TemporalAttentionPool used in encode_video().
    """

    def __init__(
        self,
        checkpoint:       str,
        embed_dim:        int = 768,
        temporal_dropout: float = 0.0,
    ):
        self._d    = embed_dim
        self._ckpt = checkpoint

        self.backbone      = self._load_model(checkpoint)
        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        log.info(f"[openus] Ready (embed_dim={self._d})")

    def _load_model(self, checkpoint: str) -> nn.Module:
        if not _VENDOR_PATH.exists():
            raise ImportError(
                f"OpenUS vendor code not found at {_VENDOR_PATH}.\n"
                "Please clone https://github.com/XZheng0427/OpenUS into "
                f"{_VENDOR_PATH} and re-run."
            )
        if str(_VENDOR_PATH) not in sys.path:
            sys.path.insert(0, str(_VENDOR_PATH))

        try:
            # Adjust import to match actual OpenUS module structure
            from model import OpenUSModel  # type: ignore[import]
            model = OpenUSModel()
            ckpt  = torch.load(checkpoint, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state, strict=True)
        except ImportError as exc:
            raise ImportError(
                f"Could not import OpenUSModel from {_VENDOR_PATH}.\n"
                "Update the import in finetune/backbones/openus_encoder.py "
                "to match the OpenUS repository structure."
            ) from exc
        return model

    @property
    def name(self) -> str:
        return "openus"

    @property
    def embed_dim(self) -> int:
        return self._d

    def encode_image(self, images: Tensor) -> dict:
        with torch.no_grad():
            out = self.backbone(images)
        if isinstance(out, dict):
            cls   = out.get("cls", out.get("pooler_output"))
            patch = out.get("patch_tokens", out.get("last_hidden_state"))
        elif isinstance(out, (list, tuple)) and len(out) >= 2:
            cls, patch = out[0], out[1]
        else:
            cls, patch = out, None
        return {"cls": cls, "patch_tokens": patch}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames   = clips.reshape(B * T, C, H, W)
        enc      = self.encode_image(frames)
        frame_tk = enc["cls"].reshape(B, T, -1)     # (B, T, D)
        clip_cls = self.temporal_pool(frame_tk)     # (B, D)
        return {"clip_cls": clip_cls, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.temporal_pool.parameters()

    def to(self, *args, **kwargs) -> "OpenUSEncoder":
        self.backbone.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "OpenUSEncoder":
        self.backbone.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "OpenUSEncoder":
        self.backbone.eval()
        self.temporal_pool.train(mode)
        return self
