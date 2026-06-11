"""
finetune/backbones/echocare_encoder.py  ·  EchoCare echo foundation model encoder
==================================================================================

Stub encoder for EchoCare (CAIR-HKISI cardiac echo foundation model).
Paper / repo: https://github.com/CAIR-HKISI/EchoCare

Setup instructions
------------------
1. Clone the EchoCare repository:
       git clone https://github.com/CAIR-HKISI/EchoCare finetune/backbones/vendor/echocare

2. Download the pretrained weights from the EchoCare release page.

3. In comparison.yaml, set:
       - key: echocare
         type: echocare
         checkpoint: /path/to/echocare_weights.pth
         embed_dim: 768          # check EchoCare docs

EchoCare is a video-native echo model; encode_video() uses the model directly
without the TemporalAttentionPool.  encode_image() encodes the image as a
single-frame clip.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)

_VENDOR_PATH = Path(__file__).parent / "vendor" / "echocare"


class EchoCareEncoder(BackboneEncoder):
    """
    EchoCare cardiac echo foundation model encoder stub.

    Parameters
    ----------
    checkpoint : str
        Path to the EchoCare pretrained weights file.
    embed_dim : int
        Feature dimension (check EchoCare docs; typically 768).
    is_video : bool
        True if EchoCare's forward() accepts video clips (T, C, H, W) natively.
        If False, falls back to per-frame encoding + TemporalAttentionPool.
    temporal_dropout : float
        Dropout for TemporalAttentionPool (only used when is_video=False).
    """

    def __init__(
        self,
        checkpoint:       str,
        embed_dim:        int = 768,
        is_video:         bool = True,
        temporal_dropout: float = 0.0,
    ):
        self._d        = embed_dim
        self._ckpt     = checkpoint
        self._is_video = is_video

        self.backbone = self._load_model(checkpoint)

        if not is_video:
            self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        log.info(f"[echocare] Ready (embed_dim={self._d}, video_native={is_video})")

    def _load_model(self, checkpoint: str) -> nn.Module:
        if not _VENDOR_PATH.exists():
            raise ImportError(
                f"EchoCare vendor code not found at {_VENDOR_PATH}.\n"
                "Please clone https://github.com/CAIR-HKISI/EchoCare into "
                f"{_VENDOR_PATH} and re-run."
            )
        if str(_VENDOR_PATH) not in sys.path:
            sys.path.insert(0, str(_VENDOR_PATH))

        try:
            # Adjust import to match actual EchoCare module structure
            from model import EchoCareModel  # type: ignore[import]
            model = EchoCareModel()
            ckpt  = torch.load(checkpoint, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state, strict=True)
        except ImportError as exc:
            raise ImportError(
                f"Could not import EchoCareModel from {_VENDOR_PATH}.\n"
                "Update the import in finetune/backbones/echocare_encoder.py "
                "to match the EchoCare repository structure."
            ) from exc
        return model

    @property
    def name(self) -> str:
        return "echocare"

    @property
    def embed_dim(self) -> int:
        return self._d

    @property
    def is_video_native(self) -> bool:
        return self._is_video

    def encode_image(self, images: Tensor) -> dict:
        with torch.no_grad():
            if self._is_video:
                clips = images.unsqueeze(1)          # (B, 1, C, H, W)
                out   = self.backbone(clips)
                cls   = out["clip_cls"] if isinstance(out, dict) else out
            else:
                out = self.backbone(images)
                cls = out["cls"] if isinstance(out, dict) else out
        return {"cls": cls, "patch_tokens": None}

    def encode_video(self, clips: Tensor) -> dict:
        if self._is_video:
            with torch.no_grad():
                out      = self.backbone(clips)
                clip_cls = out["clip_cls"] if isinstance(out, dict) else out
            return {"clip_cls": clip_cls, "tube_tokens": None}
        else:
            B, T, C, H, W = clips.shape
            frames   = clips.reshape(B * T, C, H, W)
            enc      = self.encode_image(frames)
            frame_tk = enc["cls"].reshape(B, T, -1)
            clip_cls = self.temporal_pool(frame_tk)
            return {"clip_cls": clip_cls, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        if not self._is_video and hasattr(self, "temporal_pool"):
            return self.temporal_pool.parameters()
        return iter([])

    def to(self, *args, **kwargs) -> "EchoCareEncoder":
        self.backbone.to(*args, **kwargs)
        if hasattr(self, "temporal_pool"):
            self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "EchoCareEncoder":
        self.backbone.eval()
        if hasattr(self, "temporal_pool"):
            self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "EchoCareEncoder":
        self.backbone.eval()
        if hasattr(self, "temporal_pool"):
            self.temporal_pool.train(mode)
        return self
