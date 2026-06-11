"""
finetune/backbones/usfm_encoder.py  ·  USFM ultrasound foundation model encoder
=================================================================================

Stub encoder for USFM (Unified Ultrasound Foundation Model).
Paper / repo: https://github.com/openmedlab/USFM

Setup instructions
------------------
1. Clone the USFM repository:
       git clone https://github.com/openmedlab/USFM finetune/backbones/vendor/usfm

2. Download the pretrained weights file:  USFM_latest.pth

3. In comparison.yaml, set:
       - key: usfm
         type: usfm
         checkpoint: /path/to/USFM_latest.pth
         embed_dim: 768          # check USFM docs for the correct value

The encoder will attempt to import the USFM model class from the vendor
directory.  If the vendor code is absent, a clear ImportError is raised.
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

_VENDOR_PATH = Path(__file__).parent / "vendor" / "usfm"


class USFMEncoder(BackboneEncoder):
    """
    USFM image encoder stub.

    Loads the USFM backbone from a local checkpoint and wraps it in the
    BackboneEncoder interface.  Attaches a TemporalAttentionPool for video
    task compatibility.

    Parameters
    ----------
    checkpoint : str
        Path to USFM_latest.pth.
    embed_dim : int
        USFM output feature dimension (check USFM docs; typically 768 or 1024).
    temporal_dropout : float
    """

    def __init__(
        self,
        checkpoint:       str,
        embed_dim:        int = 768,
        temporal_dropout: float = 0.0,
    ):
        self._d    = embed_dim
        self._ckpt = checkpoint

        self.backbone = self._load_model(checkpoint)
        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        log.info(f"[usfm] Ready (embed_dim={self._d})")

    def _load_model(self, checkpoint: str) -> nn.Module:
        if not _VENDOR_PATH.exists():
            raise ImportError(
                f"USFM vendor code not found at {_VENDOR_PATH}.\n"
                "Please clone https://github.com/openmedlab/USFM into "
                f"{_VENDOR_PATH} and re-run."
            )
        if str(_VENDOR_PATH) not in sys.path:
            sys.path.insert(0, str(_VENDOR_PATH))

        try:
            # Adjust the import below to match the actual USFM module structure
            from models.build import build_model  # type: ignore[import]
            model = build_model(checkpoint)
        except ImportError:
            try:
                # Fallback: attempt generic torch.load
                log.warning("[usfm] Could not import USFM build_model; "
                            "falling back to torch.load")
                ckpt  = torch.load(checkpoint, map_location="cpu")
                model = ckpt.get("model", ckpt)
                if not isinstance(model, nn.Module):
                    raise RuntimeError(
                        "USFM checkpoint does not contain an nn.Module under 'model' key. "
                        "Please implement _load_model() for your USFM version."
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load USFM model from {checkpoint}: {exc}\n"
                    "See finetune/backbones/usfm_encoder.py for setup instructions."
                ) from exc
        return model

    @property
    def name(self) -> str:
        return "usfm"

    @property
    def embed_dim(self) -> int:
        return self._d

    def encode_image(self, images: Tensor) -> dict:
        with torch.no_grad():
            out = self.backbone(images)
        # Adapt the output depending on what USFM's forward() returns.
        # Most ViT-style models return (cls, patch_tokens) or just a tensor.
        if isinstance(out, dict):
            cls   = out.get("cls", out.get("pooler_output"))
            patch = out.get("patch_tokens", out.get("last_hidden_state"))
        elif isinstance(out, (list, tuple)) and len(out) >= 2:
            cls, patch = out[0], out[1]
        else:
            cls   = out
            patch = None
        return {"cls": cls, "patch_tokens": patch}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames = clips.reshape(B * T, C, H, W)
        enc    = self.encode_image(frames)
        frame_tokens = enc["cls"].reshape(B, T, -1)      # (B, T, D)
        clip_cls     = self.temporal_pool(frame_tokens)  # (B, D)
        return {"clip_cls": clip_cls, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.temporal_pool.parameters()

    def to(self, *args, **kwargs) -> "USFMEncoder":
        self.backbone.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "USFMEncoder":
        self.backbone.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "USFMEncoder":
        self.backbone.eval()
        self.temporal_pool.train(mode)
        return self
