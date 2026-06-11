"""
finetune/backbones/biomedclip_encoder.py  ·  BioMed-CLIP image encoder
=======================================================================

Wraps the Microsoft BiomedCLIP ViT-B/16 image encoder (open_clip).
Image encoder only — text encoder is not used.

Requires: open_clip_torch
    pip install open_clip_torch

Model: hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)

_HF_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_EMBED_DIM   = 512


class BioMedCLIPEncoder(BackboneEncoder):
    """
    BioMed-CLIP ViT-B/16 image encoder.

    Outputs
    -------
    cls          : (B, 512)     image feature (after projection head)
    patch_tokens : (B, 196, 768) intermediate ViT patch tokens (pre-projection)
    """

    def __init__(
        self,
        hf_model_id:      str = _HF_MODEL_ID,
        pretrained:       bool = True,
        temporal_dropout: float = 0.0,
        cache_dir:        Optional[str] = None,
    ):
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "BioMedCLIPEncoder requires open_clip_torch.\n"
                "Install with: pip install open_clip_torch"
            ) from e

        log.info(f"[biomedclip] Loading model: {hf_model_id}")
        model, _, _ = open_clip.create_model_and_transforms(
            hf_model_id,
            pretrained="openai" if not pretrained else "openai",
            cache_dir=cache_dir,
        )

        # Extract image encoder only
        self.visual    = model.visual
        self._proj     = getattr(model, "visual_projection", None)
        self._d        = _EMBED_DIM
        self._vit_dim  = 768   # ViT-B/16 internal dim

        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.visual.parameters():
            p.requires_grad_(False)
        if self._proj is not None:
            for p in self._proj.parameters():
                p.requires_grad_(False)
        self.visual.eval()

        log.info(f"[biomedclip] Ready (embed_dim={self._d})")

    @property
    def name(self) -> str:
        return "biomedclip"

    @property
    def embed_dim(self) -> int:
        return self._d

    def _run_visual(self, images: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        """Run visual encoder; return (cls_feat, patch_tokens_or_None)."""
        with torch.no_grad():
            out = self.visual(images)
        # open_clip visual encoders return the projected CLS token as a tensor
        # or a dict depending on the model version.
        if isinstance(out, dict):
            cls         = out.get("pooler_output", out.get("last_hidden_state", out))
            patch_toks  = out.get("last_hidden_state")
            if patch_toks is not None and patch_toks.dim() == 3:
                patch_toks = patch_toks[:, 1:]  # remove CLS
        elif isinstance(out, torch.Tensor):
            cls        = out
            patch_toks = None
        else:
            cls        = out[0] if isinstance(out, (list, tuple)) else out
            patch_toks = None
        return cls, patch_toks

    def encode_image(self, images: Tensor) -> dict:
        cls, patch = self._run_visual(images)
        return {"cls": cls, "patch_tokens": patch}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames = clips.reshape(B * T, C, H, W)
        cls_bt, _ = self._run_visual(frames)                # (BT, D)
        frame_tokens = cls_bt.reshape(B, T, -1)             # (B, T, D)
        clip_cls     = self.temporal_pool(frame_tokens)     # (B, D)
        return {"clip_cls": clip_cls, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.temporal_pool.parameters()

    def to(self, *args, **kwargs) -> "BioMedCLIPEncoder":
        self.visual.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "BioMedCLIPEncoder":
        self.visual.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "BioMedCLIPEncoder":
        self.visual.eval()
        self.temporal_pool.train(mode)
        return self
