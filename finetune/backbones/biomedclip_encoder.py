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
import torch.nn.functional as F
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)

_HF_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
_EMBED_DIM   = 512
_IMG_SIZE    = 224


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
        # HF-hub models load weights from the hub; no pretrained= tag needed.
        load_kwargs: dict = {}
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        model, _, _ = open_clip.create_model_and_transforms(hf_model_id, **load_kwargs)

        # Extract image encoder only
        self.visual = model.visual
        self._d       = _EMBED_DIM
        self._vit_dim = 768   # ViT-B/16 internal dim (pre-projection patch width)

        # Return (pooled, patch_tokens) from the ViT forward pass.
        if hasattr(self.visual, "output_tokens"):
            self.visual.output_tokens = True

        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.visual.parameters():
            p.requires_grad_(False)
        self.visual.eval()

        log.info(f"[biomedclip] Ready (embed_dim={self._d})")

    @property
    def name(self) -> str:
        return "biomedclip"

    @property
    def embed_dim(self) -> int:
        return self._d

    @property
    def patch_embed_dim(self) -> int:
        """ViT patch-token width (768); differs from projected CLS dim (512)."""
        return self._vit_dim

    @staticmethod
    def _resize_images(images: Tensor, size: int = _IMG_SIZE) -> Tensor:
        if images.shape[-1] == size and images.shape[-2] == size:
            return images
        return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)

    def _run_visual(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Run visual encoder; return (cls_feat, patch_tokens)."""
        with torch.no_grad():
            out = self.visual(images)
        # With output_tokens=True, open_clip ViT returns (pooled, patch_tokens).
        if isinstance(out, (tuple, list)) and len(out) == 2:
            cls, patch_toks = out
        elif isinstance(out, dict):
            cls        = out.get("pooler_output", out.get("last_hidden_state", out))
            patch_toks = out.get("last_hidden_state")
            if patch_toks is not None and patch_toks.dim() == 3:
                patch_toks = patch_toks[:, 1:]  # remove CLS
        elif isinstance(out, torch.Tensor):
            cls, patch_toks = out, None
        else:
            cls, patch_toks = out[0], None

        if patch_toks is None:
            patch_toks = self._extract_patch_tokens(images)
        return cls, patch_toks

    def _extract_patch_tokens(self, images: Tensor) -> Tensor:
        """
        Fallback patch-token extraction when open_clip does not return them.
        Segmentation heads require (B, N, D) tokens; CLS-only output crashes.
        """
        v = self.visual
        if hasattr(v, "trunk") and hasattr(v.trunk, "forward_features"):
            feats = v.trunk.forward_features(images)
            if feats.dim() == 3 and feats.shape[1] > 1:
                return feats[:, 1:, :self._vit_dim]
        if hasattr(v, "transformer"):
            x = v.conv1(images) if hasattr(v, "conv1") else images
            if hasattr(v, "class_embedding"):
                cls_tok = v.class_embedding.to(x.dtype) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                )
                x = torch.cat([cls_tok, x], dim=1)
            if hasattr(v, "positional_embedding"):
                x = x + v.positional_embedding.to(x.dtype)
            x = v.ln_pre(x) if hasattr(v, "ln_pre") else x
            x = v.transformer(x)
            x = v.ln_post(x) if hasattr(v, "ln_post") else x
            if x.dim() == 3 and x.shape[1] > 1:
                return x[:, 1:, :self._vit_dim]
        raise RuntimeError(
            "[biomedclip] Could not extract patch tokens from visual encoder. "
            "Segmentation tasks require patch-level features."
        )

    def encode_image(self, images: Tensor) -> dict:
        dev = next(self.visual.parameters()).device
        images = self._resize_images(images.to(dev))
        cls, patch = self._run_visual(images)
        return {"cls": cls, "patch_tokens": patch}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames = clips.reshape(B * T, C, H, W)
        dev = next(self.visual.parameters()).device
        frames = self._resize_images(frames.to(dev))
        cls_bt, _ = self._run_visual(frames)                # (BT, D)
        frame_tokens = cls_bt.reshape(B, T, -1)             # (B, T, D)
        clip_cls     = self.temporal_pool(frame_tokens)     # (B, D)
        return {"clip_cls": clip_cls, "frame_tokens": frame_tokens, "tube_tokens": None}

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
