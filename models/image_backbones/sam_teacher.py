"""
models/image_backbones/sam_teacher.py  ·  SAM2 / SAM3 frozen teacher
======================================================================

Provides a permanently frozen SAM2 / SAM3 image encoder for RADIO-style
multi-teacher knowledge distillation.  The student is trained to produce
patch features that are compatible with SAM's segmentation feature space,
so the learned representations can be directly passed to SAM's prompt
encoder / mask decoder at inference time.

Supported variants
------------------
  sam2_b_plus   facebook/sam2-hiera-base-plus   D=256
  sam2_l        facebook/sam2-hiera-large        D=256
  sam2_h        facebook/sam2-hiera-huge         D=256

SAM3 note
---------
SAM3's Perception Encoder is not yet available as a public HuggingFace
model.  When released, add its HF model ID to _SAM_HF_IDS and register
it exactly like the SAM2 variants below.  The teacher wrapper is
architecture-agnostic: it expects any model with a callable
vision_encoder (or image_encoder) attribute.

Input / output contract
-----------------------
forward(pixel_values: Tensor (B, 3, H, W)) → dict:
  cls          : (B, D)     mean-pooled patch embedding (global context)
  patch_tokens : (B, N, D)  spatial patch embeddings, interpolated to match
                             the student's native-resolution patch grid.

The student's patch grid size is passed as (ph, pw) to the forward call.
When not provided, features are returned at SAM's native grid resolution.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import FrozenTeacherBase
from ..registry import register_frozen_teacher

log = logging.getLogger(__name__)

_SAM_HF_IDS = {
    "sam2_b_plus": "facebook/sam2-hiera-base-plus",
    "sam2_l":      "facebook/sam2-hiera-large",
    "sam2_h":      "facebook/sam2-hiera-huge",
}

# SAM2's image encoder expects 1024×1024 inputs but also works at 512×512.
# We default to 512 to avoid the memory spike at full resolution.
_SAM_DEFAULT_SIZE = 512

# ImageNet normalisation expected by SAM2's vision encoder
_SAM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_SAM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SAMFrozenTeacher(FrozenTeacherBase):
    """
    Permanently frozen SAM2/SAM3 image encoder for knowledge distillation.

    Resizes inputs to target_size, applies ImageNet normalisation, runs the
    SAM vision encoder, and returns patch embeddings.  Patch tokens can be
    bilinearly interpolated to match the student's spatial grid.

    Parameters
    ----------
    hf_model    : a HuggingFace SAM2Model (or compatible)
    target_size : spatial resolution fed to the SAM encoder (default 512)
    """

    def __init__(self, hf_model, target_size: int = _SAM_DEFAULT_SIZE):
        super().__init__()
        self._model      = hf_model
        self._target_size = target_size

        # Determine output feature dimension from config
        cfg = hf_model.config
        if hasattr(cfg, "vision_encoder") and hasattr(cfg.vision_encoder, "embed_dim"):
            self.hidden_size = cfg.vision_encoder.embed_dim
        elif hasattr(cfg, "hidden_size"):
            self.hidden_size = cfg.hidden_size
        else:
            self.hidden_size = 256   # SAM2 Hiera default output channels

        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,           # (B, 3, H, W)  values in [0, 1]
        target_ph: Optional[int] = None,      # student patch grid height (optional)
        target_pw: Optional[int] = None,      # student patch grid width  (optional)
        **kwargs,
    ) -> dict:
        B = pixel_values.shape[0]
        device = pixel_values.device

        # Resize to SAM's expected resolution
        resized = F.interpolate(
            pixel_values.float(),
            size=(self._target_size, self._target_size),
            mode="bilinear",
            align_corners=False,
        )

        # ImageNet normalisation (SAM2 expects this)
        mean = _SAM_MEAN.to(device)
        std  = _SAM_STD.to(device)
        resized = (resized - mean) / std

        # Run vision encoder — handle both SAM2 HuggingFace API styles
        feats = self._run_encoder(resized)   # (B, D, h, w)  or  (B, h*w, D)

        # Ensure (B, N, D) layout
        if feats.ndim == 4:
            feats = feats.flatten(2).permute(0, 2, 1)   # (B, h*w, D)
        # feats: (B, N_sam, D)

        # Optionally interpolate to match student's spatial grid
        if target_ph is not None and target_pw is not None:
            N_sam = feats.shape[1]
            h_sam = w_sam = int(N_sam ** 0.5)
            if h_sam * w_sam != N_sam:
                h_sam = target_ph   # fallback: assume student grid
                w_sam = target_pw
            sam_spatial = feats.permute(0, 2, 1).reshape(B, -1, h_sam, w_sam)
            sam_spatial = F.interpolate(
                sam_spatial.float(),
                size=(target_ph, target_pw),
                mode="bilinear",
                align_corners=False,
            )
            feats = sam_spatial.flatten(2).permute(0, 2, 1)   # (B, target_ph*pw, D)

        return {
            "cls":          feats.mean(dim=1),  # (B, D)
            "patch_tokens": feats,              # (B, N, D)
        }

    def _run_encoder(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Dispatch to whichever encoder attribute the SAM model exposes."""
        m = self._model

        # SAM2 HuggingFace: Sam2Model → vision_encoder
        if hasattr(m, "vision_encoder"):
            out = m.vision_encoder(pixel_values)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state
            if hasattr(out, "hidden_states"):
                return out.hidden_states[-1]
            if isinstance(out, torch.Tensor):
                return out

        # SAM1 / SAM2-alt: model.image_encoder
        if hasattr(m, "image_encoder"):
            out = m.image_encoder(pixel_values)
            if isinstance(out, torch.Tensor):
                return out
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state

        # Generic fallback: call model directly
        out = m(pixel_values=pixel_values)
        if hasattr(out, "image_embeddings"):
            return out.image_embeddings
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        raise ValueError(
            f"SAMFrozenTeacher: cannot extract patch features from {type(m).__name__}. "
            "Add an explicit branch to _run_encoder."
        )


# ── Registration ──────────────────────────────────────────────────────────────

def _make_sam_factory(variant_key: str):
    hf_id = _SAM_HF_IDS[variant_key]

    def factory(
        dtype: torch.dtype = torch.bfloat16,
        hf_cache_dir: Optional[str] = None,
        device: str = "cuda",
        target_size: int = _SAM_DEFAULT_SIZE,
    ) -> SAMFrozenTeacher:
        from transformers import AutoModel
        log.info(f"Loading frozen SAM teacher {variant_key} ({hf_id}) ...")
        hf_model = AutoModel.from_pretrained(
            hf_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            cache_dir=hf_cache_dir,
        )
        teacher = SAMFrozenTeacher(hf_model, target_size=target_size)
        log.info(f"  SAMFrozenTeacher ready.  D={teacher.hidden_size}")
        return teacher

    factory.__name__ = f"load_{variant_key}"
    return factory


for _key in ("sam2_b_plus", "sam2_l", "sam2_h"):
    register_frozen_teacher(_key)(_make_sam_factory(_key))
