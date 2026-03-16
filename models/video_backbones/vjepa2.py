"""
models/video_backbones/vjepa2.py  ·  V-JEPA2 video backbone family
===================================================================

Registers V-JEPA2 ViT-L, ViT-H, and ViT-G as video backbones.

Supported variants
------------------
  vjepa2_l    facebook/vjepa2-vitl-fpc64-256   D=1024   ← default
  vjepa2_h    facebook/vjepa2-vith-fpc64-256   D=1280
  vjepa2_g    facebook/vjepa2-vitg-fpc64-256   D=1408

V-JEPA2 input contract (HuggingFace)
-------------------------------------
  pixel_values_videos : (B, T, C, H, W)
  context_mask        : (B, N_ctx, 1) LongTensor  — visible token indices
  target_mask         : (B, N_tgt, 1) LongTensor  — masked token indices

  We convert our (B, T, ph, pw) bool tube_mask internally to these index
  tensors, excluding padding frames and padding patches from both sets.

Channel handling
----------------
The data pipeline (dataset.py + transforms.py) delivers (B, T, 3, H, W) float32
tensors to all backbones.  Greyscale ultrasound frames are channel-repeated to
R=G=B by to_canonical_tensor() before they arrive here.  No channel conversion
is performed in this file.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from ..base import VideoBackboneBase
from ..registry import register_video_backbone

log = logging.getLogger(__name__)

_VJEPA2_HF_IDS = {
    "vjepa2_l": "facebook/vjepa2-vitl-fpc64-256",
    "vjepa2_h": "facebook/vjepa2-vith-fpc64-256",
    "vjepa2_g": "facebook/vjepa2-vitg-fpc64-256",
}


# ── Tube mask → index conversion ──────────────────────────────────────────────

def _tube_mask_to_indices(
    tube_mask: torch.Tensor,                    # (B, T, ph, pw) bool
    padding_mask: Optional[torch.Tensor],       # (B, ph, pw) bool
    valid_frames: Optional[torch.Tensor],       # (B, T) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert bool tube_mask to (context_mask, target_mask) index tensors
    as expected by HuggingFace VJEPA2Model.

    context_mask : (B, N_ctx, 1) — indices of visible (unmasked) real tokens
    target_mask  : (B, N_tgt, 1) — indices of masked real tokens

    Padding frames (valid_frames=False) and padding patches (padding_mask=False)
    are excluded from both sets.
    """
    B, T, ph, pw = tube_mask.shape
    N = T * ph * pw

    # Build per-token real mask: frame is real AND patch is real
    real = torch.ones(B, T, ph, pw, dtype=torch.bool, device=tube_mask.device)
    if valid_frames is not None:
        real = real & valid_frames[:, :, None, None]
    if padding_mask is not None:
        real = real & padding_mask[:, None, :, :]

    real_flat    = real.flatten(1)        # (B, N)
    masked_flat  = tube_mask.flatten(1)   # (B, N)

    ctx_flat = real_flat & ~masked_flat   # visible real tokens
    tgt_flat = real_flat &  masked_flat   # masked real tokens

    # Build padded index tensors (pad with sentinel = N for out-of-range)
    ctx_lists = [ctx_flat[b].nonzero(as_tuple=False).squeeze(1) for b in range(B)]
    tgt_lists = [tgt_flat[b].nonzero(as_tuple=False).squeeze(1) for b in range(B)]

    max_ctx = max(x.shape[0] for x in ctx_lists) if ctx_lists else 1
    max_tgt = max(x.shape[0] for x in tgt_lists) if tgt_lists else 1

    ctx_out = torch.full((B, max_ctx), N, dtype=torch.long, device=tube_mask.device)
    tgt_out = torch.full((B, max_tgt), N, dtype=torch.long, device=tube_mask.device)
    for b in range(B):
        nc = ctx_lists[b].shape[0]
        nt = tgt_lists[b].shape[0]
        if nc > 0: ctx_out[b, :nc] = ctx_lists[b]
        if nt > 0: tgt_out[b, :nt] = tgt_lists[b]

    return ctx_out.unsqueeze(-1), tgt_out.unsqueeze(-1)


# ── V-JEPA2 backbone wrapper ──────────────────────────────────────────────────

class VJEPA2VideoBackbone(VideoBackboneBase):
    """
    Wraps any V-JEPA2 ViT variant as a VideoBackboneBase.

    Output dict keys:
      clip_cls    : (B, D)           mean-pooled temporal representation
      tube_tokens : (B, T*ph*pw, D)  all spatiotemporal patch tokens
      predicted   : (B, T*ph*pw, D)  predictor output (only when tube_mask given)
    """

    def __init__(self, hf_model, variant_key: str):
        super().__init__()
        self._model      = hf_model
        self.hidden_size = hf_model.config.hidden_size
        self.variant_key = variant_key


    def forward(
        self,
        pixel_values: torch.Tensor,                    # (B, T, 3, H, W)
        tube_mask: Optional[torch.Tensor] = None,      # (B, T, ph, pw)
        padding_mask: Optional[torch.Tensor] = None,   # (B, ph, pw)
        valid_frames: Optional[torch.Tensor] = None,   # (B, T)
        **kwargs,
    ) -> dict:
        context_mask = target_mask = None
        if tube_mask is not None:
            context_mask, target_mask = _tube_mask_to_indices(
                tube_mask, padding_mask, valid_frames
            )

        out = self._model(
            pixel_values_videos=pixel_values,
            context_mask=context_mask,
            target_mask=target_mask,
            output_hidden_states=False,
        )

        tube_tokens = out.last_hidden_state   # (B, T*ph*pw, D)

        # clip_cls: mean over real (non-padding, non-invalid) tokens
        if padding_mask is not None and valid_frames is not None:
            real_flat = (
                valid_frames[:, :, None, None] & padding_mask[:, None, :, :]
            ).flatten(1).float()   # (B, T*ph*pw)
            denom    = real_flat.sum(1, keepdim=True).clamp(min=1.0)
            clip_cls = (tube_tokens * real_flat.unsqueeze(-1)).sum(1) / denom
        else:
            clip_cls = tube_tokens.mean(1)

        result = {
            "clip_cls":    clip_cls,
            "tube_tokens": tube_tokens,
        }

        if hasattr(out, "predictor_output") and out.predictor_output is not None:
            result["predicted"] = out.predictor_output.last_hidden_state

        return result

    def __repr__(self):
        return (f"VJEPA2VideoBackbone(variant={self.variant_key!r}, "
                f"D={self.hidden_size})")


# ── Registration ──────────────────────────────────────────────────────────────

def _make_vjepa2_factory(variant_key: str):
    hf_id = _VJEPA2_HF_IDS[variant_key]

    def factory(
        dtype: torch.dtype = torch.bfloat16,
        hf_cache_dir: Optional[str] = None,
    ) -> VJEPA2VideoBackbone:
        import copy
        from transformers import AutoModel
        log.info(f"Loading {variant_key} ({hf_id}) ...")
        hf_model = AutoModel.from_pretrained(
            hf_id, torch_dtype=dtype, cache_dir=hf_cache_dir
        )
        backbone = VJEPA2VideoBackbone(hf_model, variant_key=variant_key)
        log.info(f"  {backbone}")
        return backbone

    factory.__name__ = f"load_{variant_key}"
    return factory


for _key in ("vjepa2_l", "vjepa2_h", "vjepa2_g"):
    register_video_backbone(_key)(_make_vjepa2_factory(_key))
