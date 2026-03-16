"""
models/image_backbones/swin_v2.py  ·  SwinTransformerV2 backbone (stub)
========================================================================

Stub registration for SwinV2-Large.  Implement forward() when needed.
SwinV2 produces hierarchical features; we use the final stage output
and average-pool to get patch tokens, with no CLS token (synthesised
by global average pooling).

HF model ID: microsoft/swinv2-large-patch4-window12-192-22k
Patch size: 4 (not 16) — requires data config patch_size: 4.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from ..base import ImageBackboneBase
from ..registry import register_image_backbone

log = logging.getLogger(__name__)

SWIN_V2_L_HF = "microsoft/swinv2-large-patch4-window12-192-22k"


class SwinV2Backbone(ImageBackboneBase):
    """
    SwinTransformerV2-Large image backbone.
    Hierarchical encoder — final stage features used as patch tokens.
    CLS synthesised by global average pooling over patch tokens.
    """

    def __init__(self, hf_model):
        super().__init__()
        self._grey2rgb   = nn.Identity()   # SwinV2 can be adapted for 1-ch
        self._model      = hf_model
        # SwinV2-L final stage hidden dim = 1536
        self.hidden_size = hf_model.config.hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # SwinV2 expects (B, C, H, W) — repeat greyscale to 3-ch
        rgb = pixel_values.expand(-1, 3, -1, -1)
        out = self._model(pixel_values=rgb, output_hidden_states=False)

        # last_hidden_state: (B, N, D) where N = (H/patch_size)*(W/patch_size)
        patch_tokens = out.last_hidden_state
        cls_token    = patch_tokens.mean(1)   # synthesised CLS

        return {
            "cls":          cls_token,
            "patch_tokens": patch_tokens,
        }

    def __repr__(self):
        return f"SwinV2Backbone(D={self.hidden_size})"


@register_image_backbone("swin_v2_l")
def _load_swin_v2_l(
    dtype: torch.dtype = torch.bfloat16,
    hf_cache_dir: Optional[str] = None,
) -> SwinV2Backbone:
    from transformers import AutoModel
    log.info(f"Loading SwinV2-L from {SWIN_V2_L_HF} ...")
    hf_model = AutoModel.from_pretrained(
        SWIN_V2_L_HF, torch_dtype=dtype, cache_dir=hf_cache_dir
    )
    backbone = SwinV2Backbone(hf_model)
    log.info(f"  {backbone}")
    return backbone
