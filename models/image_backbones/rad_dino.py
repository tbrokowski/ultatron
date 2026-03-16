"""
models/image_backbones/rad_dino.py  ·  RadDINO backbone
=========================================================

RadDINO is a DINOv2-Base model fine-tuned on radiology images (chest X-ray,
CT, MRI) by Microsoft Research.  It provides stronger out-of-the-box features
for medical imaging compared to DINOv2 trained on natural images.

HF model ID: microsoft/rad-dino
Architecture: ViT-Base/14 (patch size 14, D=768, 12 heads)
Pretrained on: MIMIC-CXR, CheXpert, PadChest, etc.

Note: patch size 14 ≠ our default 16.  When using RadDINO the data config
must set patch_size: 14.  Crops will be padded to multiples of 14.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from ..base import ImageBackboneBase
from ..registry import register_image_backbone

log = logging.getLogger(__name__)

RAD_DINO_HF = "microsoft/rad-dino"



class RadDINOBackbone(ImageBackboneBase):
    """
    RadDINO (ViT-B/14) image backbone.

    RadDINO does not have register tokens.
    Padding mask injection uses the same -inf additive bias approach.
    """

    def __init__(self, hf_model):
        super().__init__()
        self._vit        = hf_model
        self.hidden_size = hf_model.config.hidden_size   # 768
        self._n_heads    = hf_model.config.num_attention_heads
        self._n_reg      = 0   # RadDINO has no register tokens

    def _make_attn_bias(
        self,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, ph, pw = padding_mask.shape
        N   = ph * pw
        seq = 1 + N
        bias = torch.zeros(B, 1, seq, seq, device=padding_mask.device)
        is_pad = ~padding_mask.flatten(1)
        bias[:, 0, :, 1:].masked_fill_(
            is_pad.unsqueeze(1).expand(-1, seq, -1), float("-inf")
        )
        bias[:, 0, 1:, :].masked_fill_(
            is_pad.unsqueeze(2).expand(-1, -1, seq), float("-inf")
        )
        return bias.expand(-1, self._n_heads, -1, -1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # pixel_values expected as (B, 3, H, W) RGB — to_canonical_tensor() upstream
        attn_mask = None
        if padding_mask is not None:
            B, ph, pw = padding_mask.shape
            N = ph * pw
            # (B, 1+N) — True = attend, False = ignore
            attn_mask = torch.ones(B, 1 + N, dtype=torch.bool, device=pixel_values.device)
            attn_mask[:, 1:] = padding_mask.flatten(1)

        out = self._vit(
            pixel_values=pixel_values,
            bool_masked_pos=None,
            output_hidden_states=False,
        )
        hs = out.last_hidden_state   # (B, 1+N, 768)
        return {
            "cls":          hs[:, 0],
            "patch_tokens": hs[:, 1:],
        }

    def __repr__(self):
        return f"RadDINOBackbone(D={self.hidden_size})"


@register_image_backbone("rad_dino")
def _load_rad_dino(
    dtype: torch.dtype = torch.bfloat16,
    hf_cache_dir: Optional[str] = None,
) -> RadDINOBackbone:
    from transformers import AutoModel
    log.info(f"Loading RadDINO from {RAD_DINO_HF} ...")
    hf_model = AutoModel.from_pretrained(
        RAD_DINO_HF, torch_dtype=dtype, cache_dir=hf_cache_dir
    )
    backbone = RadDINOBackbone(hf_model)
    log.info(f"  {backbone}")
    return backbone
