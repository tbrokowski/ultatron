"""
models/image_backbones/dinov3.py  ·  DINOv3 image backbone family
==================================================================

Registers all DINOv3 ViT variants as image backbones.  A single wrapper
class (DINOv3ImageBackbone) handles the full family.

Supported variants
------------------
  dinov3_s      facebook/dinov3-vits16-pretrain-lvd1689m    D=384   heads=6
  dinov3_splus  facebook/dinov3-vitsplus-pretrain-lvd1689m  D=384   heads=6
  dinov3_b      facebook/dinov3-vitb16-pretrain-lvd1689m    D=768   heads=12
  dinov3_l      facebook/dinov3-vitl16-pretrain-lvd1689m    D=1024  heads=16
  dinov3_hplus  facebook/dinov3-vith16plus-pretrain-lvd1689m D=1280 heads=16

Frozen teacher
--------------
  dinov3_7b     facebook/dinov3-vit7b16-pretrain-lvd1689m   D=4096  (frozen only)

Channel handling
----------------
The data pipeline (dataset.py + transforms.py) delivers (B, 3, H, W) float32
tensors to all backbones.  Greyscale ultrasound frames are channel-repeated to
R=G=B by to_canonical_tensor() before they arrive here.  No channel conversion
is performed in this file.

Padding mask injection
----------------------
DINOv3 uses PyTorch SDPA attention.  We inject an additive -inf bias via a
forward pre-hook on each attention layer so that padding patch tokens are
masked from all attention computations.  The hook is registered once at
construction time and removed cleanly when the backbone is discarded.

The bias has shape (B, n_heads, seq, seq) where seq = 1 + n_reg + N.
Padding positions are -inf in both key and query dimensions.
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

from ..base import ImageBackboneBase, FrozenTeacherBase
from ..registry import register_image_backbone, register_frozen_teacher

log = logging.getLogger(__name__)

# ── HuggingFace model ID table ────────────────────────────────────────────────
_DINOV3_HF_IDS = {
    "dinov3_s":     "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_splus": "facebook/dinov3-vitsplus-pretrain-lvd1689m",
    "dinov3_b":     "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_l":     "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_hplus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "dinov3_7b":    "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}


# ── Padding mask injection ────────────────────────────────────────────────────

def _make_attn_bias(
    padding_mask: torch.Tensor,   # (B, ph, pw)
    n_heads: int,
    n_reg: int,
) -> torch.Tensor:
    """
    Build (B, n_heads, seq, seq) additive attention bias.
    seq = 1 (CLS) + n_reg (registers) + N (patches).
    Padding token positions get -inf in both key and query dims.
    """
    B, ph, pw = padding_mask.shape
    N   = ph * pw
    seq = 1 + n_reg + N

    bias = torch.zeros(B, 1, seq, seq,
                       device=padding_mask.device, dtype=torch.float32)
    is_pad  = ~padding_mask.flatten(1)          # (B, N)  True=padding
    p_start = 1 + n_reg                         # index where patches start

    # Key dimension: padding patches contribute nothing (columns → -inf)
    bias[:, 0, :, p_start:].masked_fill_(
        is_pad.unsqueeze(1).expand(-1, seq, -1), float("-inf")
    )
    # Query dimension: padding patches attend to nothing (rows → -inf)
    bias[:, 0, p_start:, :].masked_fill_(
        is_pad.unsqueeze(2).expand(-1, -1, seq), float("-inf")
    )
    return bias.expand(-1, n_heads, -1, -1)


class _AttnBiasHook:
    """
    Forward pre-hook that injects an additive attention bias into every
    attention layer call.  Stored as a plain object (not nn.Module) so
    it doesn't appear in module parameters/buffers.
    """
    def __init__(self):
        self._bias: Optional[torch.Tensor] = None

    def set_bias(self, bias: Optional[torch.Tensor]):
        self._bias = bias

    def __call__(self, module, args, kwargs):
        if self._bias is not None:
            kwargs["attn_bias"] = self._bias
        return args, kwargs




# ── DINOv3 backbone wrapper ───────────────────────────────────────────────────

class DINOv3ImageBackbone(ImageBackboneBase):
    """
    Wraps any DINOv3 ViT variant as an ImageBackboneBase.

    Output dict keys: cls, patch_tokens, register_tokens (optional)
    """

    def __init__(self, hf_model, variant_key: str):
        super().__init__()
        self._vit       = hf_model
        self.hidden_size = hf_model.config.hidden_size
        self._n_heads   = hf_model.config.num_attention_heads
        self._n_reg     = getattr(hf_model.config, "num_register_tokens", 0)
        self.variant_key = variant_key

        # Register attention bias hook on every attention layer
        self._hook          = _AttnBiasHook()
        self._hook_handles  = []
        for name, mod in self._vit.named_modules():
            if "attention" in name.lower() and hasattr(mod, "forward"):
                h = mod.register_forward_pre_hook(self._hook, with_kwargs=True)
                self._hook_handles.append(h)

    def forward(
        self,
        pixel_values: torch.Tensor,            # (B, 3, H, W)
        padding_mask: Optional[torch.Tensor] = None,   # (B, ph, pw)
        **kwargs,
    ) -> dict:
        if padding_mask is not None:
            bias = _make_attn_bias(padding_mask, self._n_heads, self._n_reg)
            self._hook.set_bias(bias.to(pixel_values.device))
        else:
            self._hook.set_bias(None)

        out = self._vit(pixel_values=pixel_values, output_hidden_states=False)
        self._hook.set_bias(None)   # always clear after forward

        hs = out.last_hidden_state  # (B, 1 + n_reg + N, D)
        result = {
            "cls":          hs[:, 0],
            "patch_tokens": hs[:, 1 + self._n_reg:],
        }
        if self._n_reg > 0:
            result["register_tokens"] = hs[:, 1:1 + self._n_reg]
        return result

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def __repr__(self):
        return (f"DINOv3ImageBackbone(variant={self.variant_key!r}, "
                f"D={self.hidden_size}, heads={self._n_heads}, "
                f"registers={self._n_reg})")


# ── Frozen DINOv3-7B teacher ──────────────────────────────────────────────────

class DINOv3FrozenTeacher(FrozenTeacherBase):
    """
    Permanently frozen DINOv3-7B used only for knowledge distillation.
    Loaded in bfloat16, placed on a dedicated device shard.
    """

    def __init__(self, hf_model):
        super().__init__()
        self._vit       = hf_model
        self.hidden_size = hf_model.config.hidden_size
        self._n_reg     = getattr(hf_model.config, "num_register_tokens", 0)

        for p in self._vit.parameters():
            p.requires_grad_(False)
        self._vit.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, **kwargs) -> dict:
        out = self._vit(pixel_values=pixel_values)
        hs  = out.last_hidden_state
        return {
            "cls":          hs[:, 0],
            "patch_tokens": hs[:, 1 + self._n_reg:],
        }


# ── Registration ──────────────────────────────────────────────────────────────
# Each factory loads from HF and returns an un-placed backbone instance.
# Callers do .to(device, dtype) themselves.

def _make_dinov3_factory(variant_key: str):
    hf_id = _DINOV3_HF_IDS[variant_key]

    def factory(
        dtype: torch.dtype = torch.bfloat16,
        hf_cache_dir: Optional[str] = None,
    ) -> DINOv3ImageBackbone:
        from transformers import AutoModel
        log.info(f"Loading {variant_key} ({hf_id}) ...")
        hf_model = AutoModel.from_pretrained(
            hf_id, torch_dtype=dtype, cache_dir=hf_cache_dir
        )
        backbone = DINOv3ImageBackbone(hf_model, variant_key=variant_key)
        log.info(f"  {backbone}")
        return backbone

    factory.__name__ = f"load_{variant_key}"
    return factory


# Register all trainable variants
for _key in ("dinov3_s", "dinov3_splus", "dinov3_b", "dinov3_l", "dinov3_hplus"):
    register_image_backbone(_key)(_make_dinov3_factory(_key))


# Register 7B frozen teacher separately
@register_frozen_teacher("dinov3_7b")
def _load_dinov3_7b(
    dtype: torch.dtype = torch.bfloat16,
    hf_cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> DINOv3FrozenTeacher:
    from transformers import AutoModel
    hf_id = _DINOV3_HF_IDS["dinov3_7b"]
    log.info(f"Loading frozen DINOv3-7B teacher ({hf_id}) ...")
    log.info("  Requires ~14 GB VRAM (bf16).")
    hf_model = AutoModel.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=hf_cache_dir,
    )
    teacher = DINOv3FrozenTeacher(hf_model)
    log.info(f"  DINOv3FrozenTeacher ready.  D={teacher.hidden_size}")
    return teacher
