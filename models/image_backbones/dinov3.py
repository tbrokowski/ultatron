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
DINOv3 uses the `attention_mask` parameter in DINOv3ViTAttention.forward to
apply an additive (-inf) bias over padding patch tokens.  We register a
forward pre-hook on every DINOv3ViTAttention module so that the bias is
injected at exactly the right level — NOT on child Linear projections which
don't accept extra kwargs.

The bias has shape (B, n_heads, seq, seq) where seq = 1 + n_reg + N.
Padding positions are -inf in both key and query dimensions.

Note on API change from earlier design
---------------------------------------
The original code injected `attn_bias` (custom kwarg) via hooks registered
on every module whose *path* contained the word "attention".  That approach
broke on nn.Linear layers (q_proj, k_proj, …) because they don't accept
unknown kwargs.  The current design:
  • targets isinstance(mod, DINOv3ViTAttention) only
  • injects `attention_mask`, the documented parameter DINOv3 actually reads
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import ImageBackboneBase, FrozenTeacherBase
from ..hf_loading import load_pretrained
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


# ── Padding mask helpers ──────────────────────────────────────────────────────

def _align_padding_mask_to_patch_grid(
    padding_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Resize (B, ph, pw) mask to match ViT patch grid for ``pixel_values``."""
    ph = pixel_values.shape[-2] // patch_size
    pw = pixel_values.shape[-1] // patch_size
    if padding_mask.shape[1] == ph and padding_mask.shape[2] == pw:
        return padding_mask
    pm = F.interpolate(
        padding_mask.float().unsqueeze(1),
        size=(ph, pw),
        mode="nearest",
    )
    return pm.squeeze(1).bool()


def _make_attn_bias(
    padding_mask: torch.Tensor,   # (B, ph, pw) — True = valid patch
    n_heads: int,
    n_reg: int,
) -> torch.Tensor:
    """
    Build (B, n_heads, seq, seq) additive attention bias.
    seq = 1 (CLS) + n_reg (register tokens) + N (patches).

    Only the KEY dimension of padded positions is set to -inf, preventing
    any query token from attending to padded keys.  The QUERY dimension is
    intentionally left at 0 (unmasked) so that padded-patch query tokens still
    attend to valid keys and produce finite (if semantically irrelevant) hidden
    states.  Setting the Q-side rows to -inf would make softmax(all -inf) = NaN,
    and those NaN hidden states would contaminate subsequent-layer K projections
    via the NaN + (-inf) = NaN identity, collapsing the entire forward pass.

    Padded patch tokens' outputs are excluded from every loss term via s_pmask,
    so their finite-but-meaningless representations never contribute gradients.
    """
    B, ph, pw = padding_mask.shape
    N   = ph * pw
    seq = 1 + n_reg + N

    bias = torch.zeros(B, 1, seq, seq,
                       device=padding_mask.device, dtype=torch.float32)
    is_pad  = ~padding_mask.flatten(1)          # (B, N)  True = padding
    p_start = 1 + n_reg                         # index where patch tokens start

    # Key dimension only: padding patches contribute nothing (columns → -inf).
    # Valid Q tokens (CLS, registers, real patches) will have attention weight 0
    # for padded K positions after softmax, keeping their outputs clean.
    bias[:, 0, :, p_start:].masked_fill_(
        is_pad.unsqueeze(1).expand(-1, seq, -1), float("-inf")
    )
    return bias.expand(-1, n_heads, -1, -1)     # (B, n_heads, seq, seq)


def cls_patch_attention(
    attentions: tuple,
    n_reg: int,
) -> torch.Tensor:
    """
    Mean CLS→patch attention from the last ViT layer.

    OpenUS uses ``teacher_attn.mean(dim=1)`` on a (B, N, N) matrix; for
    DINOv3 we take the CLS query row from the last layer and average heads:

        attentions[-1] : (B, n_heads, seq, seq)
        seq = 1 (CLS) + n_reg + N_patches
        returns (B, N_patches)
    """
    last = attentions[-1]
    return last[:, :, 0, 1 + n_reg:].mean(dim=1)


class _AttentionMaskHook:
    """
    Forward pre-hook registered on each DINOv3ViTAttention module.

    DINOv3ViTAttention.forward signature:
        forward(hidden_states, attention_mask=None, position_embeddings=None, **kwargs)

    The hook injects `attention_mask` into kwargs so that the attention
    interface (eager / SDPA / Flash) receives our additive padding bias.
    The hook is a plain callable (not nn.Module) to stay out of the parameter
    and buffer lists.
    """

    def __init__(self):
        self._mask: Optional[torch.Tensor] = None

    def set_mask(self, mask: Optional[torch.Tensor]) -> None:
        self._mask = mask

    def __call__(self, module, args, kwargs):
        if self._mask is not None:
            kwargs["attention_mask"] = self._mask
        return args, kwargs


# ── DINOv3 backbone wrapper ───────────────────────────────────────────────────

class DINOv3ImageBackbone(ImageBackboneBase):
    """
    Wraps any DINOv3 ViT variant as an ImageBackboneBase.

    Output dict keys: cls, patch_tokens, register_tokens (when n_reg > 0)
    """

    def __init__(self, hf_model, variant_key: str):
        super().__init__()
        self._vit        = hf_model
        self.hidden_size = hf_model.config.hidden_size
        self._n_heads    = hf_model.config.num_attention_heads
        self._n_reg      = getattr(hf_model.config, "num_register_tokens", 0)
        self.variant_key = variant_key
        self._use_gradient_checkpointing = False
        self._eager_attention_enabled = False

        # Import the concrete attention class so we can target it precisely.
        # Lazy import keeps the top-level module importable without transformers.
        from transformers.models.dinov3_vit.modeling_dinov3_vit import (
            DINOv3ViTAttention,
        )

        self._hook         = _AttentionMaskHook()
        self._hook_handles = []
        for name, mod in self._vit.named_modules():
            if isinstance(mod, DINOv3ViTAttention):
                h = mod.register_forward_pre_hook(self._hook, with_kwargs=True)
                self._hook_handles.append(h)

        log.debug(
            "DINOv3ImageBackbone: registered attention-mask hooks on "
            "%d DINOv3ViTAttention modules.", len(self._hook_handles)
        )

    def _ensure_eager_attention(self) -> None:
        """SDPA/Flash backends cannot return attention weights; switch once."""
        if self._eager_attention_enabled:
            return
        if hasattr(self._vit, "set_attn_implementation"):
            self._vit.set_attn_implementation("eager")
            self._eager_attention_enabled = True
            log.info("DINOv3ImageBackbone: switched to eager attention for output_attentions.")

    def _vit_forward_with_mask(
        self,
        pixel_values: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Inner ViT call used as the checkpointed function.

        Receives attention bias as an explicit tensor argument so that
        torch.utils.checkpoint saves and restores it correctly during the
        backward recomputation pass, keeping padding-mask behaviour correct.
        Returns last_hidden_state directly (the dict unpacking happens in
        the outer forward).
        """
        if bias is not None:
            self._hook.set_mask(bias)
        out = self._vit(pixel_values=pixel_values, output_hidden_states=False)
        self._hook.set_mask(None)
        return out.last_hidden_state

    def forward(
        self,
        pixel_values: torch.Tensor,              # (B, 3, H, W)
        padding_mask: Optional[torch.Tensor] = None,  # (B, ph, pw)
        output_attentions: bool = False,
        **kwargs,
    ) -> dict:
        bias: Optional[torch.Tensor] = None
        if padding_mask is not None:
            patch_size = getattr(self._vit.config, "patch_size", 16)
            padding_mask = _align_padding_mask_to_patch_grid(
                padding_mask, pixel_values, patch_size,
            )
            bias = _make_attn_bias(padding_mask, self._n_heads, self._n_reg)
            bias = bias.to(pixel_values.device)

        if self._use_gradient_checkpointing and torch.is_grad_enabled():
            import torch.utils.checkpoint as cp
            # Pass bias as an explicit tensor arg so checkpoint saves/restores
            # it correctly; _vit_forward_with_mask handles None bias safely.
            hs = cp.checkpoint(
                self._vit_forward_with_mask,
                pixel_values,
                bias,
                use_reentrant=False,
            )
            attn_tuple = None
        else:
            if output_attentions:
                self._ensure_eager_attention()
            if bias is not None:
                self._hook.set_mask(bias)
            out = self._vit(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=False,
            )
            self._hook.set_mask(None)
            hs = out.last_hidden_state
            attn_tuple = out.attentions if output_attentions else None

        result = {
            "cls":          hs[:, 0],
            "patch_tokens": hs[:, 1 + self._n_reg:],
        }
        if self._n_reg > 0:
            result["register_tokens"] = hs[:, 1:1 + self._n_reg]
        if attn_tuple is not None:
            result["cls_patch_attention"] = cls_patch_attention(attn_tuple, self._n_reg)
        return result

    def enable_gradient_checkpointing(self) -> None:
        """Enable whole-forward gradient checkpointing on the student ViT.

        Checkpoints the entire 24-layer ViT call as a single unit: only the
        input pixel tensor (and optional attention bias) are retained; all
        intermediate activations are recomputed during backward.  This reduces
        activation memory from O(L·B·N·D) to O(B·N·D) — roughly 24× for
        ViT-L — at the cost of one extra forward pass per backward pass.

        This implementation bypasses HuggingFace's gradient_checkpointing_enable()
        which does not properly wire up _gradient_checkpointing_func on the
        DINOv3Vit layer modules in the installed transformers version.
        """
        self._use_gradient_checkpointing = True
        log.info("DINOv3ImageBackbone: gradient checkpointing enabled.")

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
        self._vit        = hf_model
        self.hidden_size = hf_model.config.hidden_size
        self._n_reg      = getattr(hf_model.config, "num_register_tokens", 0)

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
        hf_model = load_pretrained(
            AutoModel, hf_id, dtype=dtype, hf_cache_dir=hf_cache_dir,
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
    hf_model = load_pretrained(
        AutoModel,
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        hf_cache_dir=hf_cache_dir,
    )
    teacher = DINOv3FrozenTeacher(hf_model)
    log.info(f"  DINOv3FrozenTeacher ready.  D={teacher.hidden_size}")
    return teacher
