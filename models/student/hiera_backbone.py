"""
models/student/hiera_backbone.py  ·  Hiera-based Ultatron student encoder
==========================================================================

HieraStudentBackbone wraps a pretrained Hiera-Large model and extends it
with lightweight temporal adapters for video inputs.

Pretrained weight sources
--------------------------
hiera_large_video_mae_k400
    Loaded via the `hiera-transformer` pip package:
        pip install hiera-transformer
    Hiera.from_pretrained("facebook/hiera_large_16x224.mae_k400")
    Weights: MAE pretrained on Kinetics-400 video → spatial + temporal both warm.
    Recommended default.

sam2_hiera_large
    Loaded via HuggingFace `transformers`:
        from transformers import AutoModel
        AutoModel.from_pretrained("facebook/sam2.1-hiera-large")
    Extracts the image encoder backbone only.
    Frames are processed independently (no native temporal weights).

Architecture (Hiera-Large)
--------------------------
  Patch embed : Conv2d(3, 144, kernel=7, stride=4, padding=3)
               → effective patch stride = 4
  Stages      : 4  (depths [2, 3, 16, 3] standard, [2, 6, 36, 4] SAM2-Large)
  Embed dims  : [144, 288, 576, 1152]
  Heads       : [2, 4, 8, 16]
  Downsampling: query pooling 2×2 between stages (Hiera native)

Multi-scale outputs
--------------------
After each stage the feature map is captured:
  F1 : (B, T, N1, D1)  highest resolution, stride 4
  F2 : (B, T, N2, D2)  stride 8
  F3 : (B, T, N3, D3)  stride 16
  F4 : (B, T, N4, D4)  stride 32  ← used for global token

global : (B, D4)   temporal + spatial mean pool of F4
dense  : (B, T, N4, D4)  alias for F4 (used by loss functions)

Native resolution
------------------
Hiera's default absolute position embedding is replaced with 2D sinusoidal
encodings computed on the fly from the actual (ph, pw) grid.  This allows
arbitrary input sizes without resizing.

Padding mask contract
---------------------
padding_mask : (B, ph, pw) bool, True = real patch (same as existing collators,
               but at stride-4 granularity instead of stride-16).
               Propagated through mask-unit windows and temporal adapters.

Temporal adapters
-----------------
Two FactorizedTemporalAttention (or TemporalDepthwiseConv) modules are
injected between stages 2→3 and 3→4.  For T=1 they are identity no-ops.
For T>1 they mix context across frames.  When loaded from video Hiera the
weights can be warm; when loaded from image Hiera they start zero-initialized.
"""
from __future__ import annotations

import logging
import math
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .temporal_mixing import build_temporal_mixer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hiera-L architecture constants
# ---------------------------------------------------------------------------
HIERA_LARGE_EMBED_DIMS = [144, 288, 576, 1152]
HIERA_LARGE_DEPTHS     = [2, 3, 16, 3]
HIERA_LARGE_HEADS      = [2, 4, 8, 16]

# SAM2-Hiera-Large is deeper
SAM2_HIERA_LARGE_EMBED_DIMS = [144, 288, 576, 1152]
SAM2_HIERA_LARGE_DEPTHS     = [2, 6, 36, 4]
SAM2_HIERA_LARGE_HEADS      = [2, 4, 8, 16]


def _sam2_get_pos_embed_fixed(self, hw: tuple[int, int]) -> Tensor:
    """
    Absolute-Win position embedding for variable (H, W) token grids.

    HuggingFace's Sam2HieraDetModel tiles the window embedding with floor
    division (``h // window_h``), which only matches when H and W are exact
    multiples of the window size.  Native-aspect crops (e.g. 112×96 px →
    28×24 tokens) violate that and raise a size-mismatch RuntimeError.

    Tile with ceil, then crop to (h, w) per arxiv:2311.05613 §3.
    """
    h, w = hw
    window_embed = self.pos_embed_window
    pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
    _, _, wh, ww = window_embed.shape
    th = math.ceil(h / wh)
    tw = math.ceil(w / ww)
    tiled = window_embed.tile([1, 1, th, tw])[:, :, :h, :w]
    pos_embed = pos_embed + tiled
    return pos_embed.permute(0, 2, 3, 1)


def _patch_sam2_pos_embed(backbone: nn.Module) -> None:
    """Replace SAM2 Hiera _get_pos_embed with the crop-aware variant."""
    backbone._get_pos_embed = types.MethodType(_sam2_get_pos_embed_fixed, backbone)
    log.info("Patched SAM2 _get_pos_embed for variable-aspect native crops.")


def _sam2_multiscale_block_forward_fixed(self, hidden_states: Tensor, **kwargs) -> Tensor:
    """
    Sam2MultiScaleBlock forward with correct pad_hw after Q-pooling.

    HuggingFace's block recomputes ``pad_hw`` from ``residual.shape`` when
    ``query_stride`` is set, ignoring the padded grid from ``window_partition``.
    Native-aspect token grids (e.g. 30×32) then fail in ``window_unpartition``
    with ``shape '[B, …, ws, ws, -1]' is invalid for input of size …``.
    """
    from transformers.models.sam2.modeling_sam2 import (
        do_pool,
        window_partition,
        window_unpartition,
    )

    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)

    if self.dim != self.dim_out:
        residual = do_pool(self.proj(hidden_states), self.query_stride)

    window_size = self.window_size
    pad_hw: tuple[int, int] | None = None
    crop_hw: tuple[int, int] | None = None
    if self.window_size > 0:
        crop_hw = hidden_states.shape[1], hidden_states.shape[2]
        hidden_states, pad_hw = window_partition(hidden_states, window_size)

    hidden_states = self.attn(hidden_states=hidden_states, **kwargs)

    if self.query_stride:
        window_size = self.window_size // self.query_stride[0]
        crop_hw = residual.shape[1], residual.shape[2]
        if pad_hw is not None:
            pad_hw = _infer_sam2_unpartition_pad_hw(
                hidden_states.shape[0],
                residual.shape[0],
                window_size,
                crop_hw,
                pad_hw,
                self.query_stride,
            )

    if self.window_size > 0:
        assert pad_hw is not None and crop_hw is not None
        hidden_states = window_unpartition(hidden_states, window_size, pad_hw, crop_hw)

    hidden_states = residual + hidden_states
    layernorm_output = self.layer_norm2(hidden_states)
    hidden_states = hidden_states + self.mlp(layernorm_output)
    return hidden_states


def _infer_sam2_unpartition_pad_hw(
    n_windows: int,
    batch_size: int,
    window_size: int,
    crop_hw: tuple[int, int],
    partition_pad_hw: tuple[int, int],
    query_stride: tuple[int, int],
) -> tuple[int, int]:
    """
    Recover the padded (H, W) expected by ``window_unpartition`` after Q-pooling.

    Prefer scaling the grid from ``window_partition``; fall back to factoring the
    actual window count when aspect ratios are not exact multiples.
    """
    sy, sx = query_stride
    scaled = (partition_pad_hw[0] // sy, partition_pad_hw[1] // sx)
    n_per_img = max(1, n_windows // max(1, batch_size))
    if (scaled[0] // window_size) * (scaled[1] // window_size) == n_per_img:
        return scaled

    crop_h, crop_w = crop_hw
    pad_h = crop_h + (window_size - crop_h % window_size) % window_size
    pad_w = crop_w + (window_size - crop_w % window_size) % window_size
    if (pad_h // window_size) * (pad_w // window_size) == n_per_img:
        return pad_h, pad_w

    gh = max(1, int(round(math.sqrt(n_per_img * crop_h / max(crop_w, 1)))))
    while gh > 1 and n_per_img % gh != 0:
        gh -= 1
    gw = n_per_img // gh
    return gh * window_size, gw * window_size


def _patch_sam2_multiscale_blocks(backbone: nn.Module) -> None:
    """Patch every Sam2MultiScaleBlock for variable-aspect native crops."""
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return
    n = 0
    for block in blocks:
        if hasattr(block, "query_stride") and hasattr(block, "window_size"):
            block.forward = types.MethodType(_sam2_multiscale_block_forward_fixed, block)
            n += 1
    if n:
        log.info(
            "Patched %d SAM2 MultiScaleBlock forwards for variable-aspect native crops.",
            n,
        )


# ---------------------------------------------------------------------------
# Sinusoidal 2D position encoding (variable grid)
# ---------------------------------------------------------------------------

def _make_sinusoidal_2d(ph: int, pw: int, dim: int, device: torch.device) -> Tensor:
    """
    Create 2D sinusoidal position encoding for a (ph, pw) grid.

    Returns (ph*pw, dim).  dim must be even.
    """
    assert dim % 2 == 0
    half = dim // 2

    # row and column index grids
    row = torch.arange(ph, device=device).float()
    col = torch.arange(pw, device=device).float()
    row_g, col_g = torch.meshgrid(row, col, indexing="ij")  # (ph, pw)

    # frequency bands
    freq = 1.0 / (10000 ** (2 * torch.arange(half // 2, device=device).float() / half))

    row_enc = torch.cat(
        [torch.sin(row_g.flatten()[:, None] * freq[None]),
         torch.cos(row_g.flatten()[:, None] * freq[None])], dim=-1
    )  # (ph*pw, half)
    col_enc = torch.cat(
        [torch.sin(col_g.flatten()[:, None] * freq[None]),
         torch.cos(col_g.flatten()[:, None] * freq[None])], dim=-1
    )  # (ph*pw, half)

    return torch.cat([row_enc, col_enc], dim=-1)  # (ph*pw, dim)


# ---------------------------------------------------------------------------
# Padding mask helper: resize (B, ph, pw) bool masks through pooling stages
# ---------------------------------------------------------------------------

def _pool_padding_mask(pm: Tensor, stride: int = 2) -> Tensor:
    """
    Halve a (B, ph, pw) bool padding mask using max pooling.
    A merged 2×2 super-patch is valid if ANY sub-patch was valid.
    """
    B, ph, pw = pm.shape
    # Pad so dimensions are divisible by stride
    pad_h = (stride - ph % stride) % stride
    pad_w = (stride - pw % stride) % stride
    if pad_h or pad_w:
        pm = F.pad(pm.float(), (0, pad_w, 0, pad_h), value=0.0)
    else:
        pm = pm.float()
    out = F.max_pool2d(
        pm.unsqueeze(1), kernel_size=stride, stride=stride
    ).squeeze(1).bool()
    return out


def _infer_stage1_token_grid(
    n_tokens: int,
    h_px: int,
    w_px: int,
    *,
    patch_stride: int = 4,
    padding_mask: Optional[Tensor] = None,
) -> tuple[int, int]:
    """
    Infer (ph, pw) for stage-1 tokens from input pixels and/or padding mask.

    Native-aspect crops (e.g. 112×96 px → 28×24 tokens) are not square; using
    ``isqrt(N)`` alone produces the wrong grid and breaks spatial PE addition.
    """
    ph = h_px // patch_stride
    pw = w_px // patch_stride
    if ph * pw == n_tokens:
        return ph, pw

    # Hiera patch embed: Conv2d(3, D, kernel=7, stride=4, padding=3)
    ph = (h_px + 2 * 3 - 7) // patch_stride + 1
    pw = (w_px + 2 * 3 - 7) // patch_stride + 1
    if ph * pw == n_tokens:
        return ph, pw

    if padding_mask is not None and padding_mask.dim() == 3:
        pm_ph, pm_pw = padding_mask.shape[1], padding_mask.shape[2]
        if pm_ph * pm_pw == n_tokens:
            return pm_ph, pm_pw

    for candidate_ph in range(int(math.isqrt(n_tokens)), 0, -1):
        if n_tokens % candidate_ph == 0:
            return candidate_ph, n_tokens // candidate_ph

    raise RuntimeError(
        f"Cannot infer stage-1 token grid: N={n_tokens}, "
        f"input=({h_px}, {w_px}), patch_stride={patch_stride}"
    )


def _align_padding_mask_grid(pm: Tensor, n_tokens: int) -> Tensor:
    """
    Resize a (B, ph, pw) valid-patch mask to match a feature token grid.

    Input masks arrive at student stride-4 from StudentMixedCollator; each
    Hiera stage has a different token grid (SAM2 query pooling).  This is the
    only place stage F1–F4 masks are derived.
    """
    B = pm.shape[0]
    side = int(math.isqrt(n_tokens))
    if side * side == n_tokens:
        if pm.shape[1] == side and pm.shape[2] == side:
            return pm
        pm_r = F.interpolate(pm.float().unsqueeze(1), size=(side, side), mode="nearest")
        return pm_r.squeeze(1).bool()
    flat = pm.reshape(B, -1).float().unsqueeze(1)
    flat = F.interpolate(flat, size=n_tokens, mode="nearest")
    return flat.squeeze(1).bool().reshape(B, 1, n_tokens)


# ---------------------------------------------------------------------------
# Core backbone wrapper
# ---------------------------------------------------------------------------

class HieraStudentBackbone(nn.Module):
    """
    Hiera-Large backbone with temporal adapters.

    Parameters
    ----------
    hiera_variant     : "hiera_large_video_mae_k400" | "sam2_hiera_large"
    temporal_mixing   : "factorized_attn" | "depthwise_conv"
    temporal_adapter_stages : which inter-stage gaps get adapters, e.g. [2, 3]
    max_frames        : max temporal positions for learned temporal pos enc
    trainable_stages  : last N Hiera stages to unfreeze (None = all)
    hf_cache_dir      : HuggingFace cache directory override
    """

    PATCH_STRIDE = 4   # Hiera Conv2d stride=4 → effective patch size

    def __init__(
        self,
        hiera_variant: str = "hiera_large_video_mae_k400",
        temporal_mixing: str = "factorized_attn",
        temporal_adapter_stages: list[int] | None = None,
        max_frames: int = 32,
        trainable_stages: int | None = None,
        hf_cache_dir: Optional[str] = None,
        align_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hiera_variant          = hiera_variant
        self.temporal_mixing        = temporal_mixing
        self.temporal_adapter_stages = temporal_adapter_stages or [2, 3]
        self.max_frames             = max_frames
        self.trainable_stages       = trainable_stages
        self.align_dim              = align_dim

        # Load the pretrained Hiera backbone
        self.hiera, self.embed_dims = self._load_hiera(hiera_variant, hf_cache_dir)
        self.hidden_size = self.embed_dims[-1]

        # Project student features into the shared teacher alignment space
        if align_dim is not None:
            self.global_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, align_dim, bias=False),
            )
            self.patch_proj = nn.Sequential(
                nn.LayerNorm(self.embed_dims[0]),
                nn.Linear(self.embed_dims[0], align_dim, bias=False),
            )
            self.tube_proj = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, align_dim, bias=False),
            )
        else:
            self.global_proj = self.patch_proj = self.tube_proj = None

        # Freeze backbone; selectively unfreeze trainable_stages later
        self._freeze_hiera(trainable_stages)

        # Temporal position encoding (learned, one vector per frame slot)
        # index 0 is used for T=1 images
        self.temporal_pos_enc = nn.Embedding(max_frames, self.embed_dims[0])
        nn.init.normal_(self.temporal_pos_enc.weight, std=0.02)

        # Temporal adapters — inserted at configured inter-stage positions
        self.temporal_adapters = nn.ModuleDict()
        adapter_dims = {
            2: self.embed_dims[1],  # after stage 2
            3: self.embed_dims[2],  # after stage 3
        }
        for stage_idx in self.temporal_adapter_stages:
            dim = adapter_dims.get(stage_idx, self.embed_dims[min(stage_idx, 3)])
            num_heads = max(1, dim // 64)
            self.temporal_adapters[f"stage{stage_idx}"] = build_temporal_mixer(
                temporal_mixing, dim, num_heads=num_heads, zero_init=True
            )

        log.info(
            f"HieraStudentBackbone: variant={hiera_variant}, "
            f"embed_dims={self.embed_dims}, "
            f"temporal_adapter_stages={self.temporal_adapter_stages}, "
            f"trainable_stages={trainable_stages}"
        )

    # ------------------------------------------------------------------
    # Loader helpers
    # ------------------------------------------------------------------

    def _load_hiera(
        self,
        variant: str,
        hf_cache_dir: Optional[str],
    ) -> tuple[nn.Module, list[int]]:
        if variant == "hiera_large_video_mae_k400":
            return self._load_hiera_video_k400(hf_cache_dir)
        elif variant == "sam2_hiera_large":
            return self._load_sam2_hiera(hf_cache_dir)
        else:
            raise ValueError(
                f"Unknown hiera_variant: {variant!r}. "
                "Choose 'hiera_large_video_mae_k400' or 'sam2_hiera_large'."
            )

    def _load_hiera_video_k400(
        self, hf_cache_dir: Optional[str]
    ) -> tuple[nn.Module, list[int]]:
        """
        Load Video Hiera-L (MAE K400) via the hiera-transformer package.

        pip install hiera-transformer
        """
        try:
            from hiera import Hiera as HieraModel
        except ImportError:
            raise ImportError(
                "The 'hiera-transformer' package is required for hiera_large_video_mae_k400. "
                "Install it with: pip install hiera-transformer"
            )
        log.info("Loading Video Hiera-L (mae_k400) from facebook/hiera_large_16x224.mae_k400 ...")
        model = HieraModel.from_pretrained("facebook/hiera_large_16x224.mae_k400")
        model.eval()
        return model, HIERA_LARGE_EMBED_DIMS

    def _load_sam2_hiera(
        self, hf_cache_dir: Optional[str]
    ) -> tuple[nn.Module, list[int]]:
        """
        Load SAM2.1-Hiera-Large image encoder via HuggingFace transformers.
        Extracts only the Hiera backbone (drops SAM2 neck + prompt encoder).
        """
        from transformers import AutoModel
        from models.hf_loading import load_pretrained
        log.info("Loading SAM2.1-Hiera-Large backbone from facebook/sam2.1-hiera-large ...")
        sam2 = load_pretrained(
            AutoModel,
            "facebook/sam2.1-hiera-large",
            hf_cache_dir=hf_cache_dir,
            trust_remote_code=True,
        )
        # Sam2Model.vision_encoder is Sam2VisionModel; the Hiera trunk is .backbone
        vision = getattr(sam2, "vision_encoder", None) or getattr(sam2, "image_encoder", None)
        if vision is None:
            raise RuntimeError(
                "Could not locate Hiera backbone in the SAM2 model. "
                "Expected attribute 'vision_encoder' or 'image_encoder'."
            )
        backbone = getattr(vision, "backbone", None) or getattr(vision, "trunk", None) or vision
        if hasattr(backbone, "pos_embed_window"):
            _patch_sam2_pos_embed(backbone)
            _patch_sam2_multiscale_blocks(backbone)
        backbone.eval()
        return backbone, SAM2_HIERA_LARGE_EMBED_DIMS

    def _hiera_frame_chunk_size(self, b: int, t: int) -> int:
        """
        SAM2 Hiera window attention breaks when many frames are batched together
        on native-aspect crops (residual/hidden_states batch dim mismatch).
        Process one frame per Hiera call for video; image (T=1) stays batched.
        """
        if self.hiera_variant == "sam2_hiera_large" and t > 1:
            return 1
        return b * t

    def _hiera_forward_stages_chunked(
        self, x: Tensor, b: int, t: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run ``_hiera_forward_stages`` in sub-batches along the image dimension."""
        chunk = self._hiera_frame_chunk_size(b, t)
        if x.shape[0] <= chunk:
            return self._hiera_forward_stages(x)
        parts: list[list[Tensor]] = [[], [], [], []]
        for i in range(0, x.shape[0], chunk):
            x_chunk = x[i : i + chunk]
            stage_feats = self._hiera_forward_stages(x_chunk)
            b_exp = x_chunk.shape[0]
            for j, feat in enumerate(stage_feats):
                parts[j].append(feat[:b_exp])
        return tuple(torch.cat(p, dim=0) for p in parts)

    def _freeze_hiera(self, trainable_stages: Optional[int]) -> None:
        """
        Freeze all Hiera parameters, then selectively unfreeze the last
        `trainable_stages` stage blocks + patch embedding.
        """
        for p in self.hiera.parameters():
            p.requires_grad_(False)

        if trainable_stages is None:
            for p in self.hiera.parameters():
                p.requires_grad_(True)
            return

        if trainable_stages <= 0:
            return  # everything frozen

        # Try to unfreeze the last N stages via common Hiera attribute names
        stage_attrs = ["blocks", "stages", "encoder_layers"]
        for attr in stage_attrs:
            stages = getattr(self.hiera, attr, None)
            if stages is not None and hasattr(stages, "__len__"):
                for stage in list(stages)[-trainable_stages:]:
                    for p in stage.parameters():
                        p.requires_grad_(True)
                log.info(f"  Unfroze last {trainable_stages} Hiera stages.")
                return
        # Fallback: unfreeze all if we can't identify stages
        log.warning("Could not identify Hiera stage structure; unfreezing all Hiera params.")
        for p in self.hiera.parameters():
            p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Position encoding helpers
    # ------------------------------------------------------------------

    def _spatial_pos_enc(self, ph: int, pw: int, dim: int, device: torch.device) -> Tensor:
        """
        On-the-fly 2D sinusoidal spatial position encoding.
        Returns (1, ph*pw, dim) for broadcasting over batch.
        """
        enc = _make_sinusoidal_2d(ph, pw, dim, device)
        return enc.unsqueeze(0)  # (1, N, dim)

    def _temporal_pos_enc(self, T: int, device: torch.device) -> Tensor:
        """
        Learned temporal position encoding.
        Returns (1, T, 1, D) for broadcasting over (B, T, N, D).

        When T exceeds max_frames (e.g. EchoNet loads 32 frames but the student
        was trained with max_frames=16), subsample uniform indices into the
        learned table instead of indexing out of bounds.
        """
        if T <= self.max_frames:
            idx = torch.arange(T, device=device)
        else:
            idx = torch.linspace(0, self.max_frames - 1, T, device=device).long()
        enc = self.temporal_pos_enc(idx)        # (T, D)
        return enc.unsqueeze(0).unsqueeze(2)    # (1, T, 1, D)

    # ------------------------------------------------------------------
    # Forward: try Hiera API variants
    # ------------------------------------------------------------------

    @staticmethod
    def _feat_to_tokens(f: Tensor) -> Tensor:
        """(B, H, W, D) or (B, N, D) → (B, N, D)."""
        if f.dim() == 4:
            b, h, w, d = f.shape
            return f.reshape(b, h * w, d)
        return f

    @staticmethod
    def _unwrap_hiera_output(out) -> Tensor | tuple[Tensor, ...] | list[Tensor]:
        """Normalize ModelOutput / tuple returns from assorted Hiera wrappers."""
        if hasattr(out, "intermediate_hidden_states") and out.intermediate_hidden_states:
            return out.intermediate_hidden_states
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state
        if hasattr(out, "fpn_hidden_states") and out.fpn_hidden_states:
            return out.fpn_hidden_states
        if isinstance(out, (list, tuple)):
            return out[0] if len(out) == 1 else out
        return out

    def _hiera_forward_stages(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Run the Hiera backbone and collect intermediate feature maps.

        Returns (f1, f2, f3, f4) each of shape (B*T, N_s, D_s).

        Hiera exposes intermediate features in several ways depending on
        the version. We try the most common APIs in order.
        """
        intermediates: list[Tensor] = []

        # ── API 0: SAM2 Hiera trunk (Sam2HieraDetModel) ─────────────────
        if hasattr(self.hiera, "blocks") and hasattr(self.hiera, "stage_ends"):
            try:
                out = self.hiera(x, return_dict=True)
            except TypeError:
                out = self.hiera(x)
            stage_feats = getattr(out, "intermediate_hidden_states", None)
            if stage_feats is not None and len(stage_feats) == 4:
                b_in = x.shape[0]
                out_feats: list[Tensor] = []
                for f in stage_feats:
                    tokens = self._feat_to_tokens(f)
                    if tokens.shape[0] > b_in:
                        tokens = tokens[:b_in]
                    elif tokens.shape[0] < b_in:
                        raise RuntimeError(
                            f"SAM2 Hiera returned batch {tokens.shape[0]} for input "
                            f"batch {b_in} (stage tokens {tokens.shape})."
                        )
                    out_feats.append(tokens)
                return tuple(out_feats)

        # ── API 1: hiera-transformer package `Hiera` class ──────────────
        if hasattr(self.hiera, "get_intermediate_layers"):
            # Returns list of tensors, one per stage
            feats = self.hiera.get_intermediate_layers(
                x, n=4, return_class_token=False
            )
            # feats may be a list of (B, N, D) or (B, H, W, D)
            for f in feats:
                if f.dim() == 4:
                    B2, H, W, D = f.shape
                    f = f.reshape(B2, H * W, D)
                intermediates.append(f)
            if len(intermediates) == 4:
                return tuple(intermediates)

        # ── API 2: forward_features with intermediates flag ─────────────
        if hasattr(self.hiera, "forward_features"):
            try:
                out = self.hiera.forward_features(
                    x, return_intermediates=True
                )
                if isinstance(out, (list, tuple)) and len(out) >= 4:
                    feats = out[:4]
                    result = []
                    for f in feats:
                        if f.dim() == 4:
                            B2, H, W, D = f.shape
                            f = f.reshape(B2, H * W, D)
                        result.append(f)
                    return tuple(result)
            except TypeError:
                pass

        # ── API 3: register forward hooks on stage modules ───────────────
        # Fallback: hook into stage outputs
        stage_outputs: list[Tensor] = []

        def _hook(module, inp, out):
            o = out[0] if isinstance(out, (list, tuple)) else out
            if o.dim() == 4:
                B2, H, W, D = o.shape
                o = o.reshape(B2, H * W, D)
            stage_outputs.append(o.detach() if not o.requires_grad else o)

        stage_candidates = []
        for attr in ["blocks", "stages", "encoder_layers", "layers"]:
            cands = getattr(self.hiera, attr, None)
            if cands is not None and hasattr(cands, "__len__") and len(cands) == 4:
                stage_candidates = list(cands)
                break

        if stage_candidates:
            handles = [s.register_forward_hook(_hook) for s in stage_candidates]
            _ = self.hiera(x)
            for h in handles:
                h.remove()
            if len(stage_outputs) == 4:
                return tuple(stage_outputs)

        # ── API 4: plain forward, replicate from final hidden state ─────
        # Last resort: run plain forward and tile the same tensor 4× at
        # different spatial resolutions via avg-pooling approximation.
        log.warning(
            "Could not extract 4 intermediate Hiera stages via known APIs. "
            "Using plain forward + spatial downsampling fallback. "
            "Consider updating the hiera-transformer package."
        )
        out = self.hiera(x, return_dict=True) if hasattr(self.hiera, "blocks") else self.hiera(x)
        out = self._unwrap_hiera_output(out)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        if not isinstance(out, Tensor):
            raise RuntimeError(f"Unexpected Hiera output type: {type(out)}")
        if out.dim() == 3:
            B2, N, D = out.shape
            side = int(math.isqrt(N))
        else:
            raise RuntimeError(f"Unexpected Hiera output shape: {out.shape}")

        # Build approximate F1-F4 by spatial pooling
        feat_2d = out.reshape(B2, side, side, D).permute(0, 3, 1, 2)
        f4 = out
        f3 = F.avg_pool2d(feat_2d, 1).permute(0, 2, 3, 1).reshape(B2, -1, D)
        f2 = F.avg_pool2d(feat_2d, 2, stride=2).permute(0, 2, 3, 1)
        f2 = f2.reshape(B2, -1, f2.shape[-1])
        # f2 has half the channels — project to D2
        f1 = F.avg_pool2d(feat_2d, 4, stride=4).permute(0, 2, 3, 1)
        f1 = f1.reshape(B2, -1, f1.shape[-1])
        return f1, f2, f3, f4

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,                             # (B, T, 3, H, W)
        padding_mask: Optional[Tensor] = None, # (B, ph, pw) bool
        with_patch_proj: bool = True,          # False for video SSL (F1 proj is ~16× F4)
    ) -> dict:
        """
        Parameters
        ----------
        x            : (B, T, 3, H, W)  — T=1 for images, T>1 for videos
        padding_mask : (B, ph, pw) bool — True = real patch (stride-4 grid)

        Returns
        -------
        dict with keys:
          global  : (B, D4)          temporal + spatial mean pool of F4
          F1      : (B, T, N1, D1)   highest-res features (stride 4)
          F2      : (B, T, N2, D2)   stride 8
          F3      : (B, T, N3, D3)   stride 16
          F4      : (B, T, N4, D4)   stride 32
          dense   : (B, T, N4, D4)   alias for F4
          pmasks  : list of 4 (B, ph_i, pw_i) bool padding masks per stage
        """
        B, T, C, H, W = x.shape

        # ── Flatten temporal dimension for per-frame Hiera ───────────────
        x_2d = x.flatten(0, 1)                      # (B*T, 3, H, W)

        # ── Run Hiera stages 1–4, collecting intermediate features ──────
        f1_bt, f2_bt, f3_bt, f4_bt = self._hiera_forward_stages_chunked(x_2d, B, T)
        # Each fi_bt: (B*T, Ni, Di)

        # ── Align padding masks to actual per-stage token grids ─────────
        if padding_mask is not None:
            pm1 = _align_padding_mask_grid(padding_mask, f1_bt.shape[1])
            pm2 = _align_padding_mask_grid(padding_mask, f2_bt.shape[1])
            pm3 = _align_padding_mask_grid(padding_mask, f3_bt.shape[1])
            pm4 = _align_padding_mask_grid(padding_mask, f4_bt.shape[1])
        else:
            pm1 = pm2 = pm3 = pm4 = None

        # ── Add spatial position encoding to stage-1 tokens ─────────────
        N1 = f1_bt.shape[1]
        ph1, pw1 = _infer_stage1_token_grid(
            N1, H, W, patch_stride=self.PATCH_STRIDE, padding_mask=padding_mask
        )
        spe = self._spatial_pos_enc(ph1, pw1, f1_bt.shape[-1], f1_bt.device)
        f1_bt = f1_bt + spe

        # ── Reshape to (B, T, N, D) ──────────────────────────────────────
        def _unflatten(f: Tensor) -> Tensor:
            BT, N, D = f.shape
            return f.reshape(B, T, N, D)

        F1 = _unflatten(f1_bt)
        F2 = _unflatten(f2_bt)
        F3 = _unflatten(f3_bt)
        F4 = _unflatten(f4_bt)

        # ── Add temporal position encoding (stage 1 level) ───────────────
        tpe = self._temporal_pos_enc(T, F1.device)  # (1, T, 1, D1)
        if tpe.shape[-1] == F1.shape[-1]:
            F1 = F1 + tpe

        # ── Apply temporal adapters ───────────────────────────────────────
        # After stage 2: mix temporal context then continue to stage 3
        if "stage2" in self.temporal_adapters:
            pm2_flat = pm2.reshape(B, -1)[:, :F2.shape[2]] if pm2 is not None else None
            F2 = self.temporal_adapters["stage2"](F2, pm2_flat)
            # Fold back to (B*T, N2, D2) for any remaining Hiera ops
            # (none here since we already ran all 4 stages above, but
            # keeping the structure for future inter-stage insertion)

        # After stage 3: mix temporal context
        if "stage3" in self.temporal_adapters:
            pm3_flat = pm3.reshape(B, -1)[:, :F3.shape[2]] if pm3 is not None else None
            F3 = self.temporal_adapters["stage3"](F3, pm3_flat)

        # ── Global token: temporal + spatial mean pool of F4 ─────────────
        if pm4 is not None:
            n4 = F4.shape[2]
            pm4_flat = pm4.reshape(B, -1)[:, :n4].reshape(B, 1, n4, 1).float()
            masked_F4 = F4 * pm4_flat
            denom = pm4_flat.sum(dim=2, keepdim=True).clamp(min=1)
            global_token = (masked_F4 / denom).sum(dim=(1, 2))  # (B, D4)
        else:
            global_token = F4.mean(dim=(1, 2))  # (B, D4)

        out = {
            "global": global_token,
            "F1":     F1,
            "F2":     F2,
            "F3":     F3,
            "F4":     F4,
            "dense":  F4,          # alias used by loss functions
            "pmasks": [pm1, pm2, pm3, pm4],
        }
        if self.global_proj is not None:
            out["global_proj"] = self.global_proj(global_token)
            if with_patch_proj:
                out["patch_proj"] = self.patch_proj(F1)
            out["tube_proj"] = self.tube_proj(F4)
        return out

    def parameters_for_ema(self):
        """
        Legacy/compatibility API for dual-branch callers only.

        The student pipeline uses ema_update() with full named_parameters()
        matching — do not call this from the Hiera student trainer.
        """
        if getattr(self, "is_ema_target", False):
            yield from self.parameters()
        else:
            for p in self.parameters():
                if p.requires_grad:
                    yield p
