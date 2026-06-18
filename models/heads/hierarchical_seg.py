"""
models/heads/hierarchical_seg.py  ·  UPerNet-style hierarchical segmentation decoder
=====================================================================================

UPerNetDecoder
--------------
Consumes the four multi-scale feature maps produced by HieraStudentBackbone:

  F1 : (B, T, N1, D1)   stride-4   highest resolution
  F2 : (B, T, N2, D2)   stride-8
  F3 : (B, T, N3, D3)   stride-16
  F4 : (B, T, N4, D4)   stride-32  lowest resolution

For each input frame (or selected frames for video):
  1. (Optional) Per-scale SegAdapters re-calibrate frozen token features
  2. Project each Fi to fpn_channels with 1×1 conv
  3. FPN top-down path: upsample coarser maps to each finer stage's grid, then
     attention-gate the fine features before merging (suppresses background noise)
  4. Fuse all four scales at F1 resolution with ASPP (multi-dilated context capture)
  5. Final 1×1 conv → n_classes logits

Padding-mask aware at every upsampling step: logits at padded positions
are zeroed so they don't contribute to loss.

For video input (T>1):
  - Default: process every frame and return (B, T, C, ph1, pw1)
  - Sparse mode: process only frames with valid labels (set by supervised loss)

build_hierarchical_seg_head(embed_dims, n_classes, fpn_channels) is the
factory used by student_phase_steps.py and finetune experiments.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class _ConvBnGelu(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, kernel: int = 3, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )


def _factor_token_grid(n_tokens: int, grid_hint: Optional[Tensor] = None) -> tuple[int, int]:
    """Return (ph, pw) with ph * pw == n_tokens for spatial token layouts."""
    if grid_hint is not None and grid_hint.dim() >= 2:
        ph, pw = int(grid_hint.shape[-2]), int(grid_hint.shape[-1])
        if ph * pw == n_tokens:
            return ph, pw
    n_tokens = max(1, n_tokens)
    ph = max(1, int(n_tokens ** 0.5))
    while ph > 1 and n_tokens % ph != 0:
        ph -= 1
    return ph, n_tokens // ph


# ---------------------------------------------------------------------------
# SegAdapter — lightweight residual MLP applied to token sequences
# ---------------------------------------------------------------------------

class SegAdapter(nn.Module):
    """
    Lightweight bottleneck adapter applied to frozen backbone token features.

    Wraps a down-projection → GELU → up-projection with a residual connection.
    The up-projection is zero-initialised so training starts from the unmodified
    backbone features (no regression risk).

    Parameters
    ----------
    dim        : feature dimension of the token sequence (Di for stage i)
    bottleneck : inner dim; defaults to dim // 4
    """

    def __init__(self, dim: int, bottleneck: Optional[int] = None):
        super().__init__()
        inner = bottleneck if bottleneck is not None else max(1, dim // 4)
        self.down = nn.Linear(dim, inner, bias=False)
        self.up   = nn.Linear(inner, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, N, D)  →  (B, N, D)"""
        return x + self.up(F.gelu(self.down(x)))


# ---------------------------------------------------------------------------
# ASPPFusion — multi-dilated context capture to replace the simple fusion conv
# ---------------------------------------------------------------------------

class ASPPFusion(nn.Module):
    """
    ASPP-style fusion block (DeepLab-V3 design) that replaces the simple
    concat → 1×1 → 3×3 neck used in the original UPerNetDecoder.

    Takes concatenated multi-scale features (in_c = fpn * 4) and produces
    an out_c feature map with multi-scale receptive fields.

    Parallel branches:
      - 1×1 conv (local)
      - 3×3 dilated d=6
      - 3×3 dilated d=12
      - 3×3 dilated d=18
      - Global average pool + 1×1 + bilinear upsample (image-level context)

    All branches → concat → 1×1 bottleneck → BN → GELU → 3×3 refinement.

    Parameters
    ----------
    in_c      : input channels (fpn_channels * 4 by default)
    out_c     : output channels (fpn_channels)
    dilations : list of 4 dilation rates; first is treated as 1×1
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        dilations: List[int] = (1, 6, 12, 18),
    ):
        super().__init__()
        # Internal bottleneck per branch keeps parameter count manageable
        branch_c = out_c

        self.branch1x1 = _ConvBnGelu(in_c, branch_c, kernel=1, padding=0)

        self.branches_dilated = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, branch_c, kernel_size=3,
                          padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(branch_c),
                nn.GELU(),
            )
            for d in dilations[1:]   # skip d=1 (handled by branch1x1)
        ])

        # Global average pool branch for image-level context
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, branch_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_c),
            nn.GELU(),
        )

        # 1 (1×1) + len(dilations)-1 (dilated) + 1 (global) branches
        n_branches = 1 + len(dilations) - 1 + 1
        self.bottleneck = nn.Sequential(
            nn.Conv2d(branch_c * n_branches, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            _ConvBnGelu(out_c, out_c),
        )

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        b1 = self.branch1x1(x)
        branches = [b1] + [b(x) for b in self.branches_dilated]
        # Global branch: pool to 1×1, upsample back to feature map size
        glob = self.global_branch(x)
        glob = F.interpolate(glob, size=(h, w), mode="bilinear", align_corners=False)
        branches.append(glob)
        return self.bottleneck(torch.cat(branches, dim=1))


# ---------------------------------------------------------------------------
# SubPixelRefinement — learned 2× upsample from stride-4 to stride-2
# ---------------------------------------------------------------------------

class SubPixelRefinement(nn.Module):
    """
    Lightweight sub-pixel refinement head that doubles the spatial resolution
    of decoder logits (stride-4 → stride-2) using a learned ConvTranspose2d path.

    Architecture:
      fused_features (fpn, h, w)  →  ConvTranspose2d 2×  →  (fpn//2, 2h, 2w)
      coarse_logits  (n, h, w)    →  bilinear 2×          →  (n, 2h, 2w)
      concat both branches        →  3×3 conv → BN → GELU  →  (fpn//4, 2h, 2w)
                                  →  1×1 conv              →  (n_classes, 2h, 2w)

    Residual: refined logits = fine_logits + coarse_logits_2x (stabilises training).

    Parameters
    ----------
    fpn_channels : number of FPN feature channels entering this module
    n_classes    : number of output segmentation classes
    """

    def __init__(self, fpn_channels: int, n_classes: int):
        super().__init__()
        inner = max(fpn_channels // 2, 32)
        fine  = max(fpn_channels // 4, 16)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(fpn_channels, inner, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(inner),
            nn.GELU(),
        )
        self.refine = _ConvBnGelu(inner + n_classes, fine)
        self.out    = nn.Conv2d(fine, n_classes, kernel_size=1)

    def forward(self, fused: Tensor, coarse_logits: Tensor) -> Tensor:
        """
        Parameters
        ----------
        fused        : (B, fpn, h, w)      fused features from decoder neck
        coarse_logits: (B, n_classes, h, w) logits at stride-4

        Returns
        -------
        (B, n_classes, 2h, 2w)   refined logits at stride-2
        """
        fine_feat = self.up(fused)                                               # (B, fpn//2, 2h, 2w)
        coarse_up = F.interpolate(coarse_logits, scale_factor=2,
                                  mode="bilinear", align_corners=False)          # (B, n, 2h, 2w)
        x = torch.cat([fine_feat, coarse_up], dim=1)
        return self.out(self.refine(x)) + coarse_up                              # residual connection


# ---------------------------------------------------------------------------
# UPerNetDecoder
# ---------------------------------------------------------------------------

class UPerNetDecoder(nn.Module):
    """
    UPerNet-style FPN decoder for hierarchical segmentation.

    Parameters
    ----------
    embed_dims          : list of 4 ints, e.g. [144, 288, 576, 1152] for Hiera-L
    n_classes           : number of output segmentation classes (1 for binary)
    fpn_channels        : internal FPN channel width (default 256)
    use_adapters        : if True, add per-scale SegAdapters before lateral projections
    use_attention_gates : if True, attention-gate fine features in FPN top-down path
    use_aspp            : if True, replace simple fusion conv with ASPP block
    use_refine_up       : if True, add a SubPixelRefinement head (stride-4 → stride-2)
                          for higher-fidelity boundary reconstruction
    """

    def __init__(
        self,
        embed_dims: List[int],
        n_classes: int = 1,
        fpn_channels: int = 256,
        use_adapters: bool = True,
        use_attention_gates: bool = True,
        use_aspp: bool = True,
        use_refine_up: bool = False,
    ):
        super().__init__()
        assert len(embed_dims) == 4, "embed_dims must have exactly 4 entries (one per stage)"
        self.embed_dims          = embed_dims
        self.n_classes           = n_classes
        self.fpn_channels        = fpn_channels
        self.use_adapters        = use_adapters
        self.use_attention_gates = use_attention_gates
        self.use_aspp            = use_aspp
        self.use_refine_up       = use_refine_up

        # ------------------------------------------------------------------
        # Track C: Per-scale SegAdapters (zero-init → safe residual start)
        # ------------------------------------------------------------------
        if use_adapters:
            self.adapters = nn.ModuleList([
                SegAdapter(d) for d in embed_dims
            ])

        # Lateral projections: Di → fpn_channels
        self.lateral = nn.ModuleList([
            nn.Conv2d(d, fpn_channels, kernel_size=1, bias=False)
            for d in embed_dims
        ])

        # Top-down 3×3 convs after upsampling + lateral merge
        self.td_convs = nn.ModuleList([
            _ConvBnGelu(fpn_channels, fpn_channels)
            for _ in range(3)   # stages 4→3, 3→2, 2→1
        ])

        # ------------------------------------------------------------------
        # Track A2: Attention gates on FPN skip connections
        # Each gate predicts a per-channel weight from the coarser (semantic)
        # feature to suppress background in the finer (high-res) feature.
        # ------------------------------------------------------------------
        if use_attention_gates:
            self.gates = nn.ModuleList([
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1, bias=True)
                for _ in range(3)   # gates for p3, p2, p1
            ])

        # Per-scale fusion convs (applied before concat)
        self.scale_convs = nn.ModuleList([
            _ConvBnGelu(fpn_channels, fpn_channels)
            for _ in range(4)
        ])

        # ------------------------------------------------------------------
        # Track A1: ASPP fusion or plain conv fusion
        # ------------------------------------------------------------------
        if use_aspp:
            self.fusion = ASPPFusion(fpn_channels * 4, fpn_channels)
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(fpn_channels * 4, fpn_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.GELU(),
                _ConvBnGelu(fpn_channels, fpn_channels),
            )

        self.head = nn.Conv2d(fpn_channels, n_classes, kernel_size=1)

        # ------------------------------------------------------------------
        # Track A3 (optional): Sub-pixel refinement head
        # ------------------------------------------------------------------
        if use_refine_up:
            self.refine_head = SubPixelRefinement(fpn_channels, n_classes)

    def _tokens_to_grid(
        self,
        tokens: Tensor,
        grid_hint: Optional[Tensor] = None,
    ) -> Tensor:
        """
        (B, N, D) → (B, D, h, w)   where N = h*w (rectangular grids supported).
        """
        B, N, D = tokens.shape
        ph, pw = _factor_token_grid(N, grid_hint)
        return tokens.reshape(B, ph, pw, D).permute(0, 3, 1, 2).contiguous()

    def _pool_pmask(self, pm: Tensor, target_h: int, target_w: int) -> Tensor:
        """
        Resize a (B, ph, pw) bool padding mask to (B, target_h, target_w).
        Uses nearest-neighbour interpolation to avoid fractional masks.
        """
        if pm.shape[1] == target_h and pm.shape[2] == target_w:
            return pm
        return F.interpolate(
            pm.unsqueeze(1).float(),
            size=(target_h, target_w),
            mode="nearest",
        ).squeeze(1).bool()

    def forward_single_frame(
        self,
        f1: Tensor,                             # (B, N1, D1)
        f2: Tensor,                             # (B, N2, D2)
        f3: Tensor,                             # (B, N3, D3)
        f4: Tensor,                             # (B, N4, D4)
        padding_mask: Optional[Tensor] = None,  # (B, ph1, pw1)
        stage_masks: Optional[List[Optional[Tensor]]] = None,
    ) -> Tensor:
        """
        Decode a single frame's multi-scale features to segmentation logits.

        Returns (B, n_classes, ph1, pw1).
        """
        hints = stage_masks or [padding_mask, None, None, None]

        # (Optional) adapt frozen features before lateral projection
        if self.use_adapters:
            f1 = self.adapters[0](f1)
            f2 = self.adapters[1](f2)
            f3 = self.adapters[2](f3)
            f4 = self.adapters[3](f4)

        # Convert tokens to spatial grids
        c1 = self._tokens_to_grid(f1, grid_hint=hints[0] if len(hints) > 0 else None)
        c2 = self._tokens_to_grid(f2, grid_hint=hints[1] if len(hints) > 1 else None)
        c3 = self._tokens_to_grid(f3, grid_hint=hints[2] if len(hints) > 2 else None)
        c4 = self._tokens_to_grid(f4, grid_hint=hints[3] if len(hints) > 3 else None)

        h1, w1 = c1.shape[2], c1.shape[3]

        # Lateral projections
        p1 = self.lateral[0](c1)  # (B, fpn, h1, w1)
        p2 = self.lateral[1](c2)  # (B, fpn, h2, w2)
        p3 = self.lateral[2](c3)
        p4 = self.lateral[3](c4)

        # FPN top-down: upsample to each stage's native grid.
        # With attention gates enabled, the coarser feature generates a
        # per-channel sigmoid mask that suppresses background in the finer map
        # before the element-wise merge and refine conv.
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_attention_gates:
            g3 = torch.sigmoid(self.gates[0](p4_up))
            p3 = self.td_convs[0](p3 * g3 + p4_up)
        else:
            p3 = self.td_convs[0](p3 + p4_up)

        p3_up = F.interpolate(p3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_attention_gates:
            g2 = torch.sigmoid(self.gates[1](p3_up))
            p2 = self.td_convs[1](p2 * g2 + p3_up)
        else:
            p2 = self.td_convs[1](p2 + p3_up)

        p2_up = F.interpolate(p2, size=p1.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_attention_gates:
            g1 = torch.sigmoid(self.gates[2](p2_up))
            p1 = self.td_convs[2](p1 * g1 + p2_up)
        else:
            p1 = self.td_convs[2](p1 + p2_up)

        # Upsample all to h1×w1 and apply per-scale fusion conv
        s1 = self.scale_convs[0](p1)

        s2 = F.interpolate(p2, size=(h1, w1), mode="bilinear", align_corners=False)
        s2 = self.scale_convs[1](s2)

        s3 = F.interpolate(p3, size=(h1, w1), mode="bilinear", align_corners=False)
        s3 = self.scale_convs[2](s3)

        s4 = F.interpolate(p4, size=(h1, w1), mode="bilinear", align_corners=False)
        s4 = self.scale_convs[3](s4)

        # ASPP fusion (or plain conv fusion) + predict
        fused       = torch.cat([s1, s2, s3, s4], dim=1)  # (B, fpn*4, h1, w1)
        fused       = self.fusion(fused)
        coarse_logits = self.head(fused)                   # (B, n_classes, h1, w1)

        if self.use_refine_up:
            # Sub-pixel refinement: outputs at stride-2 (2× the feature grid)
            logits = self.refine_head(fused, coarse_logits)
            refine_h, refine_w = logits.shape[-2:]
            if padding_mask is not None:
                pm = self._pool_pmask(padding_mask, refine_h, refine_w)
                logits = logits * pm.unsqueeze(1).float()
        else:
            logits = coarse_logits
            # Zero out padded positions
            if padding_mask is not None:
                pm = self._pool_pmask(padding_mask, h1, w1)
                logits = logits * pm.unsqueeze(1).float()

        return logits

    def forward(
        self,
        features: dict,                              # HieraStudentBackbone output
        padding_mask: Optional[Tensor] = None,       # (B, ph1, pw1) stride-4 pmask
        frame_indices: Optional[List[int]] = None,   # which frames to decode (None=all)
    ) -> Tensor:
        """
        Decode multi-scale Hiera features to segmentation logits.

        For T=1 (image):  returns (B, n_classes, ph1, pw1)
        For T>1 (video):  returns (B, T, n_classes, ph1, pw1)
                          or (B, len(frame_indices), n_classes, ph1, pw1)

        Parameters
        ----------
        features      : output dict from HieraStudentBackbone.forward()
        padding_mask  : (B, ph1, pw1) bool at stride-4 level
        frame_indices : optional list of frame indices to decode (saves compute
                        when only certain frames have labels)
        """
        F1 = features["F1"]   # (B, T, N1, D1)
        F2 = features["F2"]
        F3 = features["F3"]
        F4 = features["F4"]

        B, T = F1.shape[:2]

        if frame_indices is None:
            frame_indices = list(range(T))

        pmasks = features.get("pmasks") or [None, None, None, None]

        out_frames: List[Tensor] = []
        for t in frame_indices:
            logit_t = self.forward_single_frame(
                F1[:, t], F2[:, t], F3[:, t], F4[:, t],
                padding_mask=padding_mask,
                stage_masks=pmasks,
            )  # (B, n_classes, h, w)
            out_frames.append(logit_t)

        logits = torch.stack(out_frames, dim=1)   # (B, T', n_classes, h, w)

        if logits.shape[1] == 1:
            return logits.squeeze(1)              # (B, n_classes, h, w) for T=1

        return logits                             # (B, T, n_classes, h, w)

    def __repr__(self) -> str:
        flags = (
            f"adapters={self.use_adapters}, "
            f"attn_gates={self.use_attention_gates}, "
            f"aspp={self.use_aspp}, "
            f"refine_up={self.use_refine_up}"
        )
        return (
            f"UPerNetDecoder(embed_dims={self.embed_dims}, "
            f"n_classes={self.n_classes}, fpn_channels={self.fpn_channels}, "
            f"{flags})"
        )


def build_hierarchical_seg_head(
    embed_dims: List[int],
    n_classes: int = 1,
    fpn_channels: int = 256,
    use_adapters: bool = True,
    use_attention_gates: bool = True,
    use_aspp: bool = True,
    use_refine_up: bool = False,
) -> UPerNetDecoder:
    """
    Factory function for the UPerNet segmentation decoder.

    Parameters
    ----------
    embed_dims          : list of 4 ints matching the student backbone embed dims
                          e.g. [144, 288, 576, 1152] for Hiera-L
    n_classes           : output channels (1 for binary, >1 for multiclass)
    fpn_channels        : FPN internal width (default 256)
    use_adapters        : enable per-scale SegAdapters (recommended: True)
    use_attention_gates : enable FPN attention gating (recommended: True)
    use_aspp            : enable ASPP fusion (recommended: True)
    use_refine_up       : enable SubPixelRefinement head for stride-4→stride-2
                          upsampling (default False; enable for finetune tasks
                          that benefit from higher-resolution boundary precision)
    """
    return UPerNetDecoder(
        embed_dims,
        n_classes,
        fpn_channels,
        use_adapters=use_adapters,
        use_attention_gates=use_attention_gates,
        use_aspp=use_aspp,
        use_refine_up=use_refine_up,
    )
