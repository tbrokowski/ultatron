"""
models/student/cross_attention_fusion.py  ·  Local DINO-to-V-JEPA fusion
=========================================================================

Training-time only.  Never part of the deployed student.

The mechanism: frozen DINOv3 dense frame tokens are locally injected into
frozen V-JEPA2 spatiotemporal tube tokens to create fused semantic-temporal
teacher targets.  The student is trained to match these fused targets.

Key design choices
------------------
Local, not global: each V-JEPA tube token queries only the DINO patches
within its spatial neighbourhood (radius r).  This prevents spurious
correspondences from ultrasound fan borders, text overlays, and probe
shadows.

Small initial gate: σ(-3.0) ≈ 0.05, so fusion starts nearly zero and
gradually opens as training stabilises.

One-directional: V-JEPA tubes query DINO patches; no reverse direction.
A single image cannot truly contain temporal motion.

Components
----------
build_tube_to_patch_map  : precompute DINO patch neighbourhood indices
                           for each tube grid position
gather_local_patches     : index-select DINO patches for each tube
LocalGatedCrossAttentionFusion : gated cross-attention module
FusionTargetBuilder      : end-to-end wrapper; detach_output controls target detachment
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spatial neighbourhood mapping
# ---------------------------------------------------------------------------

def build_tube_to_patch_map(
    ph_dino: int,
    pw_dino: int,
    ph_vid: int,
    pw_vid: int,
    radius: int = 2,
) -> Tensor:
    """
    Precompute, for each video-grid position (h_v, w_v), the flat DINO
    patch indices within the spatial neighbourhood h_d ± radius, w_d ± radius.

    The DINO grid and video grid may differ in resolution.  Each video position
    is mapped to a DINO position by bilinear scaling.

    Parameters
    ----------
    ph_dino, pw_dino : DINO patch grid dimensions  (stride-16 of input)
    ph_vid,  pw_vid  : video tube grid dimensions  (stride-16 of V-JEPA)
    radius           : neighbourhood half-size in DINO-patch units

    Returns
    -------
    LongTensor (ph_vid, pw_vid, K) where K = (2*radius+1)^2.
    Out-of-bounds positions are clamped to the nearest valid index (0..N_img-1).
    A companion bool mask (ph_vid, pw_vid, K) indicates valid vs clamped
    entries; returned as a second element.
    """
    K = (2 * radius + 1) ** 2
    idx_map  = torch.zeros(ph_vid, pw_vid, K, dtype=torch.long)
    valid_map = torch.zeros(ph_vid, pw_vid, K, dtype=torch.bool)

    # Scale factors from video grid to DINO grid
    scale_h = ph_dino / ph_vid
    scale_w = pw_dino / pw_vid

    for hv in range(ph_vid):
        for wv in range(pw_vid):
            # Corresponding DINO grid centre (float)
            hd_centre = (hv + 0.5) * scale_h - 0.5
            wd_centre = (wv + 0.5) * scale_w - 0.5

            k = 0
            for dh in range(-radius, radius + 1):
                for dw in range(-radius, radius + 1):
                    hd = int(round(hd_centre)) + dh
                    wd = int(round(wd_centre)) + dw
                    in_bounds = (0 <= hd < ph_dino) and (0 <= wd < pw_dino)
                    hd_clamp = max(0, min(ph_dino - 1, hd))
                    wd_clamp = max(0, min(pw_dino - 1, wd))
                    idx_map[hv, wv, k]   = hd_clamp * pw_dino + wd_clamp
                    valid_map[hv, wv, k] = in_bounds
                    k += 1

    return idx_map, valid_map


def gather_local_patches(
    z_img: Tensor,              # (B, N_img, D)
    tube_to_patch_map: Tensor,  # (ph_vid, pw_vid, K)  LongTensor
) -> Tensor:
    """
    For each video tube grid position, gather the K local DINO patch tokens.

    Parameters
    ----------
    z_img             : (B, N_img, D)
    tube_to_patch_map : (ph_vid, pw_vid, K) flat DINO patch indices

    Returns
    -------
    (B, N_vid, K, D)  where N_vid = ph_vid * pw_vid
    """
    B, N_img, D = z_img.shape
    ph_vid, pw_vid, K = tube_to_patch_map.shape
    N_vid = ph_vid * pw_vid

    # Flatten map to (N_vid, K) and expand to (B, N_vid, K)
    flat_map = tube_to_patch_map.reshape(N_vid, K)          # (N_vid, K)
    flat_map = flat_map.unsqueeze(0).expand(B, -1, -1)      # (B, N_vid, K)

    # Expand z_img for gather: (B, N_img, D) → (B, N_vid, N_img, D)
    # Then gather K entries per tube position
    idx = flat_map.unsqueeze(-1).expand(-1, -1, -1, D)      # (B, N_vid, K, D)
    z_expanded = z_img.unsqueeze(1).expand(-1, N_vid, -1, -1)  # (B, N_vid, N_img, D)
    z_local = torch.gather(z_expanded, dim=2, index=idx)    # (B, N_vid, K, D)
    return z_local


# ---------------------------------------------------------------------------
# Gated local cross-attention
# ---------------------------------------------------------------------------

class LocalGatedCrossAttentionFusion(nn.Module):
    """
    V-JEPA tube tokens query their local DINO frame patches via cross-attention
    with a learned per-dimension gate.

    Gate initialised to -3.0 → σ(-3.0) ≈ 0.05, so fusion starts nearly zero
    and opens gradually during Stage 3 training.

    Parameters
    ----------
    dim        : token dimension (projected, shared for both modalities)
    num_heads  : attention heads
    init_gate  : initial gate logit value (default -3.0)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        init_gate: float = -3.0,
    ):
        super().__init__()
        self.dim = dim

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Per-dimension learned gate; starts almost closed
        self.gate = nn.Parameter(torch.full((dim,), init_gate))

        log.info(
            f"LocalGatedCrossAttentionFusion(dim={dim}, heads={num_heads}, "
            f"init_gate={init_gate:.1f} → σ={torch.sigmoid(torch.tensor(init_gate)):.3f})"
        )

    def forward(
        self,
        z_vid: Tensor,        # (B, N_vid, D)
        z_img_local: Tensor,  # (B, N_vid, K, D)  local DINO patches per tube
        valid_map: Optional[Tensor] = None,  # (N_vid, K) bool — in-bounds mask
    ) -> Tensor:
        """
        Returns z_fused = z_vid + σ(gate) · cross_attn_output
        """
        B, N_vid, D = z_vid.shape
        K = z_img_local.shape[2]
        if z_img_local.shape[1] != N_vid:
            N = min(N_vid, z_img_local.shape[1])
            z_vid = z_vid[:, :N]
            z_img_local = z_img_local[:, :N]
            N_vid = N

        # Normalise inputs
        q = self.norm_q(z_vid.float())            # (B, N_vid, D)
        kv = self.norm_kv(z_img_local.float())    # (B, N_vid, K, D)

        # Reshape for MHA: (B*N_vid, 1, D) query, (B*N_vid, K, D) key/value
        q_flat  = q.reshape(B * N_vid, 1, D)
        kv_flat = kv.reshape(B * N_vid, K, D)

        # key_padding_mask: True = IGNORE in MHA
        key_pad = None
        if valid_map is not None:
            # valid_map: (N_vid, K) bool True=in-bounds
            # MHA key_padding_mask: True=IGNORE, so invert and expand to B
            key_pad = (~valid_map).unsqueeze(0).expand(B, -1, -1)  # (B, N_vid, K)
            key_pad = key_pad.reshape(B * N_vid, K)                 # (B*N_vid, K)

        attn_out, _ = self.cross_attn(
            query=q_flat,
            key=kv_flat,
            value=kv_flat,
            key_padding_mask=key_pad,
        )  # (B*N_vid, 1, D)

        semantic_update = attn_out.reshape(B, N_vid, D)

        # Gated residual
        gate_weight = torch.sigmoid(self.gate).to(z_vid.dtype)
        z_fused = z_vid + gate_weight * semantic_update.to(z_vid.dtype)

        return z_fused


# ---------------------------------------------------------------------------
# End-to-end builder convenience wrapper
# ---------------------------------------------------------------------------

class FusionTargetBuilder(nn.Module):
    """
    Convenience wrapper used in student_stage3_step for paired batches.

    Given projected DINO frame tokens and projected V-JEPA tube tokens,
    builds the fused teacher target.  By default returns z_fused.detach()
    for use as a student distillation target; set detach_output=False and
    return_vid_baseline=True when training the fusion module (preservation loss).

    Also provides img_proj / vid_proj linear projections to map the raw
    teacher outputs into the shared align_dim space before fusion.

    Parameters
    ----------
    d_img      : DINOv3 hidden dim (e.g. 1024 for dinov3_l)
    d_vid      : V-JEPA2 hidden dim (e.g. 1024 for vjepa2_l)
    align_dim  : shared projection dim
    num_heads  : cross-attention heads
    fusion_radius : spatial neighbourhood radius in DINO-patch units
    init_gate  : initial gate logit
    """

    def __init__(
        self,
        d_img: int = 1024,
        d_vid: int = 1024,
        align_dim: int = 512,
        num_heads: int = 8,
        fusion_radius: int = 2,
        init_gate: float = -3.0,
    ):
        super().__init__()
        self.align_dim     = align_dim
        self.fusion_radius = fusion_radius

        # Projection layers (small MLP: LayerNorm + Linear)
        self.img_proj = nn.Sequential(
            nn.LayerNorm(d_img),
            nn.Linear(d_img, align_dim, bias=False),
        )
        self.vid_proj = nn.Sequential(
            nn.LayerNorm(d_vid),
            nn.Linear(d_vid, align_dim, bias=False),
        )

        self.fusion = LocalGatedCrossAttentionFusion(
            dim=align_dim,
            num_heads=num_heads,
            init_gate=init_gate,
        )

    def forward(
        self,
        z_img_raw: Tensor,          # (B, N_img, D_img) — raw DINO patch tokens
        z_vid_raw: Tensor,          # (B, N_vid, D_vid) — raw V-JEPA tube tokens
        ph_dino: int,
        pw_dino: int,
        ph_vid: int,
        pw_vid: int,
        device: Optional[torch.device] = None,
        detach_output: bool = True,
        return_vid_baseline: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Build fused teacher target in align_dim.

        Parameters
        ----------
        detach_output : when True (default), returned z_fused is detached
                        (student distillation target).  Set False for
                        preservation loss so gradients reach fusion weights.
        return_vid_baseline : when True, also return pre-fusion projected
                        V-JEPA tokens (z_vid.detach()) for preservation_loss.
        """
        device = device or z_img_raw.device

        # Project both teachers to shared space
        z_img = self.img_proj(z_img_raw.float())   # (B, N_img, align_dim)
        z_vid = self.vid_proj(z_vid_raw.float())   # (B, N_vid, align_dim)

        # Build tube-to-patch map (computed fresh each call for native-res support)
        tube_map, valid_map = build_tube_to_patch_map(
            ph_dino, pw_dino, ph_vid, pw_vid, radius=self.fusion_radius
        )
        tube_map  = tube_map.to(device)
        valid_map = valid_map.to(device).reshape(ph_vid * pw_vid, -1)  # (N_vid, K)

        # Gather local DINO patches for each tube position
        z_img_local = gather_local_patches(z_img, tube_map)  # (B, N_vid, K, align_dim)

        # Fuse: V-JEPA tubes query local DINO patches
        z_fused = self.fusion(z_vid, z_img_local, valid_map=valid_map)
        z_out = z_fused.detach() if detach_output else z_fused

        if return_vid_baseline:
            return z_out, z_vid.detach()
        return z_out
