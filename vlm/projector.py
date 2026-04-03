"""
vlm/projector.py  ·  UltatronProjector
========================================

Maps frozen Ultatron backbone tokens (1024-dim DINOv3 patch tokens and
V-JEPA2 tube tokens) into Qwen2.5-VL 7B's LLM hidden dimension (3584-dim)
so they can be prepended to Qwen's own visual token sequence.

Architecture
------------
  img branch : LayerNorm(1024) → Linear(1024, 2048) → GELU → Linear(2048, 3584)
  vid branch : LayerNorm(1024) → Linear(1024, 2048) → GELU → Linear(2048, 3584)

The two heads are kept separate because image patch tokens and video tube tokens
carry different spatial/temporal structure.

Usage
-----
  proj = UltatronProjector.from_checkpoint(
      "/capstor/.../run1/phase3_end.pt",
      img_branch_key="img_branch",
      vid_branch_key="vid_branch",
      device="cuda",
  )
  # At inference / rollout time:
  img_tokens = proj.project_image(patch_tokens)   # (B, N, 3584)
  vid_tokens = proj.project_video(tube_tokens)    # (B, T, 3584)
  ultatron_tokens = proj.project(patch_tokens, tube_tokens)  # (B, N+T, 3584)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# Dimensions
ULTATRON_DIM  = 1024   # DINOv3-L / V-JEPA2-L hidden size
QWEN_HIDDEN   = 3584   # Qwen2.5-VL 7B LLM hidden dim
_MID_DIM      = 2048   # intermediate projection width


class _BranchProjector(nn.Module):
    """Two-layer MLP projection: D_in → mid → D_out."""

    def __init__(self, d_in: int = ULTATRON_DIM, d_mid: int = _MID_DIM, d_out: int = QWEN_HIDDEN):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.fc1  = nn.Linear(d_in, d_mid, bias=True)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(d_mid, d_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D_in)  →  (B, N, D_out)"""
        return self.fc2(self.act(self.fc1(self.norm(x))))


class UltatronProjector(nn.Module):
    """
    Projects Ultatron image-patch and video-tube tokens into Qwen2.5-VL hidden space.

    Parameters
    ----------
    img_dim : input dim for image patch tokens (default 1024 for DINOv3-L)
    vid_dim : input dim for video tube tokens  (default 1024 for V-JEPA2-L)
    qwen_dim : Qwen LLM hidden size             (default 3584 for 7B)
    mid_dim  : intermediate MLP width           (default 2048)
    """

    def __init__(
        self,
        img_dim:  int = ULTATRON_DIM,
        vid_dim:  int = ULTATRON_DIM,
        qwen_dim: int = QWEN_HIDDEN,
        mid_dim:  int = _MID_DIM,
    ):
        super().__init__()
        self.img_proj = _BranchProjector(img_dim, mid_dim, qwen_dim)
        self.vid_proj = _BranchProjector(vid_dim, mid_dim, qwen_dim)

    # ── Forward helpers ─────────────────────────────────────────────────────

    def project_image(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        patch_tokens : (B, N, img_dim)
        returns      : (B, N, qwen_dim)
        """
        return self.img_proj(patch_tokens)

    def project_video(self, tube_tokens: torch.Tensor) -> torch.Tensor:
        """
        tube_tokens : (B, T, vid_dim)  — spatiotemporal tube tokens
        returns     : (B, T, qwen_dim)
        """
        return self.vid_proj(tube_tokens)

    def project(
        self,
        patch_tokens: Optional[torch.Tensor],
        tube_tokens:  Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Concatenate projected image and video tokens along the sequence axis.

        At least one of patch_tokens / tube_tokens must be provided.

        returns : (B, N+T, qwen_dim)
        """
        parts = []
        if patch_tokens is not None:
            parts.append(self.project_image(patch_tokens))
        if tube_tokens is not None:
            parts.append(self.project_video(tube_tokens))
        if not parts:
            raise ValueError("At least one of patch_tokens or tube_tokens must be provided.")
        return torch.cat(parts, dim=1)

    # ── Factory ─────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path:       str,
        img_branch_key:  str = "img_branch",
        vid_branch_key:  str = "vid_branch",
        img_dim:         int = ULTATRON_DIM,
        vid_dim:         int = ULTATRON_DIM,
        qwen_dim:        int = QWEN_HIDDEN,
        mid_dim:         int = _MID_DIM,
        device:          str = "cuda",
        dtype:           torch.dtype = torch.bfloat16,
        strict_backbone: bool = False,
    ) -> "UltatronProjector":
        """
        Build an UltatronProjector and optionally validate that the expected
        Ultatron checkpoint keys are present (does NOT load backbone weights —
        the backbone is loaded separately and kept frozen).

        Parameters
        ----------
        ckpt_path       : path to phase3_end.pt (or latest.pt) from run1
        img_branch_key  : top-level key for the image branch in the checkpoint
        vid_branch_key  : top-level key for the video branch in the checkpoint
        strict_backbone : if True, raise if backbone keys are missing from ckpt
        """
        proj = cls(img_dim=img_dim, vid_dim=vid_dim, qwen_dim=qwen_dim, mid_dim=mid_dim)
        proj = proj.to(device=device, dtype=dtype)

        path = Path(ckpt_path)
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            keys = set(ckpt.get("model", ckpt).keys())
            img_ok = any(k.startswith(img_branch_key) for k in keys)
            vid_ok = any(k.startswith(vid_branch_key) for k in keys)
            if not img_ok and strict_backbone:
                raise KeyError(f"'{img_branch_key}' not found in checkpoint {path}")
            if not vid_ok and strict_backbone:
                raise KeyError(f"'{vid_branch_key}' not found in checkpoint {path}")
            log.info(
                f"UltatronProjector: checkpoint verified ({path.name}), "
                f"img_ok={img_ok}, vid_ok={vid_ok}.  Projector weights freshly initialised."
            )
        else:
            log.warning(f"UltatronProjector: checkpoint not found at {path}. "
                        f"Projector initialised from scratch.")

        return proj

    @classmethod
    def load_projector_weights(
        cls,
        proj_ckpt_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "UltatronProjector":
        """
        Resume from a previously saved projector-only checkpoint (saved during
        VLM GRPO training).
        """
        proj = cls(**kwargs).to(device=device, dtype=dtype)
        state = torch.load(proj_ckpt_path, map_location="cpu")
        proj.load_state_dict(state)
        log.info(f"UltatronProjector weights loaded from {proj_ckpt_path}")
        return proj
