"""
models/base.py  ·  Abstract backbone interfaces
================================================

Every model that can be used as an image or video backbone must implement
one of these interfaces.  The branches (ImageBranch, VideoBranch) depend
only on these abstractions — they never import a specific backbone directly.

ImageBackboneBase
-----------------
Wraps any image encoder.  forward() must always return:
  {
    "cls"          : (B, D)    CLS or mean-pooled global token
    "patch_tokens" : (B, N, D) spatial patch tokens, N = ph*pw, D = hidden_size
    "hidden_size"  : int       (constant attribute, not per-forward)
  }

  Optional keys:
    "register_tokens" : (B, R, D) register tokens when present

  The backbone is responsible for:
    - ensure_rgb() is called at transform level; pixel_values are always (B, 3, H, W)
    - padding mask injection (however the underlying architecture does it)
    - token parsing (slicing CLS, registers, patches out of HF last_hidden_state)

  The branch (caller) is responsible for:
    - EMA updates
    - loss computation
    - gradient control (@torch.no_grad on teacher calls)

VideoBackboneBase
-----------------
Wraps any video encoder.  forward() must always return:
  {
    "clip_cls"    : (B, D)           mean-pooled temporal representation
    "tube_tokens" : (B, T*ph*pw, D)  all spatiotemporal tokens
    "hidden_size" : int
  }

  Optional keys:
    "predicted" : (B, T*ph*pw, D) predictor output (only when tube_mask given)

  The backbone handles tube_mask → context/target index conversion internally,
  so the branch can pass our (B, T, ph, pw) bool tube_mask directly.

FrozenTeacherBase
-----------------
Interface for a permanently frozen large-scale teacher used only for
knowledge distillation (e.g. DINOv3-7B).  No EMA, no gradients ever.
  {
    "cls"          : (B, D)
    "patch_tokens" : (B, N, D)
  }
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class ImageBackboneBase(nn.Module, ABC):
    """
    Abstract base for image backbone wrappers.

    Subclasses must set self.hidden_size in __init__.
    """

    hidden_size: int   # set by subclass

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,           # (B, 3, H, W) RGB [0,1]
        padding_mask: Optional[torch.Tensor] = None,  # (B, ph, pw) bool
        **kwargs,
    ) -> dict:
        """
        Returns dict with at minimum:
          cls          : (B, D)
          patch_tokens : (B, N, D)
        """
        ...

    def parameters_for_ema(self):
        """
        Yields parameters that should be EMA-updated.
        Default: all parameters.  Override if you want to exclude frozen layers.
        """
        return self.parameters()


class VideoBackboneBase(nn.Module, ABC):
    """Abstract base for video backbone wrappers."""

    hidden_size: int

    @abstractmethod
    def forward(
        self,
        pixel_values: torch.Tensor,                    # (B, T, 3, H, W) RGB [0,1]
        tube_mask: Optional[torch.Tensor] = None,      # (B, T, ph, pw) bool
        padding_mask: Optional[torch.Tensor] = None,   # (B, ph, pw) bool
        valid_frames: Optional[torch.Tensor] = None,   # (B, T) bool
        **kwargs,
    ) -> dict:
        """
        Returns dict with at minimum:
          clip_cls    : (B, D)
          tube_tokens : (B, T*ph*pw, D)
        Optional:
          predicted   : (B, T*ph*pw, D)  only when tube_mask given
        """
        ...

    def parameters_for_ema(self):
        return self.parameters()


class FrozenTeacherBase(nn.Module, ABC):
    """
    Abstract base for permanently frozen large-scale teacher models.
    No gradients, no EMA.  Only forward() is ever called.
    """

    hidden_size: int

    @abstractmethod
    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,   # (B, 3, H, W) RGB [0,1]
        **kwargs,
    ) -> dict:
        """
        Returns dict with at minimum:
          cls          : (B, D)
          patch_tokens : (B, N, D)
        """
        ...
