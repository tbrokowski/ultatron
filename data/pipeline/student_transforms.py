"""
data/pipeline/student_transforms.py  ·  Student-specific augmentations
=======================================================================

Two new augmentations for the single-student training pipeline:

TemporalDropout
---------------
For video batches, randomly collapses T>1 to T=1 by keeping one frame.
Prevents the encoder from depending on temporal context in all paths
and bridges the image/video representation gap.
Applied with low probability (default 0.10).

ImageToPseudoClip
-----------------
Turns a still image (T=1) into a T-frame pseudo-clip by repeating the
frame with optional per-frame light augmentation.  Used sparingly
(default 0.05) so the model learns that not all videos have motion,
without over-learning static clips.

StudentVideoSSLTransform
------------------------
Thin wrapper around the existing VideoSSLTransform that applies
TemporalDropout and (optionally) ImageToPseudoClip before returning
the standard batch dict.

All augmentations are no-ops when not triggered, so the pipeline
output shape contract is identical to the existing transforms.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# TemporalDropout
# ---------------------------------------------------------------------------

class TemporalDropout:
    """
    Randomly collapse a T>1 clip to T=1 by keeping one random frame.

    Applied to (T, C, H, W) tensors or (B, T, C, H, W) batches.

    Parameters
    ----------
    p : probability of collapsing the temporal dimension
    """

    def __init__(self, p: float = 0.10):
        assert 0.0 <= p <= 1.0
        self.p = p

    def __call__(self, clip: Tensor) -> Tensor:
        """
        Parameters
        ----------
        clip : (T, C, H, W) or (B, T, C, H, W)

        Returns
        -------
        (1, C, H, W) or (B, 1, C, H, W) if triggered, else unchanged clip
        """
        if self.p <= 0.0 or random.random() >= self.p:
            return clip

        if clip.dim() == 4:         # (T, C, H, W)
            T = clip.shape[0]
            if T <= 1:
                return clip
            t = random.randint(0, T - 1)
            return clip[t:t+1]      # keep (1, C, H, W)

        elif clip.dim() == 5:       # (B, T, C, H, W)
            T = clip.shape[1]
            if T <= 1:
                return clip
            t = random.randint(0, T - 1)
            return clip[:, t:t+1]   # keep (B, 1, C, H, W)

        return clip

    def __repr__(self) -> str:
        return f"TemporalDropout(p={self.p})"


# ---------------------------------------------------------------------------
# ImageToPseudoClip
# ---------------------------------------------------------------------------

class ImageToPseudoClip:
    """
    Turn a still image (T=1) into a T-frame pseudo-clip by repeating
    with optional per-frame light augmentation.

    This teaches the video encoder that not all clips have meaningful
    temporal dynamics.  Use sparingly (low p).

    Parameters
    ----------
    target_T  : number of frames in the pseudo-clip (e.g. 4 or 8)
    p         : probability of expansion
    aug_fn    : optional callable (Tensor → Tensor) applied independently
                to each repeated frame for mild diversity
                (e.g. slight colour jitter, tiny crop jitter)
    """

    def __init__(
        self,
        target_T: int = 4,
        p: float = 0.05,
        aug_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        assert 0.0 <= p <= 1.0
        self.target_T = target_T
        self.p        = p
        self.aug_fn   = aug_fn

    def __call__(self, image: Tensor) -> Tensor:
        """
        Parameters
        ----------
        image : (1, C, H, W)  — single-frame image

        Returns
        -------
        (target_T, C, H, W) pseudo-clip if triggered, else unchanged input
        """
        if self.p <= 0.0 or random.random() >= self.p:
            return image

        if image.dim() != 4 or image.shape[0] != 1:
            return image

        frames: List[Tensor] = []
        for _ in range(self.target_T):
            frame = image[0]          # (C, H, W)
            if self.aug_fn is not None:
                frame = self.aug_fn(frame)
            frames.append(frame)

        return torch.stack(frames, dim=0)  # (target_T, C, H, W)

    def __repr__(self) -> str:
        return f"ImageToPseudoClip(T={self.target_T}, p={self.p})"


# ---------------------------------------------------------------------------
# Config dataclass for student-specific transform settings
# ---------------------------------------------------------------------------

@dataclass
class StudentTransformConfig:
    """
    Configuration for student-specific augmentations.

    Fields
    ------
    temporal_dropout_p   : probability of TemporalDropout for video clips
    pseudo_clip_p        : probability of ImageToPseudoClip for still images
    pseudo_clip_T        : number of frames in pseudo-clips
    patch_size           : patch stride for padding-mask computation (4 for Hiera)
    """
    temporal_dropout_p: float = 0.10
    pseudo_clip_p:      float = 0.05
    pseudo_clip_T:      int   = 4
    patch_size:         int   = 4   # Hiera effective stride


# ---------------------------------------------------------------------------
# StudentVideoSSLTransform wrapper
# ---------------------------------------------------------------------------

class StudentVideoSSLTransform:
    """
    Wraps the existing VideoSSLTransform to add TemporalDropout.

    For the student pipeline, call this instead of VideoSSLTransform.
    When temporal_dropout triggers, the returned batch has T=1 and
    sample_type is updated to "image" by the collator downstream.

    Parameters
    ----------
    base_transform  : existing VideoSSLTransform instance
    student_cfg     : StudentTransformConfig
    """

    def __init__(
        self,
        base_transform,
        student_cfg: StudentTransformConfig,
    ):
        self.base        = base_transform
        self.t_dropout   = TemporalDropout(p=student_cfg.temporal_dropout_p)

    def __call__(self, sample: dict) -> dict:
        out = self.base(sample)

        # Apply temporal dropout to clip tensors if present
        for key in ("full_clips", "visible_clips"):
            if key in out and out[key] is not None:
                out[key] = self.t_dropout(out[key])

        return out

    def __repr__(self) -> str:
        return f"StudentVideoSSLTransform(base={self.base}, dropout={self.t_dropout})"


def build_student_transforms(
    base_image_transform,
    base_video_transform,
    student_cfg: Optional[StudentTransformConfig] = None,
):
    """
    Factory that wraps existing transforms with student-specific augmentations.

    Returns (image_transform, video_transform) pair where video_transform
    has TemporalDropout applied.

    Parameters
    ----------
    base_image_transform : existing ImageSSLTransform instance
    base_video_transform : existing VideoSSLTransform instance
    student_cfg          : StudentTransformConfig (uses defaults if None)
    """
    if student_cfg is None:
        student_cfg = StudentTransformConfig()

    video_transform = StudentVideoSSLTransform(base_video_transform, student_cfg)
    return base_image_transform, video_transform
