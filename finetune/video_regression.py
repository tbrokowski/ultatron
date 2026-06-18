"""
finetune/video_regression.py  ·  Shared video EF regression utilities
=====================================================================

EchoNet-Dynamic is a **video** task: EF is derived from a full cine loop spanning
the cardiac cycle.  Encoders fall into two groups:

video_native (Student, V-JEPA, Ultatron video branch)
    The backbone ingests (B, T, C, H, W) jointly and returns a temporally fused
    ``clip_cls``.  The regression head is a deep MLP on that token.

frame_based (ResNet, ViT, BioMed-CLIP, USFM, EchoCare, OpenUS, …)
    Each frame is encoded independently; per-frame CLS embeddings are stacked
    as ``frame_tokens`` (B, T, D).  A **trainable** TemporalAttentionPool inside
    the task head aggregates frames (EchoNet paper used LSTM; attention is the
    modern frozen-FM analogue), then a deep MLP predicts EF.
"""
from __future__ import annotations

from typing import Iterator, List

import torch
import torch.nn as nn
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.regression_head import VideoRegressionHead


def encode_clip_for_regression(encoder: BackboneEncoder, clips: Tensor) -> dict:
    """
    Encode a video batch for EF / measurement regression.

    Returns
    -------
    dict with:
        clip_cls      : (B, D)  fused video embedding (always present)
        frame_tokens  : (B, T, D) or None  per-frame CLS (frame-based only)
        video_native  : bool
    """
    out = encoder.encode_video(clips)
    return {
        "clip_cls":      out["clip_cls"],
        "frame_tokens":  out.get("frame_tokens"),
        "video_native":  encoder.is_video_native,
    }


def regression_head_forward(head: nn.Module, enc_out: dict) -> Tensor:
    """Dispatch head forward for video-native vs frame-based encodings."""
    if isinstance(head, VideoRegressionHead):
        if enc_out["video_native"]:
            return head(clip_cls=enc_out["clip_cls"])
        return head(frame_tokens=enc_out["frame_tokens"])
    return head(enc_out["clip_cls"])


def regression_head_params(head: nn.Module) -> List[nn.Parameter]:
    """All parameters trained for a video regression head."""
    if hasattr(head, "trainable_parameters"):
        return list(head.trainable_parameters())
    return list(head.parameters())
