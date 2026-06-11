"""
finetune/backbones/base.py  ·  BackboneEncoder ABC
===================================================

Uniform interface for all backbones used in the finetune comparison:
  - Ultatron (our model, Phase 1 / 2 / 3 checkpoints)
  - Standalone V-JEPA2
  - ResNet50  (torchvision)
  - ViT-B/16  (torchvision)
  - BioMed-CLIP (open_clip)
  - USFM, EchoCare, OpenUS  (vendor stubs)

Gradient handling
-----------------
Backbone parameters are frozen (requires_grad=False) in the encoder's
__init__.  Trainable parameters (e.g. TemporalAttentionPool for image-only
models) are returned by trainable_parameters() and added to the task
optimiser alongside the task head.

For image-only encoders, encode_video() uses torch.no_grad() for the
frozen backbone sub-call while the TemporalAttentionPool runs outside
that context so pool gradients flow correctly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor


class BackboneEncoder(ABC):
    """Abstract base for all backbone encoders in the comparison suite."""

    # ── Required ───────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier, e.g. 'ours_phase3', 'resnet50'."""
        ...

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimension of the image-branch CLS embedding."""
        ...

    @abstractmethod
    def encode_image(self, images: Tensor) -> dict:
        """
        Encode a batch of images.

        Parameters
        ----------
        images : (B, C, H, W)

        Returns
        -------
        dict with keys:
            cls          : (B, D)      global image embedding
            patch_tokens : (B, N, D)   spatial patch tokens (None if unavailable)
        """
        ...

    # ── Optional overrides ────────────────────────────────────────────────────

    @property
    def video_embed_dim(self) -> int:
        """Video-branch embedding dimension. Defaults to embed_dim."""
        return self.embed_dim

    @property
    def is_video_native(self) -> bool:
        """True if the encoder has a genuine temporal video encoder."""
        return False

    def encode_video(self, clips: Tensor) -> dict:
        """
        Encode a batch of video clips.

        Parameters
        ----------
        clips : (B, T, C, H, W)

        Returns
        -------
        dict with keys:
            clip_cls    : (B, D)         global video embedding
            tube_tokens : (B, T*N, D)    spatiotemporal tokens (None if unavailable)
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement encode_video(). "
            "Image-only subclasses with TemporalAttentionPool override this."
        )

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """
        Parameters trained alongside the task head (not the frozen backbone).

        Image-only models: TemporalAttentionPool parameters.
        Video-native models: empty iterator.
        """
        return iter([])

    # ── nn.Module forwarding ───────────────────────────────────────────────────

    def to(self, *args, **kwargs) -> "BackboneEncoder":
        for attr in self._nn_modules():
            attr.to(*args, **kwargs)
        return self

    def eval(self) -> "BackboneEncoder":
        for attr in self._nn_modules():
            attr.eval()
        return self

    def train(self, mode: bool = True) -> "BackboneEncoder":
        for attr in self._nn_modules():
            attr.train(mode)
        return self

    def _nn_modules(self):
        for v in self.__dict__.values():
            if isinstance(v, nn.Module):
                yield v

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"embed_dim={self.embed_dim}, "
            f"video_embed_dim={self.video_embed_dim}, "
            f"is_video_native={self.is_video_native})"
        )
