"""Tests for clip-consistent video augmentations."""
from __future__ import annotations

import torch

from data.pipeline.transforms import (
    AugmentParams,
    VideoSSLTransformConfig,
    apply_augment,
)


def test_video_hflip_consistent_across_frames():
    cfg = VideoSSLTransformConfig()
    clip = torch.rand(4, 3, 32, 32)
    flipped = apply_augment(
        clip,
        AugmentParams(hflip=True),
        cfg,
    )
    manual = torch.stack([torch.flip(clip[t], dims=(-1,)) for t in range(4)])
    assert torch.allclose(flipped, manual)


def test_video_speckle_shared_noise_map():
    cfg = VideoSSLTransformConfig(apply_speckle=True, speckle_sigma=0.1)
    clip = torch.ones(3, 3, 16, 16)
    params = AugmentParams(apply_speckle=True, speckle_sigma=0.1)
    out = apply_augment(clip, params, cfg)
    diff0 = (out[0] - out[1]).abs().max()
    diff1 = (out[1] - out[2]).abs().max()
    assert diff0.item() == 0.0
    assert diff1.item() == 0.0
    assert not torch.allclose(out, clip)


def test_grayscale_applied():
    cfg = VideoSSLTransformConfig()
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.2, 0.2], [0.2, 0.2]], [[0.0, 1.0], [1.0, 0.0]]])
    out = apply_augment(x, AugmentParams(grayscale=True), cfg)
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[1], out[2])
