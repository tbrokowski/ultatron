"""Tests for VideoRegressionHead and video EF encoding dispatch."""
from __future__ import annotations

import torch

from finetune.video_regression import regression_head_forward
from models.heads.regression_head import VideoRegressionHead, build_video_regression_head


class _FakeNativeEncoder:
    is_video_native = True
    video_embed_dim = 64

    def encode_video(self, clips):
        b = clips.shape[0]
        return {"clip_cls": torch.randn(b, 64), "frame_tokens": torch.randn(b, 4, 64)}


class _FakeFrameEncoder:
    is_video_native = False
    video_embed_dim = 64

    def encode_video(self, clips):
        b, t = clips.shape[0], clips.shape[1]
        return {"clip_cls": torch.randn(b, 64), "frame_tokens": torch.randn(b, t, 64)}


def test_video_regression_head_native_forward():
    head = VideoRegressionHead(embed_dim=64, hidden_dim=32, video_native=True)
    out = head(clip_cls=torch.randn(3, 64))
    assert out.shape == (3,)


def test_video_regression_head_frame_forward():
    head = VideoRegressionHead(embed_dim=64, hidden_dim=32, video_native=False)
    out = head(frame_tokens=torch.randn(3, 8, 64))
    assert out.shape == (3,)


def test_video_regression_head_multi_output():
    head = VideoRegressionHead(embed_dim=64, n_outputs=3, video_native=True)
    out = head(clip_cls=torch.randn(2, 64))
    assert out.shape == (2, 3)


def test_build_mlp_vs_linear():
    mlp = build_video_regression_head("mlp", 64, video_native=False)
    lin = build_video_regression_head("linear", 64, video_native=True)
    assert isinstance(mlp, VideoRegressionHead)
    assert not isinstance(lin, VideoRegressionHead)


def test_regression_head_forward_dispatch():
    clips = torch.randn(2, 4, 3, 16, 16)
    mlp_head = build_video_regression_head("mlp", 64, video_native=False)

    native_enc = _FakeNativeEncoder()
    frame_enc = _FakeFrameEncoder()

    from finetune.video_regression import encode_clip_for_regression

    native_out = encode_clip_for_regression(native_enc, clips)
    frame_out = encode_clip_for_regression(frame_enc, clips)
    assert native_out["video_native"] is True
    assert frame_out["video_native"] is False
    assert native_out["frame_tokens"].shape == (2, 4, 64)

    mlp_native = build_video_regression_head("mlp", 64, video_native=True)
    p1 = regression_head_forward(mlp_native, native_out)
    p2 = regression_head_forward(mlp_head, frame_out)
    assert p1.shape == (2,)
    assert p2.shape == (2,)


def test_frame_head_trainable_pool_params():
    head = VideoRegressionHead(embed_dim=64, video_native=False)
    param_names = {n for n, _ in head.named_parameters()}
    assert any("frame_pool" in n for n in param_names)
