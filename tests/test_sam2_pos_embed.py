"""Regression test for SAM2 variable-aspect position embedding patch."""
from __future__ import annotations

import math
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.student.hiera_backbone import (
    HieraStudentBackbone,
    _infer_sam2_unpartition_pad_hw,
    _infer_stage1_token_grid,
    _patch_sam2_multiscale_blocks,
    _patch_sam2_pos_embed,
    _sam2_get_pos_embed_fixed,
    _sam2_multiscale_block_forward_fixed,
    _make_sinusoidal_2d,
)


class _FakeSam2Backbone(nn.Module):
    def __init__(self, window: int = 8, channels: int = 16):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, channels, 14, 14))
        self.pos_embed_window = nn.Parameter(torch.randn(1, channels, window, window))


def _broken_get_pos_embed(self, hw: tuple[int, int]) -> torch.Tensor:
    h, w = hw
    window_embed = self.pos_embed_window
    pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
    pos_embed = pos_embed + window_embed.tile(
        [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
    )
    return pos_embed.permute(0, 2, 3, 1)


@pytest.mark.parametrize("h,w", [(28, 24), (30, 22), (40, 36)])
def test_sam2_pos_embed_fixed_handles_non_window_multiples(h: int, w: int) -> None:
    model = _FakeSam2Backbone()
    out = _sam2_get_pos_embed_fixed(model, (h, w))
    assert out.shape == (1, h, w, 16)


def test_sam2_pos_embed_fixed_matches_original_for_square_multiples() -> None:
    model = _FakeSam2Backbone()
    h = w = 32
    fixed = _sam2_get_pos_embed_fixed(model, (h, w))
    original = _broken_get_pos_embed(model, (h, w))
    assert torch.allclose(fixed, original, atol=1e-6)


def test_patch_replaces_method() -> None:
    model = _FakeSam2Backbone()
    _patch_sam2_pos_embed(model)
    out = model._get_pos_embed((28, 24))
    assert out.shape == (1, 28, 24, 16)


def test_infer_stage1_token_grid_native_aspect() -> None:
    ph, pw = _infer_stage1_token_grid(672, h_px=112, w_px=96)
    assert (ph, pw) == (28, 24)


def test_infer_stage1_token_grid_square() -> None:
    ph, pw = _infer_stage1_token_grid(625, h_px=100, w_px=100)
    assert (ph, pw) == (25, 25)


def test_sinusoidal_2d_matches_token_count_for_native_aspect() -> None:
    ph, pw = _infer_stage1_token_grid(672, h_px=112, w_px=96)
    enc = _make_sinusoidal_2d(ph, pw, dim=144, device=torch.device("cpu"))
    assert enc.shape == (672, 144)


def test_temporal_pos_enc_subsamples_when_t_exceeds_max_frames() -> None:
    """EchoNet loads 32 frames; student smoke config uses max_frames=16."""

    class _Stub(nn.Module):
        max_frames = 16

        def __init__(self) -> None:
            super().__init__()
            self.temporal_pos_enc = nn.Embedding(16, 144)

        _temporal_pos_enc = HieraStudentBackbone._temporal_pos_enc

    stub = _Stub()
    out = stub._temporal_pos_enc(32, torch.device("cpu"))
    assert out.shape == (1, 32, 1, 144)


def test_sam2_hiera_video_uses_single_frame_chunks() -> None:
    """32-frame video must not batch all frames through SAM2 Hiera at once."""

    class _Stub(HieraStudentBackbone):
        def __init__(self) -> None:
            self.hiera_variant = "sam2_hiera_large"

        def _hiera_forward_stages(self, x: torch.Tensor):
            self.calls.append(x.shape[0])
            n = x.shape[0] * 2  # emulate SAM2 returning 2× batch rows per input
            return tuple(torch.zeros(n, 4, 8) for _ in range(4))

    stub = _Stub()
    stub.calls = []
    x = torch.randn(32, 3, 64, 64)
    outs = stub._hiera_forward_stages_chunked(x, b=1, t=32)
    assert stub.calls == [1] * 32
    assert all(o.shape[0] == 32 for o in outs)


def test_sam2_hiera_image_keeps_full_batch() -> None:
    class _Stub:
        hiera_variant = "sam2_hiera_large"

    stub = _Stub()
    assert HieraStudentBackbone._hiera_frame_chunk_size(stub, 32, 1) == 32
    assert HieraStudentBackbone._hiera_frame_chunk_size(stub, 1, 32) == 1


def test_multiscale_block_patch_scales_pad_hw_after_q_pool() -> None:
    """Regression for ultatron_ft_busi: window_unpartition on native-aspect grids."""
    pytest.importorskip("transformers")
    from transformers.models.sam2.modeling_sam2 import window_unpartition

    # Reproduces log error: shape '[8, 15, 16, 2, 2, -1]' invalid for size 4718592
    batch_size = 8
    channels = 144
    ws_out = 2
    n_win = 8192
    windows = torch.randn(n_win, ws_out, ws_out, channels)

    fixed_pad_hw = (64, 64)
    crop_hw = (60, 64)
    out = window_unpartition(windows, ws_out, fixed_pad_hw, crop_hw)
    assert out.shape == (batch_size, crop_hw[0], crop_hw[1], channels)

    broken_pad_hw = (32, 32)
    with pytest.raises(RuntimeError, match="invalid for input"):
        window_unpartition(windows, ws_out, broken_pad_hw, (30, 32))


def test_infer_sam2_unpartition_pad_hw_from_window_count() -> None:
    pad = _infer_sam2_unpartition_pad_hw(
        n_windows=8192,
        batch_size=8,
        window_size=2,
        crop_hw=(60, 64),
        partition_pad_hw=(64, 64),
        query_stride=(2, 2),
    )
    assert pad == (64, 64)


def test_patch_multiscale_blocks_replaces_forwards() -> None:
    class _FakeBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_stride = (2, 2)
            self.window_size = 8

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states

    class _FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_FakeBlock()])

    backbone = _FakeBackbone()
    _patch_sam2_multiscale_blocks(backbone)
    assert backbone.blocks[0].forward.__func__ is _sam2_multiscale_block_forward_fixed
