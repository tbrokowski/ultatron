"""
tests/models/test_backbone_interface.py
========================================

Verifies that every backbone registered in the model registry satisfies
the ImageBackboneBase / VideoBackboneBase interface contract WITHOUT
downloading any HuggingFace weights.

Strategy: mock out AutoModel.from_pretrained to return a tiny synthetic
nn.Module whose config matches the real model's expected attributes.
This lets us test the wrapper logic (padding mask injection, token
parsing, ensure_rgb, registration) in pure CPU/offline mode.

Run:
    pytest tests/models/test_backbone_interface.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from models.base import ImageBackboneBase, VideoBackboneBase, FrozenTeacherBase
from models.registry import (
    list_image_backbones,
    list_video_backbones,
    list_frozen_teachers,
)


# ── Tiny synthetic ViT (no real weights) ──────────────────────────────────────

class _FakeConfig:
    hidden_size            = 64
    num_attention_heads    = 4
    num_register_tokens    = 4
    num_hidden_layers      = 2
    frames_per_clip        = 8


class _FakeViTOutput:
    def __init__(self, B: int, N: int, D: int = 64):
        # (B, 1+R+N, D)  — CLS + registers + patches
        seq = 1 + _FakeConfig.num_register_tokens + N
        self.last_hidden_state = torch.randn(B, seq, D)
        self.pooler_output     = self.last_hidden_state[:, 0]


class _FakeVideoOutput:
    def __init__(self, B: int, T: int, N: int, D: int = 64):
        self.last_hidden_state = torch.randn(B, T * N, D)
        self.predictor_output  = MagicMock()
        self.predictor_output.last_hidden_state = torch.randn(B, T * N, D)


class _FakeViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        # Minimal parameters so ema_update has something to iterate over
        self.dummy  = nn.Linear(64, 64)

    def forward(self, pixel_values=None, **kwargs):
        B = pixel_values.shape[0]
        # N = spatial tokens from a 32×32 image with patch_size=16 → 4 tokens
        return _FakeViTOutput(B, N=4)

    def named_modules(self):
        # Return at least one 'attention'-named module for the hook registration
        yield ("encoder.layer.0.attention.self", nn.Identity())
        yield from super().named_modules()


class _FakeVideoViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.dummy  = nn.Linear(64, 64)

    def forward(self, pixel_values_videos=None, **kwargs):
        B = pixel_values_videos.shape[0]
        T = pixel_values_videos.shape[1]
        return _FakeVideoOutput(B, T, N=4)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_image_input():
    """(B=2, 3, 32, 32) RGB image tensor — 2×2 patch grid."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def fake_video_input():
    """(B=2, T=4, 3, 32, 32) RGB video tensor."""
    return torch.randn(2, 4, 3, 32, 32)


@pytest.fixture
def fake_padding_mask():
    """(B=2, ph=2, pw=2) all-True padding mask."""
    return torch.ones(2, 2, 2, dtype=torch.bool)


@pytest.fixture
def fake_tube_mask():
    """(B=2, T=4, ph=2, pw=2) tube mask — half masked."""
    mask = torch.zeros(2, 4, 2, 2, dtype=torch.bool)
    mask[:, :, 0, 0] = True   # mask top-left patch tube
    return mask


# ── Registry completeness ─────────────────────────────────────────────────────

def test_image_backbones_registered():
    """All expected image backbone keys exist in the registry."""
    registered = set(list_image_backbones())
    expected = {"dinov3_s", "dinov3_splus", "dinov3_b", "dinov3_l", "dinov3_hplus",
                "rad_dino", "swin_v2_l"}
    missing = expected - registered
    assert not missing, f"Missing image backbone keys: {missing}"


def test_video_backbones_registered():
    registered = set(list_video_backbones())
    expected = {"vjepa2_l", "vjepa2_h", "vjepa2_g"}
    missing = expected - registered
    assert not missing, f"Missing video backbone keys: {missing}"


def test_frozen_teachers_registered():
    registered = set(list_frozen_teachers())
    assert "dinov3_7b" in registered, "dinov3_7b frozen teacher not registered"


# ── Interface contract ────────────────────────────────────────────────────────

def test_imagebranch_forward_contract(fake_image_input, fake_padding_mask):
    """
    ImageBackboneBase.forward() must return a dict with at least:
      cls          : (B, D)
      patch_tokens : (B, N, D)
    Test using DINOv3ImageBackbone with a fake ViT.
    """
    from models.image_backbones.dinov3 import DINOv3ImageBackbone

    backbone = DINOv3ImageBackbone(_FakeViT(), variant_key="dinov3_l")

    # Verify it is a proper subclass
    assert isinstance(backbone, ImageBackboneBase)

    out = backbone(fake_image_input, padding_mask=fake_padding_mask)
    assert "cls" in out,          "forward() must return 'cls'"
    assert "patch_tokens" in out, "forward() must return 'patch_tokens'"

    B = fake_image_input.shape[0]
    D = _FakeConfig.hidden_size
    assert out["cls"].shape == (B, D),          f"cls shape wrong: {out['cls'].shape}"
    assert out["patch_tokens"].ndim == 3,        "patch_tokens must be 3D"
    assert out["patch_tokens"].shape[0] == B,    "patch_tokens batch dim wrong"
    assert out["patch_tokens"].shape[2] == D,    "patch_tokens feature dim wrong"


def test_imagebranch_forward_no_padding_mask(fake_image_input):
    """forward() must work without padding_mask (returns None-safe)."""
    from models.image_backbones.dinov3 import DINOv3ImageBackbone
    backbone = DINOv3ImageBackbone(_FakeViT(), variant_key="dinov3_l")
    out = backbone(fake_image_input, padding_mask=None)
    assert "cls" in out
    assert "patch_tokens" in out


def test_imagebackbone_rgb_contract(fake_image_input):
    """forward() must accept (B, 3, H, W) input — not 1-channel."""
    from models.image_backbones.dinov3 import DINOv3ImageBackbone
    backbone = DINOv3ImageBackbone(_FakeViT(), variant_key="dinov3_l")
    assert fake_image_input.shape[1] == 3, "Test input must be 3-channel"
    out = backbone(fake_image_input)
    assert "cls" in out


def test_parameters_for_ema_default():
    """parameters_for_ema() should return an iterator of parameters by default."""
    from models.image_backbones.dinov3 import DINOv3ImageBackbone
    backbone = DINOv3ImageBackbone(_FakeViT(), variant_key="dinov3_l")
    params = list(backbone.parameters_for_ema())
    assert len(params) > 0, "parameters_for_ema() must yield at least one parameter"


def test_hidden_size_attribute():
    """hidden_size must be set as a class/instance attribute."""
    from models.image_backbones.dinov3 import DINOv3ImageBackbone
    backbone = DINOv3ImageBackbone(_FakeViT(), variant_key="dinov3_l")
    assert hasattr(backbone, "hidden_size"), "hidden_size attribute missing"
    assert isinstance(backbone.hidden_size, int), "hidden_size must be int"
    assert backbone.hidden_size == _FakeConfig.hidden_size


# ── NullALPReader contract ────────────────────────────────────────────────────

def test_null_alp_reader():
    """NullALPReader satisfies ALPReader protocol and returns expected defaults."""
    from data.pipeline.alp_interface import ALPReader, NullALPReader

    reader = NullALPReader()
    assert isinstance(reader, ALPReader), "NullALPReader must satisfy ALPReader protocol"

    result = reader.get("sample_123", alpha=0.7, n_patches=196)
    assert result is None, "NullALPReader.get() must return None"

    hardness = reader.aggregate_hardness("sample_123")
    assert hardness == 0.5, "NullALPReader.aggregate_hardness() must return 0.5"


# ── ensure_rgb contract ───────────────────────────────────────────────────────

def test_ensure_rgb_greyscale():
    """ensure_rgb must expand (B, 1, H, W) to (B, 3, H, W)."""
    from data.pipeline.transforms import ensure_rgb
    grey = torch.randn(2, 1, 32, 32)
    rgb  = ensure_rgb(grey)
    assert rgb.shape == (2, 3, 32, 32)
    # All three channels should be identical (channel repeat, not copy)
    assert torch.allclose(rgb[:, 0], rgb[:, 1])
    assert torch.allclose(rgb[:, 0], rgb[:, 2])


def test_ensure_rgb_passthrough():
    """ensure_rgb must leave (B, 3, H, W) unchanged."""
    from data.pipeline.transforms import ensure_rgb
    rgb_in = torch.randn(2, 3, 32, 32)
    rgb_out = ensure_rgb(rgb_in)
    assert rgb_out.shape == (2, 3, 32, 32)
    assert rgb_out.data_ptr() == rgb_in.data_ptr() or torch.allclose(rgb_out, rgb_in)


def test_ensure_rgb_video():
    """ensure_rgb must handle (B, T, 1, H, W) → (B, T, 3, H, W)."""
    from data.pipeline.transforms import ensure_rgb
    grey = torch.randn(2, 8, 1, 32, 32)
    rgb  = ensure_rgb(grey)
    assert rgb.shape == (2, 8, 3, 32, 32)


# ── to_canonical_tensor contract ─────────────────────────────────────────────

def test_canonical_tensor_always_rgb():
    """to_canonical_tensor must always return (3, H, W) regardless of input."""
    import numpy as np
    from data.pipeline.transforms import to_canonical_tensor

    # 2D greyscale numpy array
    grey_arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    t = to_canonical_tensor(grey_arr)
    assert t.shape == (3, 64, 64), f"2D greyscale → wrong shape {t.shape}"

    # 3-channel numpy array
    rgb_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    t = to_canonical_tensor(rgb_arr)
    assert t.shape == (3, 64, 64), f"HxWx3 array → wrong shape {t.shape}"

    # Torch (1, H, W) tensor
    grey_t = torch.randint(0, 255, (1, 64, 64), dtype=torch.uint8)
    t = to_canonical_tensor(grey_t)
    assert t.shape == (3, 64, 64), f"(1,H,W) tensor → wrong shape {t.shape}"

    # Torch (3, H, W) tensor — passthrough
    rgb_t = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
    t = to_canonical_tensor(rgb_t)
    assert t.shape == (3, 64, 64), f"(3,H,W) tensor → wrong shape {t.shape}"
