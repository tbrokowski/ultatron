"""
finetune/backbones/usfm_encoder.py  ·  USFM ultrasound foundation model encoder
=================================================================================

USFM (Ultrasound Foundation Model) is a ViT-based encoder pre-trained with
frequency-domain masked image modelling on diverse ultrasound data.

Paper / repo: https://github.com/openmedlab/USFM

Architecture:
    ViT backbone (from the usdsgen package's modelling infrastructure).
    The USFM_latest.pth checkpoint contains the full backbone state_dict.

Setup:
    1. Clone vendor code and install usdsgen:
           git clone https://github.com/openmedlab/USFM \\
               finetune/backbones/vendor/usfm
           pip install -e finetune/backbones/vendor/usfm

    2. Set checkpoint path (or use env override):
           US_USFM_CHECKPOINT = /capstor/store/.../checkpoints/Ablations/USFM_latest.pth
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)

_VENDOR_PATH = Path(__file__).parent / "vendor" / "usfm"
_USFM_REPO   = "https://github.com/openmedlab/USFM"

# USFM uses a ViT-based architecture; exact embedding dim depends on
# the specific config used for pre-training USFM_latest.pth.
# Based on the USFM paper, the default backbone is ViT-B/16 → embed_dim=768.
_DEFAULT_EMBED_DIM = 768
_USFM_IMG_SIZE     = 224

# Architecture matches configs/model/Cls/vit.yaml in the USFM repo.
_USFM_VIT_CFG = dict(
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=0,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    init_values=0.1,
    use_abs_pos_emb=False,
    use_rel_pos_bias=True,
    use_shared_rel_pos_bias=False,
    use_mean_pooling=False,
    pretrained=False,
)


def _ensure_vendor_clone() -> bool:
    """Clone USFM repo and install usdsgen if absent. Returns True on success."""
    if (_VENDOR_PATH / "usdsgen").exists():
        return True

    log.info("[usfm] Cloning %s → %s", _USFM_REPO, _VENDOR_PATH)
    try:
        _VENDOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth=1", _USFM_REPO, str(_VENDOR_PATH)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            log.error("[usfm] git clone failed:\n%s", result.stderr)
            return False
        log.info("[usfm] Clone succeeded.")
    except Exception as exc:
        log.error("[usfm] git clone error: %s", exc)
        return False

    log.info("[usfm] Installing usdsgen package …")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(_VENDOR_PATH)],
            capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            log.error("[usfm] pip install failed:\n%s", result.stderr)
            return False
        log.info("[usfm] usdsgen installed.")
        return True
    except Exception as exc:
        log.error("[usfm] pip install error: %s", exc)
        return False


class USFMEncoder(BackboneEncoder):
    """
    USFM ultrasound foundation model encoder.

    Loads the USFM backbone from USFM_latest.pth and wraps it in the
    BackboneEncoder interface.  If usdsgen is not installed, setup will
    clone the repo and install it automatically.

    Parameters
    ----------
    checkpoint : str
        Path to USFM pre-trained weights (USFM_latest.pth).
    embed_dim : int
        Feature dimension of the backbone CLS token (default 768 for ViT-B).
    temporal_dropout : float
        Dropout for TemporalAttentionPool used in encode_video().
    """

    def __init__(
        self,
        checkpoint:      str,
        embed_dim:       int   = _DEFAULT_EMBED_DIM,
        temporal_dropout: float = 0.0,
    ):
        self._d    = embed_dim
        self._ckpt = checkpoint

        self.backbone      = self._load_model(checkpoint)
        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        log.info(f"[usfm] Ready (embed_dim={self._d})")

    def _load_model(self, checkpoint: str) -> nn.Module:
        if not _ensure_vendor_clone():
            raise ImportError(
                f"USFM vendor code not found at {_VENDOR_PATH}.\n"
                f"Please clone {_USFM_REPO} into {_VENDOR_PATH} and "
                f"run `pip install -e {_VENDOR_PATH}`."
            )

        if str(_VENDOR_PATH) not in sys.path:
            sys.path.insert(0, str(_VENDOR_PATH))

        model = self._build_backbone()
        self._load_weights(model, checkpoint)
        return model

    def _build_backbone(self) -> nn.Module:
        """Build the USFM ViT backbone (usdsgen VisionTransformer, weights loaded separately)."""
        from functools import partial

        errors = []

        try:
            from usdsgen.modules.backbone.vision_transformer import VisionTransformer  # type: ignore[import]
            vit_kwargs = {k: v for k, v in _USFM_VIT_CFG.items() if k != "pretrained"}
            model = VisionTransformer(
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                **vit_kwargs,
            )
            log.info("[usfm] Backbone built via usdsgen.VisionTransformer")
            return model
        except Exception as e:
            errors.append(f"usdsgen.VisionTransformer: {e}")

        raise ImportError(
            "Could not build USFM backbone. Tried:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\nEnsure usdsgen is installed: pip install -e finetune/backbones/vendor/usfm"
        )

    @staticmethod
    def _resize_images(images: Tensor, size: int = _USFM_IMG_SIZE) -> Tensor:
        if images.shape[-1] == size and images.shape[-2] == size:
            return images
        return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)

    @staticmethod
    def _forward_tokens(model: nn.Module, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Run USFM ViT blocks and return (cls, patch_tokens)."""
        x = model.patch_embed(images)
        batch_size = x.size(0)
        cls_tokens = model.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if model.pos_embed is not None:
            x = x + model.pos_embed
        x = model.pos_drop(x)

        rel_pos_bias = model.rel_pos_bias() if model.rel_pos_bias is not None else None
        for blk in model.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = model.norm(x)
        return x[:, 0], x[:, 1:]

    def _load_weights(self, model: nn.Module, checkpoint: str) -> None:
        """Load USFM checkpoint weights into model (strict=False)."""
        log.info("[usfm] Loading checkpoint: %s", checkpoint)
        ckpt = torch.load(checkpoint, map_location="cpu")

        # USFM may store the state_dict in various keys
        if isinstance(ckpt, dict):
            state = (
                ckpt.get("state_dict")
                or ckpt.get("model")
                or ckpt.get("backbone")
                or ckpt
            )
        else:
            state = ckpt

        # Strip common prefixes
        state = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state.items()
        }

        # Official USFM loader expands top-level rel_pos_bias into every block.
        # Without this remap, 24 block-level RPB tables stay random → garbage features.
        try:
            from usdsgen.utils.modelutils import remap_pretrained_keys_vit  # type: ignore[import]
            state = remap_pretrained_keys_vit(model, state, log)
        except Exception as exc:
            log.warning("[usfm] remap_pretrained_keys_vit failed (%s) — loading raw keys", exc)

        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(
            "[usfm] Weights loaded  missing=%d  unexpected=%d",
            len(missing), len(unexpected),
        )
        if missing:
            log.warning("[usfm] Missing keys (first 5): %s", missing[:5])

    @property
    def name(self) -> str:
        return "usfm"

    @property
    def embed_dim(self) -> int:
        return self._d

    def encode_image(self, images: Tensor) -> dict:
        """
        Encode images with the USFM backbone.

        Parameters
        ----------
        images : (B, C, H, W)

        Returns
        -------
        dict with:
            cls          : (B, D)      CLS token from the final ViT layer
            patch_tokens : (B, N, D)   all patch tokens
        """
        with torch.no_grad():
            dev = next(self.backbone.parameters()).device
            images = self._resize_images(images.to(dev))
            cls, patch_tokens = self._forward_tokens(self.backbone, images)

        return {"cls": cls, "patch_tokens": patch_tokens}

    def encode_video(self, clips: Tensor) -> dict:
        B, T, C, H, W = clips.shape
        frames       = clips.reshape(B * T, C, H, W)
        enc          = self.encode_image(frames)
        frame_tokens = enc["cls"].reshape(B, T, -1)
        clip_cls     = self.temporal_pool(frame_tokens)
        return {"clip_cls": clip_cls, "frame_tokens": frame_tokens, "tube_tokens": None}

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return self.temporal_pool.parameters()

    def to(self, *args, **kwargs) -> "USFMEncoder":
        self.backbone.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "USFMEncoder":
        self.backbone.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "USFMEncoder":
        self.backbone.eval()
        self.temporal_pool.train(mode)
        return self
