"""
finetune/backbones/openus_encoder.py  ·  OpenUS ultrasound foundation model encoder
====================================================================================

OpenUS uses VMamba (Vision Mamba) as its backbone, pre-trained with
self-adaptive masked contrastive learning on ~308K ultrasound images.

Paper / repo: https://github.com/XZheng0427/OpenUS

Architecture:
    VMamba-Small backbone (vmamba_small, hidden_dim=768)
    Checkpoint key for the SSL teacher weights: "teacher"
    Also requires VMamba-Small ImageNet pre-trained weights for initialisation.

Setup:
    1. Clone vendor code:
           git clone https://github.com/XZheng0427/OpenUS \\
               finetune/backbones/vendor/openus

    2. Install VMamba CUDA extension (requires CUDA 12.x + PyTorch 2.2):
           pip install https://github.com/state-spaces/mamba/releases/download/ \\
               v2.2.4/mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

    3. Set checkpoint paths (or use env overrides):
           US_OPENUS_CHECKPOINT        = .../checkpoints/Ablations/openus_cpt0150.pth
           US_OPENUS_VMAMBA_CHECKPOINT = .../checkpoints/Ablations/vssm_small_0229_ckpt_epoch_222.pth

Checkpoint loading (from official eval scripts):
    ckpt = torch.load(openus_checkpoint)
    state_dict = ckpt["teacher"]
    model.load_state_dict(state_dict, strict=False)
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor

from finetune.backbones.base import BackboneEncoder
from models.heads.temporal_pool import TemporalAttentionPool

log = logging.getLogger(__name__)

_VENDOR_PATH    = Path(__file__).parent / "vendor" / "openus"
_VMAMBA_EMBED   = 768    # VMamba-Small final feature dimension
_OPENUS_REPO    = "https://github.com/XZheng0427/OpenUS"
_MAMBA_WHEEL    = (
    "https://github.com/state-spaces/mamba/releases/download/v2.2.4/"
    "mamba_ssm-2.2.4+cu12torch2.2cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
)


def _ensure_vendor_clone() -> bool:
    """Clone the OpenUS repo into vendor/openus if absent. Returns True on success."""
    if _VENDOR_PATH.exists():
        return True
    log.info("[openus] Cloning %s → %s", _OPENUS_REPO, _VENDOR_PATH)
    try:
        _VENDOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth=1", _OPENUS_REPO, str(_VENDOR_PATH)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            log.error("[openus] git clone failed:\n%s", result.stderr)
            return False
        log.info("[openus] Clone succeeded.")
        return True
    except Exception as exc:
        log.error("[openus] git clone error: %s", exc)
        return False


class OpenUSEncoder(BackboneEncoder):
    """
    OpenUS general-purpose ultrasound foundation model encoder.

    Uses the VMamba-Small backbone with weights loaded from the OpenUS
    teacher checkpoint (checkpoint_key="teacher").

    Parameters
    ----------
    checkpoint : str
        Path to the OpenUS pre-trained checkpoint (openus_cpt0150.pth).
    vmamba_checkpoint : str or None
        Path to VMamba-Small ImageNet pre-trained weights
        (vssm_small_0229_ckpt_epoch_222.pth).  Optional — if not provided,
        the model is initialised from scratch before loading OpenUS weights.
    embed_dim : int
        VMamba-Small feature dimension (default 768).
    temporal_dropout : float
        Dropout for TemporalAttentionPool used in encode_video().
    """

    def __init__(
        self,
        checkpoint:        str,
        vmamba_checkpoint: Optional[str] = None,
        embed_dim:         int   = _VMAMBA_EMBED,
        temporal_dropout:  float = 0.0,
    ):
        self._d                 = embed_dim
        self._ckpt              = checkpoint
        self._vmamba_ckpt       = vmamba_checkpoint

        self.backbone      = self._load_model(checkpoint, vmamba_checkpoint)
        self.temporal_pool = TemporalAttentionPool(self._d, dropout=temporal_dropout)

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()
        log.info(f"[openus] Ready (embed_dim={self._d})")

    def _load_model(self, checkpoint: str, vmamba_checkpoint: Optional[str]) -> nn.Module:
        # Ensure mamba_ssm is available
        try:
            import mamba_ssm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "OpenUSEncoder requires the mamba_ssm CUDA extension.\n"
                "On x86_64 + CUDA 12 + PyTorch 2.2:\n"
                f"  pip install {_MAMBA_WHEEL}\n"
                "On GH200 (aarch64), build from source inside the ultr-ai GPU container:\n"
                "  pip install packaging ninja einops\n"
                "  pip install causal-conv1d --no-build-isolation\n"
                "  pip install mamba-ssm --no-build-isolation\n"
                "(Do NOT use the x86_64 wheel on aarch64.)"
            ) from exc

        # Ensure vendor code is cloned
        if not _ensure_vendor_clone():
            raise ImportError(
                f"OpenUS vendor code not found at {_VENDOR_PATH}.\n"
                f"Please clone {_OPENUS_REPO} into {_VENDOR_PATH} and re-run."
            )

        if str(_VENDOR_PATH) not in sys.path:
            sys.path.insert(0, str(_VENDOR_PATH))

        # Build VMamba model — try multiple import paths across repo versions
        model = self._build_vmamba_model()

        # Load VMamba ImageNet pre-trained weights (backbone initialisation)
        if vmamba_checkpoint and Path(vmamba_checkpoint).exists():
            log.info("[openus] Loading VMamba backbone weights: %s", vmamba_checkpoint)
            vmamba_ckpt = torch.load(vmamba_checkpoint, map_location="cpu")
            # VMamba checkpoints may be wrapped in 'model' key
            vm_state = vmamba_ckpt.get("model", vmamba_ckpt.get("state_dict", vmamba_ckpt))
            missing, unexpected = model.load_state_dict(vm_state, strict=False)
            log.info("[openus] VMamba weights loaded  missing=%d  unexpected=%d",
                     len(missing), len(unexpected))

        # Load OpenUS teacher weights
        log.info("[openus] Loading OpenUS checkpoint: %s", checkpoint)
        ckpt = torch.load(checkpoint, map_location="cpu")
        if "teacher" in ckpt:
            state = ckpt["teacher"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        # Strip module. prefix if present (DDP checkpoints)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info("[openus] OpenUS weights loaded  missing=%d  unexpected=%d",
                 len(missing), len(unexpected))

        return model

    def _build_vmamba_model(self) -> nn.Module:
        """
        Build a VMamba-Small model from the vendor/openus repo.

        Imports VSSM from vendor/openus/vmamba.py. vendor/openus/ is
        added to sys.path by _load_model before this is called.

        Returns
        -------
        nn.Module
            VMamba-Small with depths=[2,2,9,2], dims=[96,192,384,768].
        """
        try:
            from vmamba import VSSM  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Could not import VSSM from vendor/openus/vmamba.py.\n"
                f"Ensure the OpenUS repo is cloned at {_VENDOR_PATH} and that\n"
                "mamba_ssm is installed (requires CUDA).\n"
                f"Original error: {exc}"
            ) from exc

        model = VSSM(
            patch_size=4, in_chans=3,
            depths=[2, 2, 9, 2], dims=[96, 192, 384, 768],
            ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto",
            mlp_ratio=4.0, patch_norm=True, use_checkpoint=False,
        )
        log.info("[openus] VMamba-Small built via vmamba.VSSM")
        return model

    @property
    def name(self) -> str:
        return "openus"

    @property
    def embed_dim(self) -> int:
        return self._d

    def encode_image(self, images: Tensor) -> dict:
        """
        Encode images with the OpenUS VMamba backbone.

        For VMamba, the output is a spatial feature map or a set of hierarchical
        features.  We global-average-pool to obtain the cls token.

        Parameters
        ----------
        images : (B, C, H, W)

        Returns
        -------
        dict with:
            cls          : (B, D)
            patch_tokens : (B, N, D) or None
        """
        with torch.no_grad():
            out = self.backbone(images)

        if isinstance(out, (list, tuple)):
            feat = out[-1]   # last stage feature map
        else:
            feat = out

        if feat.dim() == 4:
            # (B, C, H, W) → global avg pool
            cls = feat.mean(dim=(2, 3))
            B, C, H, W = feat.shape
            patch_tokens = feat.reshape(B, C, H * W).permute(0, 2, 1)
        elif feat.dim() == 3:
            # (B, N, D) sequence output
            cls          = feat.mean(dim=1)
            patch_tokens = feat
        else:
            cls          = feat
            patch_tokens = None

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

    def to(self, *args, **kwargs) -> "OpenUSEncoder":
        self.backbone.to(*args, **kwargs)
        self.temporal_pool.to(*args, **kwargs)
        return self

    def eval(self) -> "OpenUSEncoder":
        self.backbone.eval()
        self.temporal_pool.eval()
        return self

    def train(self, mode: bool = True) -> "OpenUSEncoder":
        self.backbone.eval()
        self.temporal_pool.train(mode)
        return self
