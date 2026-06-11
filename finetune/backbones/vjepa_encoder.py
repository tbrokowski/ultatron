"""
finetune/backbones/vjepa_encoder.py  ·  Standalone V-JEPA2 encoder
===================================================================

Wraps the existing V-JEPA2 VideoBranch as a BackboneEncoder for comparison.
Does NOT require an Ultatron dual-branch checkpoint — uses HuggingFace
pretrained weights directly, or a standalone V-JEPA2 checkpoint if provided.

For image experiments, the image is encoded as a 1-frame clip.
"""
from __future__ import annotations
import logging
from typing import Iterator, Optional
import torch
import torch.nn as nn
from torch import Tensor
from finetune.backbones.base import BackboneEncoder

log = logging.getLogger(__name__)

_VJEPA2_EMBED_DIMS = {"vjepa2_l": 1024, "vjepa2_h": 1280, "vjepa2_g": 1408}


class VJEPAEncoder(BackboneEncoder):
    """
    Standalone V-JEPA2 video encoder for backbone comparison.

    Parameters
    ----------
    variant : str
        Registry key: 'vjepa2_l', 'vjepa2_h', or 'vjepa2_g'.
    checkpoint : str or None
        Path to a standalone checkpoint containing a 'vid_teacher' key.
        If None, uses HuggingFace pretrained weights.
    device : str
    hf_cache_dir : str or None
    """

    def __init__(
        self,
        variant:      str = "vjepa2_l",
        checkpoint:   Optional[str] = None,
        device:       str = "cpu",
        hf_cache_dir: Optional[str] = None,
    ):
        from models import ModelConfig, build_video_branch

        self._variant = variant
        self._embed   = _VJEPA2_EMBED_DIMS.get(variant, 1024)

        cfg = ModelConfig.from_dict({
            "image_backbone": "dinov3_l",   # placeholder; only vid_branch used
            "video_backbone":  variant,
            "frozen_teacher":  None,
            "hf_cache_dir":    hf_cache_dir,
        })

        log.info(f"[vjepa/{variant}] Building video branch")
        self.vid_branch = build_video_branch(cfg, device=device)

        if checkpoint is not None:
            log.info(f"[vjepa/{variant}] Loading checkpoint: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location="cpu")
            key  = "vid_teacher" if "vid_teacher" in ckpt else "model"
            self.vid_branch.teacher.load_state_dict(ckpt[key], strict=True)
            log.info(f"[vjepa/{variant}] Checkpoint loaded")
        else:
            log.info(f"[vjepa/{variant}] Using HuggingFace pretrained weights")

        for p in self.vid_branch.parameters():
            p.requires_grad_(False)
        self.vid_branch.eval()

    @property
    def name(self) -> str:
        return self._variant

    @property
    def embed_dim(self) -> int:
        return self._embed

    @property
    def video_embed_dim(self) -> int:
        return self._embed

    @property
    def is_video_native(self) -> bool:
        return True

    @torch.no_grad()
    def encode_image(self, images: Tensor) -> dict:
        """Encode image as a single-frame clip; return clip_cls as cls."""
        clips = images.unsqueeze(1)              # (B, 1, C, H, W)
        out   = self.vid_branch.forward_teacher(clips)
        return {"cls": out["clip_cls"], "patch_tokens": out.get("tube_tokens")}

    @torch.no_grad()
    def encode_video(self, clips: Tensor) -> dict:
        return self.vid_branch.forward_teacher(clips)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return iter([])

    def to(self, *args, **kwargs):
        self.vid_branch.to(*args, **kwargs)
        return self

    def eval(self):
        self.vid_branch.eval()
        return self

    def train(self, mode: bool = True):
        self.vid_branch.eval()
        return self
