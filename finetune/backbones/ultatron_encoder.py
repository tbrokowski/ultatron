"""
finetune/backbones/ultatron_encoder.py  ·  Ultatron checkpoint encoder
"""
from __future__ import annotations
import logging
from typing import Iterator
import torch
import torch.nn as nn
from torch import Tensor
from finetune.backbones.base import BackboneEncoder

log = logging.getLogger(__name__)


class UltatronEncoder(BackboneEncoder):
    """Loads an Ultatron dual-branch checkpoint via BackboneEncoder interface."""

    def __init__(self, train_cfg: dict, checkpoint: str, label: str, device: str = "cpu"):
        from models import ModelConfig, build_image_branch, build_video_branch
        cfg = ModelConfig.from_dict(train_cfg)
        cfg.frozen_teacher = None
        self._label = label
        log.info(f"[{label}] Building image backbone: {cfg.image_backbone}")
        self.img_branch = build_image_branch(cfg, device=device)
        log.info(f"[{label}] Building video backbone: {cfg.video_backbone}")
        self.vid_branch = build_video_branch(cfg, device=device)
        log.info(f"[{label}] Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu")
        self.img_branch.teacher.load_state_dict(ckpt["img_teacher"], strict=True)
        self.vid_branch.teacher.load_state_dict(ckpt["vid_teacher"], strict=True)
        log.info(f"[{label}] Loaded step={ckpt.get('global_step', '?')}")
        for p in self.img_branch.parameters():
            p.requires_grad_(False)
        for p in self.vid_branch.parameters():
            p.requires_grad_(False)
        self.img_branch.eval()
        self.vid_branch.eval()

    @property
    def name(self) -> str:
        return self._label

    @property
    def embed_dim(self) -> int:
        return self.img_branch.embed_dim

    @property
    def video_embed_dim(self) -> int:
        return self.vid_branch.embed_dim

    @property
    def is_video_native(self) -> bool:
        return True

    @torch.no_grad()
    def encode_image(self, images: Tensor) -> dict:
        return self.img_branch.forward_teacher(images)

    @torch.no_grad()
    def encode_video(self, clips: Tensor) -> dict:
        return self.vid_branch.forward_teacher(clips)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return iter([])

    def to(self, *args, **kwargs):
        self.img_branch.to(*args, **kwargs)
        self.vid_branch.to(*args, **kwargs)
        return self

    def eval(self):
        self.img_branch.eval()
        self.vid_branch.eval()
        return self

    def train(self, mode: bool = True):
        self.img_branch.eval()
        self.vid_branch.eval()
        return self


class UltatronBranchEncoder(BackboneEncoder):
    """
    Backward-compatible wrapper around pre-built ImageBranch / VideoBranch.
    Used by FinetuneExperiment.setup() when img_branch/vid_branch are passed
    directly (legacy code path from scripts/finetune.py and trainer).
    """

    def __init__(self, img_branch, vid_branch=None, label: str = "ultatron"):
        self.img_branch = img_branch
        self.vid_branch = vid_branch
        self._label     = label

    @property
    def name(self) -> str:
        return self._label

    @property
    def embed_dim(self) -> int:
        return self.img_branch.embed_dim

    @property
    def video_embed_dim(self) -> int:
        if self.vid_branch is not None:
            return self.vid_branch.embed_dim
        return self.img_branch.embed_dim

    @property
    def is_video_native(self) -> bool:
        return self.vid_branch is not None

    @torch.no_grad()
    def encode_image(self, images: Tensor) -> dict:
        return self.img_branch.forward_teacher(images)

    @torch.no_grad()
    def encode_video(self, clips: Tensor) -> dict:
        if self.vid_branch is None:
            raise RuntimeError("UltatronBranchEncoder: vid_branch not provided.")
        return self.vid_branch.forward_teacher(clips)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return iter([])

    def to(self, *args, **kwargs):
        if self.img_branch is not None:
            self.img_branch.to(*args, **kwargs)
        if self.vid_branch is not None:
            self.vid_branch.to(*args, **kwargs)
        return self

    def eval(self):
        if self.img_branch is not None:
            self.img_branch.eval()
        if self.vid_branch is not None:
            self.vid_branch.eval()
        return self

    def train(self, mode: bool = True):
        return self.eval()
