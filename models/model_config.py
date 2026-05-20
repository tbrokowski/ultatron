"""
models/model_config.py  ·  ModelConfig + branch factories
==========================================================

ModelConfig
-----------
A dataclass that fully specifies which models are used for each branch.
It is loaded from the YAML config's ``model:`` section and passed to
build_image_branch / build_video_branch.

Example YAML (in data_config.yaml under a ``model:`` top-level key):

    model:
      image_backbone:    dinov3_l         # registered key in image registry
      video_backbone:    vjepa2_l         # registered key in video registry
      frozen_teacher:    dinov3_7b        # null to disable
      ema_momentum:      0.9995
      n_prototypes:      256
      align_dim:         256              # cross-branch alignment projection dim
      dtype:             bfloat16         # float32 | bfloat16 | float16
      hf_cache_dir:      null             # override HF cache (use CSCS store path)
      trainable_image_layers: 4           # null full tune, 0 frozen, K last blocks
      trainable_video_layers: 4

build_image_branch / build_video_branch
----------------------------------------
These are the only public entry points that train.py needs to call.
They accept a ModelConfig (or a raw dict from YAML) and return fully
constructed, device-placed ImageBranch / VideoBranch instances.

The 7B frozen teacher is placed on a dedicated device 
"""
from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .registry import (
    build_image_backbone as _build_img_bb,
    build_video_backbone as _build_vid_bb,
    build_frozen_teacher as _build_teacher,
)
from .branches.image_branch import ImageBranch
from .branches.video_branch import VideoBranch
from .branches.shared import CrossBranchDistillation, PrototypeHead

log = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float32":  torch.float32,
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
}


@dataclass
class ModelConfig:
    """
    Full model specification.  Loaded from the YAML ``model:`` section.

    Fields
    ------
    image_backbone   : str   key in the image backbone registry
    video_backbone   : str   key in the video backbone registry
    frozen_teacher   : str | None  key in the frozen teacher registry (or null)
    ema_momentum     : float  EMA momentum for student→teacher update
    n_prototypes     : int    number of prototype vectors
    align_dim        : int    cross-branch projection dimension
    dtype            : str    "float32" | "bfloat16" | "float16"
    hf_cache_dir     : str | None  HF model cache override
    trainable_*_layers : int | None  null full tune, 0 frozen, K last blocks
    """
    image_backbone:  str   = "dinov3_l"
    video_backbone:  str   = "vjepa2_l"
    frozen_teacher:  Optional[str] = "dinov3_7b"
    ema_momentum:    float = 0.9995
    n_prototypes:    int   = 1024
    align_dim:       int   = 256
    dtype:           str   = "bfloat16"
    hf_cache_dir:    Optional[str] = None
    use_gradient_checkpointing: bool = True
    # null = full backbone fine-tuning. 0 = frozen backbone. K > 0 = train
    # only the last K transformer blocks plus final norms/predictors.
    trainable_image_layers: Optional[int] = None
    trainable_video_layers: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Build from a raw YAML dict (the ``model:`` section)."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            return _DTYPE_MAP[self.dtype]
        except KeyError:
            raise ValueError(
                f"Unknown dtype '{self.dtype}'. "
                f"Choose from: {list(_DTYPE_MAP.keys())}"
            )


_BLOCK_INDEX_PATTERNS = (
    re.compile(r"(?:^|\.)(?:encoder\.)?(?:layer|layers|blocks)\.(\d+)\."),
    re.compile(r"(?:^|\.)(?:_vit|_model|model)\.(?:encoder\.)?(?:layer|layers|blocks)\.(\d+)\."),
)


def _block_index(param_name: str) -> Optional[int]:
    for pattern in _BLOCK_INDEX_PATTERNS:
        match = pattern.search(param_name)
        if match:
            return int(match.group(1))
    return None


def _apply_trainability(module: nn.Module, last_k: Optional[int], label: str) -> None:
    """Best-effort last-K transformer tuning for HF-style ViT backbones."""
    params = list(module.named_parameters())
    if last_k is None:
        for _, param in params:
            param.requires_grad_(True)
        log.info("%s backbone trainability: full fine-tuning", label)
        return

    if last_k < 0:
        raise ValueError(f"{label} trainable layer count must be >= 0 or null")

    if last_k == 0:
        for _, param in params:
            param.requires_grad_(False)
        log.info("%s backbone trainability: frozen backbone", label)
        return

    block_ids = [_block_index(name) for name, _ in params]
    concrete = [idx for idx in block_ids if idx is not None]
    if not concrete:
        log.warning(
            "%s backbone trainability requested last %d layers, but no "
            "transformer block indices were detected; leaving backbone fully trainable.",
            label, last_k,
        )
        for _, param in params:
            param.requires_grad_(True)
        return

    max_block = max(concrete)
    cutoff = max(0, max_block - last_k + 1)
    trainable_markers = (
        "norm", "layernorm", "ln_", "fc_norm", "final_norm",
        "head", "predictor", "adapter", "proj",
    )
    n_trainable = 0
    n_total = 0
    for name, param in params:
        idx = _block_index(name)
        lower = name.lower()
        train = (idx is not None and idx >= cutoff) or any(m in lower for m in trainable_markers)
        param.requires_grad_(train)
        n_total += param.numel()
        if train:
            n_trainable += param.numel()

    log.info(
        "%s backbone trainability: last %d/%d blocks plus norms/predictors "
        "(%.1fM/%.1fM params trainable)",
        label,
        min(last_k, max_block + 1),
        max_block + 1,
        n_trainable / 1e6,
        n_total / 1e6,
    )


def build_image_branch(
    cfg: ModelConfig,
    device: str = "cuda",
) -> ImageBranch:
    """
    Build and return a fully constructed ImageBranch.

    Steps
    -----
    1. Instantiate student backbone from registry (cfg.image_backbone).
    2. Deep-copy student to create EMA teacher (identical initialisation).
    3. Optionally instantiate frozen distillation teacher (cfg.frozen_teacher).
    4. Wrap in ImageBranch and move to device.

    The frozen teacher is placed on the last available GPU (or same device
    if only one GPU) to isolate its memory from the student/teacher.
    """
    dtype     = cfg.torch_dtype
    cache_dir = cfg.hf_cache_dir

    log.info(f"Building ImageBranch: backbone={cfg.image_backbone!r}, "
             f"frozen_teacher={cfg.frozen_teacher!r}, dtype={cfg.dtype}")

    # ── Student ───────────────────────────────────────────────────────────────
    student = _build_img_bb(cfg.image_backbone, dtype=dtype, hf_cache_dir=cache_dir)
    _apply_trainability(student, cfg.trainable_image_layers, "Image")

    # ── EMA teacher: deep-copy of student (before enabling grad checkpoint) ────
    log.info("  Copying student to EMA teacher ...")
    teacher = copy.deepcopy(student)

    # ── Gradient checkpointing on student only ────────────────────────────────
    if cfg.use_gradient_checkpointing:
        student.enable_gradient_checkpointing()

    # ── Frozen distillation teacher ───────────────────────────────────────────
    teacher_d = None
    if cfg.frozen_teacher:
        # Place on last GPU to isolate memory; fall back to same device
        if torch.cuda.device_count() > 1:
            teacher_device = f"cuda:{torch.cuda.device_count() - 1}"
        else:
            teacher_device = device
        log.info(f"  Loading frozen teacher {cfg.frozen_teacher!r} "
                 f"on {teacher_device} ...")
        teacher_d = _build_teacher(
            cfg.frozen_teacher,
            dtype=torch.bfloat16,      # always bf16 for large teachers
            hf_cache_dir=cache_dir,
            device=teacher_device,
        )
        teacher_d = teacher_d.to(device=teacher_device, dtype=torch.bfloat16)

    # ── Assemble branch ───────────────────────────────────────────────────────
    branch = ImageBranch(student=student, teacher=teacher, teacher_d=teacher_d)
    branch = branch.to(device=device, dtype=dtype)

    # Frozen teacher stays on its own device
    if teacher_d is not None:
        branch.teacher_d = branch.teacher_d.to(dtype=torch.bfloat16)

    n_student = sum(p.numel() for p in branch.student.parameters())
    log.info(f"  ImageBranch ready.  Student params: {n_student/1e6:.1f}M  "
             f"Hidden dim: {branch.embed_dim}")
    return branch


def build_video_branch(
    cfg: ModelConfig,
    device: str = "cuda",
) -> VideoBranch:
    """
    Build and return a fully constructed VideoBranch.

    Steps
    -----
    1. Instantiate student backbone from registry (cfg.video_backbone).
    2. Deep-copy student to EMA teacher.
    3. Wrap in VideoBranch and move to device.
    """
    dtype     = cfg.torch_dtype
    cache_dir = cfg.hf_cache_dir

    log.info(f"Building VideoBranch: backbone={cfg.video_backbone!r}, "
             f"dtype={cfg.dtype}")

    student = _build_vid_bb(cfg.video_backbone, dtype=dtype, hf_cache_dir=cache_dir)
    _apply_trainability(student, cfg.trainable_video_layers, "Video")

    log.info("  Copying student to EMA teacher ...")
    teacher = copy.deepcopy(student)

    if cfg.use_gradient_checkpointing:
        student.enable_gradient_checkpointing()

    branch = VideoBranch(student, teacher)
    branch = branch.to(device=device, dtype=dtype)

    n_student = sum(p.numel() for p in branch.student.parameters())
    log.info(f"  VideoBranch ready.  Student params: {n_student/1e6:.1f}M  "
             f"Hidden dim: {branch.embed_dim}")
    return branch


def build_heads(
    cfg: ModelConfig,
    device: str = "cuda",
) -> dict:
    """
    Build the shared cross-branch and prototype heads.

    Returns dict with keys:
      cross_distill : CrossBranchDistillation
      proto_head    : PrototypeHead
    """
    # We need the actual embed dims — build dummy backbones just to read config,
    # then throw them away.  In practice train.py builds the branches first
    # and passes the dims in directly; this helper is a convenience for tests.
    img_bb = _build_img_bb(cfg.image_backbone, dtype=torch.float32)
    vid_bb = _build_vid_bb(cfg.video_backbone, dtype=torch.float32)
    img_dim = img_bb.hidden_size
    vid_dim = vid_bb.hidden_size
    del img_bb, vid_bb

    dtype = cfg.torch_dtype
    cross = CrossBranchDistillation(img_dim, vid_dim, cfg.align_dim).to(device=device, dtype=dtype)
    proto = PrototypeHead(cfg.align_dim, cfg.n_prototypes).to(device=device, dtype=dtype)
    return {"cross_distill": cross, "proto_head": proto}
