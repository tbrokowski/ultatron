"""
train/training_integration.py  — DEPRECATED REDIRECT
======================================================

This file has been consolidated.  All functionality has moved to dedicated
canonical modules:

  GramTeacher / gram_loss  →  train.gram
  build_datamodule         →  train.trainer (or construct USFoundationDataModule
                               directly from data.pipeline.datamodule)
  phase1_step … phase4_step →  train.phase_steps

These re-exports are kept for backward compatibility only.
New code should import directly from the canonical locations above.
"""
from __future__ import annotations

# Re-export Gram utilities from their canonical home
from .gram import GramTeacher, gram_loss  # noqa: F401

# Re-export phase steps from their canonical home
from .phase_steps import (  # noqa: F401
    phase1_step,
    phase2_step,
    phase3_step,
    phase4_step,
)

# Convenience datamodule factory — kept here for callers that already import it
from data.pipeline.datamodule import USFoundationDataModule
from data.pipeline.transforms import ImageSSLTransformConfig, VideoSSLTransformConfig


def build_datamodule(cfg: dict) -> USFoundationDataModule:
    """
    Construct a USFoundationDataModule from a plain config dict.

    Canonical location: kept here for backward compat; prefer instantiating
    USFoundationDataModule directly in new code.
    """
    img_raw  = dict(cfg["transforms"]["image"])
    vid_raw  = dict(cfg["transforms"]["video"])
    return USFoundationDataModule(
        manifest_path           = cfg["manifest"]["path"],
        image_batch_size        = cfg["loaders"]["image_batch_size"],
        video_batch_size        = cfg["loaders"]["video_batch_size"],
        num_workers             = cfg["loaders"]["num_workers"],
        pin_memory              = cfg["loaders"]["pin_memory"],
        patch_size              = cfg["transforms"]["patch_size"],
        total_training_steps    = cfg["curriculum"]["total_training_steps"],
        image_samples_per_epoch = cfg["curriculum"]["image_samples_per_epoch"],
        video_samples_per_epoch = cfg["curriculum"]["video_samples_per_epoch"],
        anatomy_weights         = cfg.get("anatomy_weights", {}),
        root_remap              = cfg["manifest"].get("root_remap", {}),
        image_cfg               = ImageSSLTransformConfig(**img_raw),
        video_cfg               = VideoSSLTransformConfig(**vid_raw),
    )


__all__ = [
    "GramTeacher", "gram_loss",
    "phase1_step", "phase2_step", "phase3_step", "phase4_step",
    "build_datamodule",
]
