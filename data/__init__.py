"""
Ultrasound Foundation Model – Data Pipeline
============================================

Quick-start:

    from data import USFoundationDataModule
    from configs import load_data_config

    dm = USFoundationDataModule(**load_data_config("configs/data_config.yaml"))
    dm.setup()

    # Phase 1: image branch only
    for batch in dm.image_loader():
        global_crops = batch["global_crops"]   # B × n_g × 3 × H × H
        patch_masks  = batch["patch_masks"]    # B × ph × pw  (bool)
        # patch_masks now derived from frequency-domain energy loss,
        # not random spatial sampling. Shape and semantics unchanged.
        ...

    # Phase 2: video branch only
    for batch in dm.video_loader():
        full_clip    = batch["full_clips"]     # B × T × 3 × H × W
        visible_clip = batch["visible_clips"]  # B × T × 3 × H × W  (freq-masked)
        tube_mask    = batch["tube_masks"]     # B × T × ph × pw
        ...

    # Phase 3: joint (yields DualStreamBatch)
    for dual in dm.combined_loader():
        img_batch = dual.image_batch
        vid_batch = dual.video_batch
        dm.update_step(global_step)  # advances curriculum
"""

from .schema.manifest import USManifestEntry, Instance, load_manifest, manifest_stats
from .pipeline.dataset import ImageSSLDataset, VideoSSLDataset, USFoundationDataset
from .pipeline.datamodule import USFoundationDataModule
from .pipeline.collators import ImageSSLCollator, VideoSSLCollator, DualStreamBatch
from .pipeline.samplers import CombinedSampler, AnatomyStratifiedSampler, CurriculumSampler
from .pipeline.transforms import (
    # Config dataclasses
    ImageSSLTransform, ImageSSLTransformConfig,
    VideoSSLTransform, VideoSSLTransformConfig,
    FreqMaskConfig,
    # Core frequency masking functions
    freq_mask_image,
    freq_mask_image_alp,
    freq_mask_video,
    # Utilities
    to_canonical_tensor,
    add_speckle_noise,
    # Legacy spatial masking (backward compat — prefer freq_mask_* above)
    priority_weighted_mask,
    random_tube_mask,
    random_patch_mask,
)
from .labels.label_spec import LabelSpec, TaskType, LossType, TaskConfig

__all__ = [
    # Manifest
    "USManifestEntry", "Instance", "load_manifest", "manifest_stats",
    # Datasets
    "ImageSSLDataset", "VideoSSLDataset", "USFoundationDataset",
    # DataModule
    "USFoundationDataModule",
    # Collators
    "ImageSSLCollator", "VideoSSLCollator", "DualStreamBatch",
    # Samplers
    "CombinedSampler", "AnatomyStratifiedSampler", "CurriculumSampler",
    # Transform classes + configs
    "ImageSSLTransform", "ImageSSLTransformConfig",
    "VideoSSLTransform", "VideoSSLTransformConfig",
    "FreqMaskConfig",
    # Frequency masking functions
    "freq_mask_image",
    "freq_mask_image_alp",
    "freq_mask_video",
    # Utilities
    "to_canonical_tensor",
    "add_speckle_noise",
    # Legacy (kept for backward compat)
    "priority_weighted_mask",
    "random_tube_mask",
    "random_patch_mask",
    # Label specification
    "LabelSpec", "TaskType", "LossType", "TaskConfig",
]
