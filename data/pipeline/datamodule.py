"""
datamodule.py  ·  Top-level DataModule (Lightning-compatible)
=============================================================

USFoundationDataModule
  - Loads the master manifest
  - Builds ImageSSLDataset + VideoSSLDataset
  - Wraps both with CombinedSampler (stratified + curriculum)
  - Exposes image_loader() and video_loader() for two-stream training
  - Exposes fine_tune_loader() for head training with full supervision

The trainer calls:
    dm.update_step(global_step)   # at each step to advance curriculum
    dm.current_alpha()            # to set ALP alpha in datasets
    dm.current_mask_ratio()       # to set mask ratio in video dataset

Phase coupling (matches architecture training phases):
    Phase 1:  image_loader() only
    Phase 2:  video_loader() only
    Phase 3:  both via combined_loader()
    Phase 4:  fine_tune_loader()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import DataLoader

from data.schema.manifest import USManifestEntry, load_manifest, manifest_stats
from data.pipeline.dataset import ImageSSLDataset, VideoSSLDataset, PairedSSLDataset
from data.pipeline.collators import ImageSSLCollator, VideoSSLCollator, DualStreamBatch
from data.pipeline.samplers import CombinedSampler
from data.pipeline.transforms import ImageSSLTransformConfig, VideoSSLTransformConfig
from data.pipeline.alp_interface import ALPReader, NullALPReader

log = logging.getLogger(__name__)


class USFoundationDataModule:
    def __init__(
        self,
        manifest_path: str,
        # Loader settings
        image_batch_size: int = 256,
        video_batch_size: int = 32,
        num_workers: int = 8,
        pin_memory: bool = True,
        # Transform settings
        image_cfg: Optional[ImageSSLTransformConfig] = None,
        video_cfg: Optional[VideoSSLTransformConfig] = None,
        patch_size: int = 16,
        # Curriculum
        total_training_steps: int = 300_000,
        image_samples_per_epoch: int = 500_000,
        video_samples_per_epoch: int = 100_000,
        curriculum_stage_fracs: Optional[List[float]] = None,
        alp_alpha_init: float = 0.1,
        alp_alpha_final: float = 0.9,
        alp_guidance_threshold_init: float = 0.1,
        alp_guidance_threshold_final: float = 0.9,
        alp_n_frames: Optional[List[int]] = None,
        alp_reader: Optional[ALPReader] = None,
        hardness_temperature: float = 1.0,
        # Filtering
        anatomy_weights: Optional[Dict[str, float]] = None,
        exclude_datasets: Optional[List[str]] = None,
        split: str = "train",
        root_remap: Optional[Dict[str, str]] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_batch_size = image_batch_size
        self.video_batch_size = video_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.patch_size = patch_size
        self.total_steps = total_training_steps
        self.image_samples = image_samples_per_epoch
        self.video_samples = video_samples_per_epoch
        self.curriculum_stage_fracs = curriculum_stage_fracs
        self.alp_alpha_init = alp_alpha_init
        self.alp_alpha_final = alp_alpha_final
        self.alp_guidance_threshold_init = alp_guidance_threshold_init
        self.alp_guidance_threshold_final = alp_guidance_threshold_final
        self.alp_n_frames = alp_n_frames
        self.alp_reader = alp_reader or NullALPReader()
        self._hardness_temperature = hardness_temperature
        self.anatomy_weights = anatomy_weights
        self.exclude_datasets = set(exclude_datasets or ())
        self.split = split
        self.root_remap = root_remap or {}

        self.image_cfg = image_cfg or ImageSSLTransformConfig()
        self.video_cfg = video_cfg or VideoSSLTransformConfig()

        self._image_entries: List[USManifestEntry] = []
        self._video_entries: List[USManifestEntry] = []
        self._paired_entries: List[USManifestEntry] = []
        self._image_sampler: Optional[CombinedSampler] = None
        self._video_sampler: Optional[CombinedSampler] = None
        self._paired_sampler: Optional[CombinedSampler] = None
        self._image_dataset: Optional[ImageSSLDataset] = None
        self._video_dataset: Optional[VideoSSLDataset] = None
        self._paired_dataset: Optional[PairedSSLDataset] = None

        self._setup_done = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        if self._setup_done: return
        log.info(f"Loading manifest: {self.manifest_path}")
        all_entries = load_manifest(self.manifest_path, split=self.split)
        if self.exclude_datasets:
            before = len(all_entries)
            all_entries = [e for e in all_entries if e.dataset_id not in self.exclude_datasets]
            log.info(
                "Excluded datasets %s: %d → %d entries",
                sorted(self.exclude_datasets), before, len(all_entries),
            )
        log.info(f"Manifest loaded: {len(all_entries)} entries")

        stats = manifest_stats(all_entries)
        log.info(f"Dataset stats: {stats}")

        # Split into two streams + paired stream (Phase 3)
        self._image_entries = [
            e for e in all_entries if e.ssl_stream in ("image", "both")
        ]
        self._video_entries = [
            e for e in all_entries if e.ssl_stream in ("video", "both")
        ]
        self._paired_entries = [
            e for e in all_entries if e.ssl_stream == "both"
        ]
        log.info(f"Image stream: {len(self._image_entries)} | "
                 f"Video stream: {len(self._video_entries)} | "
                 f"Paired stream: {len(self._paired_entries)}")

        # Datasets
        self._image_dataset = ImageSSLDataset(
            self._image_entries,
            cfg=self.image_cfg,
            root_remap=self.root_remap,
            alp_reader=self.alp_reader,
        )
        self._video_dataset = VideoSSLDataset(
            self._video_entries,
            cfg=self.video_cfg,
            patch_size=self.patch_size,
            root_remap=self.root_remap,
        )
        self._paired_dataset = PairedSSLDataset(
            self._paired_entries,
            img_cfg=self.image_cfg,
            vid_cfg=self.video_cfg,
            patch_size=self.patch_size,
            root_remap=self.root_remap,
            alp_reader=self.alp_reader,
        )

        sampler_kw = dict(
            global_step=0,
            total_steps=self.total_steps,
            anatomy_weights=self.anatomy_weights,
            stage_fracs=self.curriculum_stage_fracs,
            alpha_init=self.alp_alpha_init,
            alpha_final=self.alp_alpha_final,
            guidance_threshold_init=self.alp_guidance_threshold_init,
            guidance_threshold_final=self.alp_guidance_threshold_final,
            n_frames_by_stage=self.alp_n_frames,
            alp_reader=None if isinstance(self.alp_reader, NullALPReader) else self.alp_reader,
            hardness_temperature=float(
                getattr(self, "_hardness_temperature", 1.0)
            ),
        )
        self._image_sampler = CombinedSampler(
            self._image_entries,
            samples_per_epoch=self.image_samples,
            **sampler_kw,
        )
        self._video_sampler = CombinedSampler(
            self._video_entries,
            samples_per_epoch=self.video_samples,
            **sampler_kw,
        )
        self._paired_sampler = CombinedSampler(
            self._paired_entries,
            samples_per_epoch=self.video_samples,
            **sampler_kw,
        )
        self._setup_done = True

    # ── Curriculum interface ──────────────────────────────────────────────────

    def update_step(self, global_step: int):
        if self._image_sampler: self._image_sampler.update_step(global_step)
        if self._video_sampler: self._video_sampler.update_step(global_step)
        if self._paired_sampler: self._paired_sampler.update_step(global_step)
        # Push current alpha and mask_ratio into datasets
        if self._image_dataset:
            self._image_dataset.alpha = self.current_alpha()
            self._image_dataset.mask_ratio = self.current_mask_ratio()
            self._image_dataset.mask_guidance_threshold = self.current_mask_guidance_threshold()
        if self._video_dataset:
            self._video_dataset.mask_ratio = self.current_mask_ratio()
            self._video_dataset.transform.cfg.n_frames = self.current_n_frames()
        if self._paired_dataset:
            self._paired_dataset.img_alpha = self.current_alpha()
            self._paired_dataset.img_mask_ratio = self.current_mask_ratio()
            self._paired_dataset.img_mask_guidance_threshold = self.current_mask_guidance_threshold()
            self._paired_dataset.vid_mask_ratio = self.current_mask_ratio()
            self._paired_dataset.vid_transform.cfg.n_frames = self.current_n_frames()

    def current_alpha(self) -> float:
        return self._image_sampler.current_alpha() if self._image_sampler else 1.0

    def current_mask_ratio(self) -> float:
        return self._video_sampler.current_mask_ratio() if self._video_sampler else 0.75

    def current_mask_guidance_threshold(self) -> float:
        return (
            self._image_sampler.current_mask_guidance_threshold()
            if self._image_sampler else 0.5
        )

    def current_n_frames(self) -> int:
        return self._video_sampler.current_n_frames() if self._video_sampler else 16

    def current_stage(self) -> int:
        if self._image_sampler:
            return self._image_sampler.current_stage()
        return 1

    def curriculum_snapshot(self) -> dict:
        """Scalar curriculum state for logging / monitoring."""
        cs = self._image_sampler.curriculum if self._image_sampler else None
        stage = self.current_stage()
        tier_counts = {}
        tier_in_pool = {}
        if cs is not None:
            for tier in sorted(cs.tier_idx.keys()):
                tier_counts[tier] = len(cs.tier_idx[tier])
                tier_in_pool[tier] = 0
            for idx in cs._pool:
                tier = cs.entries[idx].curriculum_tier
                tier_in_pool[tier] = tier_in_pool.get(tier, 0) + 1
        return {
            "alp_stage": stage,
            "alp_alpha": self.current_alpha(),
            "mask_guidance_threshold": self.current_mask_guidance_threshold(),
            "mask_ratio": self.current_mask_ratio(),
            "hardness_weight": self._image_sampler.current_hardness_weight()
            if self._image_sampler else 0.0,
            "n_frames": self.current_n_frames(),
            "tier_pool_size": len(cs._pool) if cs else 0,
            "tier1_in_pool": tier_in_pool.get(1, 0),
            "tier2_in_pool": tier_in_pool.get(2, 0),
            "tier3_in_pool": tier_in_pool.get(3, 0),
            "tier1_total": tier_counts.get(1, 0),
            "tier2_total": tier_counts.get(2, 0),
            "tier3_total": tier_counts.get(3, 0),
        }

    # ── Loaders ───────────────────────────────────────────────────────────────

    def image_loader(self) -> DataLoader:
        """Phase 1 + Phase 3 image stream."""
        self.setup()
        return DataLoader(
            self._image_dataset,
            sampler=self._image_sampler,
            batch_size=self.image_batch_size,
            num_workers=self.num_workers,
            collate_fn=ImageSSLCollator(),
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def video_loader(self) -> DataLoader:
        """Phase 2 + Phase 3 video stream."""
        self.setup()
        return DataLoader(
            self._video_dataset,
            sampler=self._video_sampler,
            batch_size=self.video_batch_size,
            num_workers=self.num_workers,
            collate_fn=VideoSSLCollator(),
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def paired_loader(self) -> DataLoader:
        """
        Phase 3 paired stream.

        Each batch is an AlignedDualStreamBatch where image_batch[i] and
        video_batch[i] are drawn from the *same* source clip entry
        (ssl_stream='both'), so alignment_pairs contains exactly
        ``video_batch_size`` pairs, all with weight=1.0.

        This replaces the zip(image_loader, video_loader) approach in Phase 3,
        guaranteeing n_align_pairs == batch_size every step.
        """
        from data.pipeline.collators import PairedSSLCollator
        self.setup()
        return DataLoader(
            self._paired_dataset,
            sampler=self._paired_sampler,
            batch_size=self.video_batch_size,
            num_workers=self.num_workers,
            collate_fn=PairedSSLCollator(),
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def combined_loader(self):
        """
        Phase 3: yields AlignedDualStreamBatch objects by zipping both loaders.
        Alignment pairs link image/video samples sharing the same study_id and
        overlapping frame ranges (see collators.build_alignment_pairs).
        The shorter loader determines epoch length.
        """
        from data.pipeline.collators import make_aligned_dual_stream
        self.setup()
        img_loader = self.image_loader()
        vid_loader = self.video_loader()
        for img_batch, vid_batch in zip(img_loader, vid_loader):
            yield make_aligned_dual_stream(img_batch, vid_batch)

    def fine_tune_loader(
        self,
        anatomy_families: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        supervised_only: bool = True,
    ) -> DataLoader:
        """
        Phase 4: supervised fine-tuning loader.
        Returns samples with ground-truth masks/labels.
        Optionally filter to specific anatomy families.
        """
        self.setup()
        entries = load_manifest(
            self.manifest_path,
            split="train",
            anatomy_families=anatomy_families,
        )
        if supervised_only:
            entries = [e for e in entries if e.task_type != "ssl_only"]

        from dataset import ImageSSLDataset as _DS
        dataset = _DS(entries, cfg=self.image_cfg, root_remap=self.root_remap)
        return DataLoader(
            dataset,
            batch_size=batch_size or self.image_batch_size,
            num_workers=self.num_workers,
            collate_fn=ImageSSLCollator(),
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_loader(self, stream: str = "image") -> DataLoader:
        """Validation loader (no augmentation / fixed curriculum)."""
        self.setup()
        entries = load_manifest(self.manifest_path, split="val",
                                ssl_stream=stream if stream != "both" else None)
        if stream in ("image", "both"):
            ds = ImageSSLDataset(entries, cfg=self.image_cfg, root_remap=self.root_remap)
            collator = ImageSSLCollator()
            bs = self.image_batch_size
        else:
            ds = VideoSSLDataset(entries, cfg=self.video_cfg,
                                  patch_size=self.patch_size, root_remap=self.root_remap)
            collator = VideoSSLCollator()
            bs = self.video_batch_size
        return DataLoader(ds, batch_size=bs, num_workers=self.num_workers,
                          collate_fn=collator, pin_memory=self.pin_memory,
                          shuffle=False)
