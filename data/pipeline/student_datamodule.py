"""
data/pipeline/student_datamodule.py  ·  Mixed batch data module for student training
====================================================================================

StudentDataModule
-----------------
Manages three data streams with per-stage mix ratios:
  image  : standalone images + frames sampled from video (T=1)
  video  : video clips (T>1)
  paired : one video frame + its parent clip (T=1 and T>1 together)

Per-stage mix ratios are loaded from YAML config:

  stage1:  {image_frac: 0.90, video_frac: 0.10, paired_frac: 0.00}
  stage2:  {image_frac: 0.40, video_frac: 0.60, paired_frac: 0.00}
  stage3:  {image_frac: 0.50, video_frac: 0.30, paired_frac: 0.20}
  stage4:  {image_frac: 0.50, video_frac: 0.30, paired_frac: 0.20}  # EMA divergence

Each batch dict is tagged with batch["sample_type"] ∈ {"image","video","paired"}.
Loss functions in student_phase_steps.py route on this key.

StudentMixedCollator
--------------------
Thin wrapper around the existing ImageSSLCollator / VideoSSLCollator.
Adds the sample_type tag.  All output tensors have the same shape contract
as the existing collators (native-resolution, zero-padded, with padding_mask)
except that image collation uses stride-4 patch granularity for the Hiera
student (vs stride-16 for the existing dual-branch model).

Frame-from-video sampling
-------------------------
sample_frame_as_image(entry) selects a random frame from a video manifest
entry and returns a synthetic image entry with ssl_stream="image".
Used to augment the image stream with diverse video frames.

No changes to existing USFoundationDataModule, adapters, or manifest schema.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.pipeline.samplers import CombinedSampler, DistributedCombinedSampler

log = logging.getLogger(__name__)

# Legacy dual-branch collators emit masks at ViT patch stride 16.
SOURCE_MASK_PATCH_STRIDE = 16

_SAMPLE_TYPE_TO_IDX = {"image": 0, "video": 1, "paired": 2}
_IDX_TO_SAMPLE_TYPE = ("image", "video", "paired")


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _ddp_barrier() -> None:
    if _ddp_active():
        dist.barrier()


def _ddp_broadcast_sample_type(sample_type: str) -> str:
    """
    Broadcast the batch-type decision from rank 0 to all DDP ranks.

    Without this, each rank independently samples image/video/paired and runs
    different forward paths (e.g. stage-2 image vs video), causing NCCL
    ALLREDUCE deadlocks on mismatched parameter usage.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return sample_type
    t = torch.zeros(1, dtype=torch.long, device=torch.cuda.current_device())
    if dist.get_rank() == 0:
        t[0] = _SAMPLE_TYPE_TO_IDX[sample_type]
    dist.broadcast(t, src=0)
    return _IDX_TO_SAMPLE_TYPE[int(t.item())]


# ---------------------------------------------------------------------------
# Per-stage mix ratio config
# ---------------------------------------------------------------------------

@dataclass
class StageMixConfig:
    image_frac:  float = 0.90
    video_frac:  float = 0.10
    paired_frac: float = 0.00

    def __post_init__(self):
        total = self.image_frac + self.video_frac + self.paired_frac
        assert abs(total - 1.0) < 1e-4, (
            f"Stage mix fractions must sum to 1.0, got {total:.3f}"
        )

    def sample_type(self) -> str:
        """Sample a batch type according to fractions."""
        r = random.random()
        if r < self.image_frac:
            return "image"
        elif r < self.image_frac + self.video_frac:
            return "video"
        else:
            return "paired"


@dataclass
class StudentDataConfig:
    """
    Configuration for StudentDataModule.

    Fields
    ------
    stage_mixes       : per-stage mix ratios (list of 4, indexed 0=stage1 .. 3=stage4)
    image_batch_size  : batch size for image samples
    video_batch_size  : batch size for video samples
    paired_batch_size : batch size for paired samples
    num_workers       : dataloader worker count
    patch_size        : Hiera stride for padding mask (4)
    video_sample_as_image_frac : fraction of image batches drawn from video frames
    """
    stage_mixes: List[StageMixConfig] = field(
        default_factory=lambda: [
            StageMixConfig(image_frac=0.90, video_frac=0.10, paired_frac=0.00),  # stage 1
            StageMixConfig(image_frac=0.40, video_frac=0.60, paired_frac=0.00),  # stage 2
            StageMixConfig(image_frac=0.50, video_frac=0.30, paired_frac=0.20),  # stage 3
            StageMixConfig(image_frac=0.50, video_frac=0.30, paired_frac=0.20),  # stage 4
        ]
    )
    image_batch_size:  int   = 64
    video_batch_size:  int   = 16
    paired_batch_size: int   = 8
    num_workers:       int   = 8
    patch_size:        int   = 4    # Hiera effective stride
    video_sample_as_image_frac: float = 0.30  # 30% of image batches from video frames

    @classmethod
    def from_dict(cls, d: dict) -> "StudentDataConfig":
        cfg = cls()
        if "stage_mixes" in d:
            cfg.stage_mixes = [StageMixConfig(**m) for m in d["stage_mixes"]]
        for k in ["image_batch_size", "video_batch_size", "paired_batch_size",
                  "num_workers", "patch_size", "video_sample_as_image_frac"]:
            if k in d:
                setattr(cfg, k, d[k])
        return cfg


# ---------------------------------------------------------------------------
# Frame-from-video sampling utility
# ---------------------------------------------------------------------------

def sample_frame_as_image(video_entry: dict) -> dict:
    """
    Given a manifest entry with ssl_stream="video", produce a synthetic
    image entry by selecting one random frame.

    Parameters
    ----------
    video_entry : USManifestEntry-like dict with "frames" or "video_path" key

    Returns
    -------
    dict with ssl_stream="image", T=1 frame selected at random
    """
    entry = dict(video_entry)
    entry["ssl_stream"] = "image"

    # Record sampled frame index for the dataset to use
    frames_key = None
    for k in ("frames", "frame_paths", "n_frames", "num_frames"):
        if k in entry:
            frames_key = k
            break

    n_frames = 1
    if frames_key == "n_frames" or frames_key == "num_frames":
        n_frames = entry[frames_key]
    elif frames_key in ("frames", "frame_paths"):
        frames_list = entry[frames_key]
        if isinstance(frames_list, (list, tuple)):
            n_frames = len(frames_list)

    entry["sampled_frame_idx"] = random.randint(0, max(0, n_frames - 1))
    return entry


# ---------------------------------------------------------------------------
# Mask grid utilities (stride-16 teachers → stride-4 student)
# ---------------------------------------------------------------------------

def upsample_mask_grid(mask: torch.Tensor, factor: int) -> torch.Tensor:
    """Nearest-neighbour upsample a bool mask by `factor` along spatial axes."""
    if factor <= 1:
        return mask
    if mask.dim() == 4:
        b, n, ph, pw = mask.shape
        m = mask.reshape(b * n, 1, ph, pw).float()
        m = F.interpolate(m, scale_factor=factor, mode="nearest")
        ph2, pw2 = m.shape[-2:]
        return m.reshape(b, n, ph2, pw2).bool()
    if mask.dim() == 3:
        m = F.interpolate(mask.float().unsqueeze(1), scale_factor=factor, mode="nearest")
        return m.squeeze(1).bool()
    raise ValueError(f"upsample_mask_grid: unexpected dim {mask.dim()}")


def downsample_mask_grid(mask: torch.Tensor, factor: int) -> torch.Tensor:
    """Max-pool a bool mask by `factor` (valid if any sub-cell is valid)."""
    if factor <= 1:
        return mask
    if mask.dim() == 4:
        b, n, ph, pw = mask.shape
        m = mask.reshape(b * n, 1, ph, pw).float()
        m = F.max_pool2d(m, kernel_size=factor, stride=factor)
        ph2, pw2 = m.shape[-2:]
        return m.reshape(b, n, ph2, pw2).bool()
    if mask.dim() == 3:
        m = F.max_pool2d(mask.float().unsqueeze(1), kernel_size=factor, stride=factor)
        return m.squeeze(1).bool()
    raise ValueError(f"downsample_mask_grid: unexpected dim {mask.dim()}")


def resample_batch_masks_for_student(
    batch: dict,
    *,
    student_patch_stride: int = 4,
    source_patch_stride: int = SOURCE_MASK_PATCH_STRIDE,
) -> dict:
    """
    Upsample collator masks to the student (Hiera) input grid.

    Preserves stride-16 copies as ``*_s16`` for frozen DINO / V-JEPA teachers.
    """
    factor = source_patch_stride // student_patch_stride
    if factor <= 1:
        return batch

    batch["student_mask_stride"] = student_patch_stride
    batch["teacher_mask_stride"] = source_patch_stride

    for key in ("global_pmasks", "patch_masks", "padding_masks"):
        if key not in batch or batch[key] is None:
            continue
        batch[f"{key}_s16"] = batch[key]
        batch[key] = upsample_mask_grid(batch[key], factor)

    if batch.get("tube_masks") is not None and batch.get("tube_masks_s16") is None:
        batch["tube_masks_s16"] = batch["tube_masks"]
    return batch


# ---------------------------------------------------------------------------
# StudentMixedCollator
# ---------------------------------------------------------------------------

class StudentMixedCollator:
    """
    Wraps the existing ImageSSLCollator or VideoSSLCollator and adds
    batch["sample_type"] ∈ {"image", "video", "paired"}.

    Also converts padding masks from the existing stride-16 granularity
    to stride-4 granularity needed by the Hiera student when patch_size=4.

    Parameters
    ----------
    image_collator  : ImageSSLCollator instance
    video_collator  : VideoSSLCollator instance
    paired_collator : PairedSSLCollator instance (or None)
    patch_size      : target padding-mask patch stride (4 for Hiera)
    """

    def __init__(
        self,
        image_collator,
        video_collator,
        paired_collator=None,
        patch_size: int = 4,
    ):
        self.image_collator  = image_collator
        self.video_collator  = video_collator
        self.paired_collator = paired_collator
        self.patch_size      = patch_size
        self._mask_factor    = SOURCE_MASK_PATCH_STRIDE // patch_size

    def _finalize_batch(self, batch: dict, sample_type: str) -> dict:
        batch["sample_type"] = sample_type
        if self._mask_factor > 1:
            resample_batch_masks_for_student(
                batch,
                student_patch_stride=self.patch_size,
                source_patch_stride=SOURCE_MASK_PATCH_STRIDE,
            )
        return batch

    def collate_image(self, samples: list) -> dict:
        return self._finalize_batch(self.image_collator(samples), "image")

    def collate_video(self, samples: list) -> dict:
        return self._finalize_batch(self.video_collator(samples), "video")

    def _flatten_paired_batch(self, aligned) -> dict:
        """Merge PairedSSLCollator output into a flat dict for student_stage3_step."""
        img = aligned.image_batch
        vid = aligned.video_batch
        batch = dict(vid)
        batch["frame"] = img["global_crops"][:, 0]
        if img["global_crops"].shape[1] >= 2:
            batch["clean_frame"] = img["global_crops"][:, 1]
        if img.get("global_pmasks") is not None:
            batch["global_pmasks"] = img["global_pmasks"]
            if img["global_pmasks"].shape[1] >= 2:
                batch["clean_frame_pmask"] = img["global_pmasks"][:, 1]
        if img.get("patch_masks") is not None:
            batch["patch_masks"] = img["patch_masks"]
        batch["alignment_pairs"] = aligned.alignment_pairs
        return batch

    def collate_paired(self, samples: list) -> dict:
        if self.paired_collator is not None:
            batch = self._flatten_paired_batch(self.paired_collator(samples))
        else:
            batch = self.video_collator(samples)
        batch = self._finalize_batch(batch, "paired")
        if batch.get("frame_pmask") is None and batch.get("global_pmasks") is not None:
            batch["frame_pmask"] = batch["global_pmasks"][:, 0]
        return batch


# ---------------------------------------------------------------------------
# StudentDataModule
# ---------------------------------------------------------------------------

class StudentDataModule:
    """
    Data module for the single-student pretraining curriculum.

    Wraps the existing USFoundationDataModule manifest infrastructure and
    adds per-stage batch-type routing.

    Usage
    -----
    dm = StudentDataModule(base_datamodule, cfg)
    dm.set_stage(1)      # switch mix ratios for stage 1, 2, 3, or 4
    batch = dm.next_batch()   # returns mixed batch with sample_type tag

    Parameters
    ----------
    base_dm   : existing USFoundationDataModule (provides datasets + manifest)
    cfg       : StudentDataConfig
    collators : StudentMixedCollator instance
    """

    def __init__(
        self,
        base_dm,
        cfg: StudentDataConfig,
        collators: StudentMixedCollator,
    ):
        self.base_dm   = base_dm
        self.cfg       = cfg
        self.collators = collators
        self._stage    = 0   # 0-indexed (stage1 = 0)
        self._curriculum_stage = 0
        self._image_sampler = None
        self._video_sampler = None
        self._paired_sampler = None

        # Build DataLoaders lazily
        self._image_loader:  Optional[DataLoader] = None
        self._video_loader:  Optional[DataLoader] = None
        self._paired_loader: Optional[DataLoader] = None
        self._image_iter  = None
        self._video_iter  = None
        self._paired_iter = None

    def set_stage(self, stage: int) -> None:
        """
        Set the current curriculum stage (1-indexed).

        Parameters
        ----------
        stage : 1, 2, 3, or 4
        """
        assert 1 <= stage <= 4, f"stage must be 1-4, got {stage}"
        self._stage = stage - 1
        log.info(
            f"StudentDataModule: stage={stage}, "
            f"mix={self.cfg.stage_mixes[self._stage]}"
        )

    @property
    def current_mix(self) -> StageMixConfig:
        return self.cfg.stage_mixes[self._stage]

    def _reset_iters(self) -> None:
        """Drop cached loader iterators (e.g. after ALP curriculum stage change)."""
        self._image_iter = None
        self._video_iter = None
        self._paired_iter = None
        self._release_loader("_image_loader")
        self._release_loader("_video_loader")
        self._release_loader("_paired_loader")

    def update_curriculum(self, global_step: int) -> dict:
        """
        Advance OPENUS/ALP curriculum on the base datamodule and refresh datasets.

        Returns a snapshot dict for logging (tier pool, alpha, mask ratio, …).
        """
        self.base_dm.setup()
        old_stage = self.base_dm.current_stage()
        self.base_dm.update_step(global_step)
        new_stage = self.base_dm.current_stage()
        if new_stage != old_stage:
            log.info(
                "ALP curriculum stage %d → %d at step %d (pool=%d)",
                old_stage, new_stage, global_step,
                self.base_dm.curriculum_snapshot().get("tier_pool_size", 0),
            )
            self._reset_iters()
        self._curriculum_stage = new_stage
        return self.base_dm.curriculum_snapshot()

    def _wrap_sampler(self, combined: CombinedSampler):
        if _ddp_active():
            return DistributedCombinedSampler(combined)
        return combined

    def _get_stream_sampler(self, stream: str) -> CombinedSampler:
        self.base_dm.setup()
        if stream == "image":
            return self.base_dm._image_sampler
        if stream == "video":
            return self.base_dm._video_sampler
        return self.base_dm._paired_sampler

    def _build_loader(
        self,
        ds: Dataset,
        batch_size: int,
        collate_fn,
        stream: str,
    ) -> DataLoader:
        """Build a DataLoader with curriculum + optional DDP sharding."""
        nw = self.cfg.num_workers
        combined = self._get_stream_sampler(stream)
        sampler = self._wrap_sampler(combined)
        if stream == "image":
            self._image_sampler = sampler
        elif stream == "video":
            self._video_sampler = sampler
        else:
            self._paired_sampler = sampler

        kwargs = dict(
            batch_size=batch_size,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
            persistent_workers=nw > 0,
            sampler=sampler,
        )
        if nw > 0:
            # DDP: keep prefetch low — stage 3 may hold 3 loaders × workers × prefetch.
            kwargs["prefetch_factor"] = 1 if _ddp_active() else 4
        return DataLoader(ds, **kwargs)

    def _prefetch_batches(self, loader: DataLoader, n_batches: int) -> None:
        if n_batches <= 0:
            return
        it = iter(loader)
        for _ in range(n_batches):
            try:
                next(it)
            except StopIteration:
                break

    def _release_loader(self, loader_attr: str) -> None:
        """Shut down DataLoader workers to avoid stacking them during warmup."""
        loader = getattr(self, loader_attr, None)
        if loader is not None:
            del loader
            setattr(self, loader_attr, None)

    def warmup_loaders(self, n_batches: int = 2, *, epoch: int = 0) -> None:
        """Spawn workers and prefetch a few batches before the training loop."""
        if n_batches <= 0:
            return
        if _ddp_active():
            self.set_epoch(epoch)

        def _log_warmup_done(stream: str, t_start: float) -> None:
            if not _ddp_active() or dist.get_rank() == 0:
                log.info("Warmup %s stream done (%.1fs)", stream, time.time() - t_start)

        t0 = time.time()
        self._prefetch_batches(self._get_image_loader(), n_batches)
        _ddp_barrier()
        _log_warmup_done("image", t0)
        self._release_loader("_image_loader")

        t1 = time.time()
        self._prefetch_batches(self._get_video_loader(), n_batches)
        _ddp_barrier()
        _log_warmup_done("video", t1)

        if self.current_mix.paired_frac > 0:
            self._release_loader("_video_loader")
            t2 = time.time()
            self._prefetch_batches(self._get_paired_loader(), n_batches)
            _ddp_barrier()
            _log_warmup_done("paired", t2)

    def set_epoch(self, epoch: int) -> None:
        """Advance curriculum samplers in lockstep across DDP ranks."""
        for sampler in (self._image_sampler, self._video_sampler, self._paired_sampler):
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        if not _ddp_active():
            return
        for loader_attr in ("_image_loader", "_video_loader", "_paired_loader"):
            loader = getattr(self, loader_attr, None)
            sampler = getattr(loader, "sampler", None) if loader is not None else None
            if isinstance(sampler, DistributedCombinedSampler):
                sampler.set_epoch(epoch)

    def _get_image_loader(self) -> DataLoader:
        if self._image_loader is None:
            self._image_loader = self._build_loader(
                self._get_image_dataset(),
                self.cfg.image_batch_size,
                self.collators.collate_image,
                stream="image",
            )
        return self._image_loader

    def _get_video_loader(self) -> DataLoader:
        if self._video_loader is None:
            self._video_loader = self._build_loader(
                self._get_video_dataset(),
                self.cfg.video_batch_size,
                self.collators.collate_video,
                stream="video",
            )
        return self._video_loader

    def _get_paired_loader(self) -> DataLoader:
        if self._paired_loader is None:
            self._paired_loader = self._build_loader(
                self._get_paired_dataset(),
                self.cfg.paired_batch_size,
                self.collators.collate_paired,
                stream="paired",
            )
        return self._paired_loader

    def _get_image_dataset(self) -> Dataset:
        """
        Returns the image SSL dataset from the base datamodule.
        USFoundationDataModule stores these as _image_dataset after setup().
        """
        self.base_dm.setup()
        for attr in ("_image_dataset", "image_dataset", "img_dataset", "train_image_dataset"):
            ds = getattr(self.base_dm, attr, None)
            if ds is not None:
                return ds
        raise RuntimeError(
            f"{type(self.base_dm).__name__} has no image dataset "
            f"(checked _image_dataset, image_dataset, img_dataset)"
        )

    def _get_video_dataset(self) -> Dataset:
        self.base_dm.setup()
        for attr in ("_video_dataset", "video_dataset", "vid_dataset", "train_video_dataset"):
            ds = getattr(self.base_dm, attr, None)
            if ds is not None:
                return ds
        raise RuntimeError(
            f"{type(self.base_dm).__name__} has no video dataset "
            f"(checked _video_dataset, video_dataset, vid_dataset)"
        )

    def _get_paired_dataset(self) -> Dataset:
        self.base_dm.setup()
        for attr in ("_paired_dataset", "paired_dataset", "train_paired_dataset"):
            ds = getattr(self.base_dm, attr, None)
            if ds is not None:
                return ds
        # Fallback to video dataset; collator synthesizes paired structure
        return self._get_video_dataset()

    def _next_from(self, loader: DataLoader, it_attr: str) -> dict:
        """Cycle through a dataloader indefinitely."""
        it = getattr(self, it_attr)
        if it is None:
            it = iter(loader)
            setattr(self, it_attr, it)
        try:
            batch = next(it)
        except StopIteration:
            # Epoch rollover is instant (no I/O); sync so all ranks restart together.
            _ddp_barrier()
            it = iter(loader)
            setattr(self, it_attr, it)
            batch = next(it)
        return batch

    def next_batch(self) -> dict:
        """
        Sample the next batch according to the current stage mix ratios.

        Returns a batch dict with batch["sample_type"] set.
        """
        if _ddp_active():
            sample_type = (
                self.current_mix.sample_type()
                if dist.get_rank() == 0
                else "image"
            )
            sample_type = _ddp_broadcast_sample_type(sample_type)
        else:
            sample_type = self.current_mix.sample_type()

        if sample_type == "image":
            batch = self._next_from(self._get_image_loader(), "_image_iter")
        elif sample_type == "video":
            batch = self._next_from(self._get_video_loader(), "_video_iter")
        elif self.current_mix.paired_frac <= 0.0:
            # Fallback: return video if no paired data configured
            batch = self._next_from(self._get_video_loader(), "_video_iter")
        else:
            batch = self._next_from(self._get_paired_loader(), "_paired_iter")

        # Slow I/O on one rank must not let others enter NCCL collectives in train_step.
        _ddp_barrier()
        return batch
