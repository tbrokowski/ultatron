"""
samplers.py  ·  Stratified and curriculum-aware samplers
=========================================================

Three samplers:
  AnatomyStratifiedSampler  – balances samples across anatomy families to
      prevent large cardiac / breast datasets from dominating pretraining.
  CurriculumSampler         – progressively reveals harder samples as
      training advances (Stage 1→2→3 as in OPENUS).
  CombinedSampler           – chains both (stratified within each tier).

Usage in DataLoader:
    sampler = CombinedSampler(entries, global_step=step, total_steps=total)
    loader  = DataLoader(dataset, batch_sampler=BatchSamplerWrapper(sampler, bs))
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from data.schema.manifest import USManifestEntry
from data.pipeline.alp_interface import ALPReader, NullALPReader

class AnatomyStratifiedSampler(Sampler):
    """
    At each epoch, draw `samples_per_epoch` indices such that each
    anatomy family is represented roughly equally (capped at its actual size).

    Rationale: MIMIC-IV-ECHO alone has 525K studies; without stratification
    the model would be ~50% cardiac during pretraining.

    Args:
        entries:            full list of manifest entries
        samples_per_epoch:  total samples drawn per epoch
        min_per_family:     floor to include rare anatomies every epoch
        max_per_family:     cap to limit over-represented families
        weights:            optional per-anatomy family override weights
    """
    def __init__(
        self,
        entries: List[USManifestEntry],
        samples_per_epoch: int = 500_000,
        min_per_family: int = 100,
        max_per_family: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.entries = entries
        self.samples_per_epoch = samples_per_epoch
        self.min_per_family = min_per_family
        self.max_per_family = max_per_family
        self.weights = weights or {}

        # Build family → list[idx]
        self.family_idx: Dict[str, List[int]] = defaultdict(list)
        for i, e in enumerate(entries):
            self.family_idx[e.anatomy_family].append(i)

        self.families = sorted(self.family_idx.keys())
        n_families = len(self.families)
        # Base quota: equal share
        base = samples_per_epoch // n_families
        self.quota: Dict[str, int] = {}
        for fam in self.families:
            w = self.weights.get(fam, 1.0)
            q = max(min_per_family, int(base * w))
            if max_per_family:
                q = min(q, max_per_family)
            q = min(q, len(self.family_idx[fam]))
            self.quota[fam] = q

    def __len__(self) -> int:
        return sum(self.quota.values())

    def __iter__(self) -> Iterator[int]:
        indices = []
        for fam, q in self.quota.items():
            pool = self.family_idx[fam]
            if q >= len(pool):
                indices.extend(pool)
            else:
                indices.extend(random.sample(pool, q))
        random.shuffle(indices)
        return iter(indices)


# ── Curriculum sampler ────────────────────────────────────────────────────────

class CurriculumSampler(Sampler):
    """
    OPENUS-inspired 3-stage progressive curriculum.

    Stage 1 (0–33% of training):   tier-1 only        (short, masked, known anatomy)
    Stage 2 (33–66% of training):  tier-1 + tier-2
    Stage 3 (66–100% of training): all tiers

    Within each stage, sampling is uniform.
    The current stage is determined by `global_step / total_steps`.
    Call `update_step(step)` from the training loop each epoch/step.
    """
    def __init__(
        self,
        entries: List[USManifestEntry],
        global_step: int = 0,
        total_steps: int = 100_000,
        samples_per_epoch: int = 200_000,
        stage_fracs: Optional[List[float]] = None,
        alpha_init: float = 0.1,
        alpha_final: float = 0.9,
        guidance_threshold_init: float = 0.1,
        guidance_threshold_final: float = 0.9,
        n_frames_by_stage: Optional[List[int]] = None,
    ):
        self.entries = entries
        self.total_steps = total_steps
        self.samples_per_epoch = samples_per_epoch
        # Fractions of total_steps at which ALP tier pool expands (stage 1→2, 2→3).
        fracs = stage_fracs or [0.33, 0.66]
        if len(fracs) != 2:
            raise ValueError(f"stage_fracs must have length 2, got {len(fracs)}")
        self.stage_fracs = fracs
        # OpenUS-style cosine ramps (see finetune/backbones/vendor/openus/main_openus.py).
        self.alpha_init = alpha_init
        self.alpha_final = alpha_final
        self.guidance_threshold_init = guidance_threshold_init
        self.guidance_threshold_final = guidance_threshold_final
        nf = n_frames_by_stage or [8, 16, 32]
        if len(nf) != 3:
            raise ValueError(f"n_frames_by_stage must have length 3, got {len(nf)}")
        self.n_frames_by_stage = nf

        self.tier_idx: Dict[int, List[int]] = defaultdict(list)
        for i, e in enumerate(entries):
            self.tier_idx[e.curriculum_tier].append(i)

        self.current_step = global_step
        self._pool: List[int] = []
        self._rebuild_pool()

    def update_step(self, step: int):
        old_stage = self._stage(self.current_step)
        self.current_step = step
        if self._stage(step) != old_stage:
            self._rebuild_pool()

    def _stage(self, step: int) -> int:
        frac = step / max(self.total_steps, 1)
        if frac < self.stage_fracs[0]:
            return 1
        if frac < self.stage_fracs[1]:
            return 2
        return 3

    @staticmethod
    def _cosine_ramp(start: float, end: float, step: int, total: int) -> float:
        """Cosine interpolation from *start* → *end* over [0, total-1]."""
        if total <= 1:
            return end
        t = step / max(total - 1, 1)
        return start + (end - start) * (1.0 - math.cos(math.pi * t)) / 2.0

    def current_alpha(self) -> float:
        """
        OpenUS blend weight for teacher saliency in
        ``masking_score = alpha * S_k + (1-alpha) * H_k``.

        Cosine ramp from ``alpha_init`` (student-hardness heavy) to
        ``alpha_final`` (teacher-saliency heavy) over training.
        """
        return self._cosine_ramp(
            self.alpha_init, self.alpha_final,
            self.current_step, self.total_steps,
        )

    def current_mask_guidance_threshold(self) -> float:
        """
        OpenUS ``m_t``: fraction of mask budget drawn from highest-ALP patches.

        Ramps 0.1 → 0.9 so early training mixes random masks with feedback,
        then increasingly targets hard/salient regions.
        """
        return self._cosine_ramp(
            self.guidance_threshold_init, self.guidance_threshold_final,
            self.current_step, self.total_steps,
        )

    def current_mask_ratio(self) -> float:
        """Mask ratio increases with difficulty stage."""
        return {1: 0.40, 2: 0.65, 3: 0.80}[self._stage(self.current_step)]

    def current_n_frames(self) -> int:
        """Clip length increases with ALP curriculum stage."""
        return self.n_frames_by_stage[self._stage(self.current_step) - 1]

    def current_hardness_weight(self) -> float:
        """
        Fraction of hardness-aware (vs uniform) sampling within the tier pool.
        Ramps up as curriculum progresses so the student sees harder examples.
        """
        return {1: 0.0, 2: 0.3, 3: 0.6}[self._stage(self.current_step)]

    def _rebuild_pool(self):
        stage = self._stage(self.current_step)
        pool = []
        for tier in range(1, stage + 1):
            pool.extend(self.tier_idx.get(tier, []))
        self._pool = pool

    def __len__(self) -> int:
        return min(self.samples_per_epoch, len(self._pool))

    def __iter__(self) -> Iterator[int]:
        if len(self._pool) <= self.samples_per_epoch:
            indices = list(self._pool)
        else:
            indices = random.sample(self._pool, self.samples_per_epoch)
        random.shuffle(indices)
        return iter(indices)


# ── Combined sampler ──────────────────────────────────────────────────────────

class CombinedSampler(Sampler):
    """
    Anatomy-stratified + curriculum combined.
    Applies stratification within the tier-filtered pool.
    """
    def __init__(
        self,
        entries: List[USManifestEntry],
        global_step: int = 0,
        total_steps: int = 100_000,
        samples_per_epoch: int = 200_000,
        anatomy_weights: Optional[Dict[str, float]] = None,
        stage_fracs: Optional[List[float]] = None,
        alpha_init: float = 0.1,
        alpha_final: float = 0.9,
        guidance_threshold_init: float = 0.1,
        guidance_threshold_final: float = 0.9,
        n_frames_by_stage: Optional[List[int]] = None,
        alp_reader: Optional[ALPReader] = None,
        hardness_temperature: float = 1.0,
    ):
        self.curriculum = CurriculumSampler(
            entries, global_step, total_steps, samples_per_epoch * 4,
            stage_fracs=stage_fracs,
            alpha_init=alpha_init,
            alpha_final=alpha_final,
            guidance_threshold_init=guidance_threshold_init,
            guidance_threshold_final=guidance_threshold_final,
            n_frames_by_stage=n_frames_by_stage,
        )
        self.samples_per_epoch = samples_per_epoch
        self.entries = entries
        self.anatomy_weights = anatomy_weights or {}
        self.alp_reader = alp_reader
        self.hardness_temperature = hardness_temperature
        self._current_pool_entries: Optional[List] = None

    def update_step(self, step: int):
        self.curriculum.update_step(step)

    def current_alpha(self) -> float:
        return self.curriculum.current_alpha()

    def current_mask_ratio(self) -> float:
        return self.curriculum.current_mask_ratio()

    def current_n_frames(self) -> int:
        return self.curriculum.current_n_frames()

    def current_stage(self) -> int:
        return self.curriculum._stage(self.curriculum.current_step)

    def current_hardness_weight(self) -> float:
        return self.curriculum.current_hardness_weight()

    def current_mask_guidance_threshold(self) -> float:
        return self.curriculum.current_mask_guidance_threshold()

    def _apply_hardness_reweight(
        self,
        indices: List[int],
        rng: random.Random,
    ) -> List[int]:
        """Resample indices toward high-hardness samples (OPENUS-style)."""
        hw = self.current_hardness_weight()
        if (
            hw <= 0.0
            or self.alp_reader is None
            or isinstance(self.alp_reader, NullALPReader)
            or len(indices) <= 1
        ):
            return indices

        hardness = np.array([
            self.alp_reader.aggregate_hardness(self.entries[i].sample_id)
            for i in indices
        ], dtype=np.float64)
        temp = max(self.hardness_temperature, 1e-6)
        logits = hardness / temp
        logits -= logits.max()
        w_hard = np.exp(logits)
        w_hard /= w_hard.sum() + 1e-8

        w_uniform = np.ones(len(indices), dtype=np.float64) / len(indices)
        weights = (1.0 - hw) * w_uniform + hw * w_hard
        weights /= weights.sum()

        n = min(self.samples_per_epoch, len(indices))
        chosen_pos = rng.choices(range(len(indices)), weights=weights, k=n)
        # Deduplicate while preserving hardness bias (then fill if needed)
        seen = set()
        chosen: List[int] = []
        for pos in chosen_pos:
            idx = indices[pos]
            if idx not in seen:
                seen.add(idx)
                chosen.append(idx)
        if len(chosen) < n:
            remaining = [i for i in indices if i not in seen]
            rng.shuffle(remaining)
            chosen.extend(remaining[: n - len(chosen)])
        rng.shuffle(chosen)
        return chosen[:n]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def sample_indices(self, seed: Optional[int] = None) -> List[int]:
        """Deterministic (when seed is set) anatomy-stratified index draw."""
        rng = random.Random(seed) if seed is not None else random

        pool_idx = list(self.curriculum._pool)
        pool_entries = [self.entries[i] for i in pool_idx]

        fam_to_local: Dict[str, List[int]] = defaultdict(list)
        for local_i, e in enumerate(pool_entries):
            fam_to_local[e.anatomy_family].append(local_i)

        families = sorted(fam_to_local.keys())
        n_fam = max(1, len(families))
        base_q = self.samples_per_epoch // n_fam

        sampled_local = []
        for fam in families:
            w = self.anatomy_weights.get(fam, 1.0)
            q = max(10, int(base_q * w))
            q = min(q, len(fam_to_local[fam]))
            sampled_local.extend(rng.sample(fam_to_local[fam], q))

        sampled_global = [pool_idx[li] for li in sampled_local]
        rng.shuffle(sampled_global)
        base = sampled_global[: self.samples_per_epoch]
        return self._apply_hardness_reweight(base, rng)

    def __iter__(self) -> Iterator[int]:
        return iter(self.sample_indices(seed=None))


# ── DDP wrapper for CombinedSampler ───────────────────────────────────────────

class DistributedCombinedSampler(Sampler):
    """
    Shards CombinedSampler indices across DDP ranks with epoch-seeded shuffles.
    """

    def __init__(
        self,
        combined: CombinedSampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        drop_last: bool = True,
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.combined = combined
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def update_step(self, step: int) -> None:
        self.combined.update_step(step)

    @property
    def curriculum(self):
        return self.combined.curriculum

    def current_alpha(self) -> float:
        return self.combined.current_alpha()

    def current_mask_ratio(self) -> float:
        return self.combined.current_mask_ratio()

    def current_n_frames(self) -> int:
        return self.combined.current_n_frames()

    def current_stage(self) -> int:
        return self.combined.current_stage()

    def current_hardness_weight(self) -> float:
        return self.combined.current_hardness_weight()

    def current_mask_guidance_threshold(self) -> float:
        return self.combined.current_mask_guidance_threshold()

    def __len__(self) -> int:
        n = len(self.combined)
        if self.drop_last:
            return n // self.num_replicas
        return math.ceil(n / self.num_replicas)

    def __iter__(self) -> Iterator[int]:
        indices = self.combined.sample_indices(seed=self.epoch)
        if self.drop_last:
            total = len(indices) - (len(indices) % self.num_replicas)
            indices = indices[:total]
        else:
            total = len(indices)
        per_rank = total // self.num_replicas
        start = self.rank * per_rank
        return iter(indices[start : start + per_rank])


# ── Quality-weighted sampler (optional) ──────────────────────────────────────

class QualityWeightedSampler(Sampler):
    """
    Sample with probability proportional to SonoDQS quality score.
    Diamond/Platinum datasets are sampled more frequently.
    Useful when mixing high- and low-quality datasets.
    """
    def __init__(
        self,
        entries: List[USManifestEntry],
        samples_per_epoch: int = 200_000,
        temperature: float = 0.5,   # <1 → sharper quality weighting
    ):
        scores = torch.tensor([float(e.quality_score) for e in entries])
        logits = scores / temperature
        self.weights = torch.softmax(logits, dim=0)
        self.samples_per_epoch = samples_per_epoch
        self.n = len(entries)

    def __len__(self): return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        idx = torch.multinomial(
            self.weights, self.samples_per_epoch, replacement=True
        )
        return iter(idx.tolist())
