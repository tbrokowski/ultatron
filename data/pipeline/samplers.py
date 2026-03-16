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

import torch
from torch.utils.data import Sampler

from data.schema.manifest import USManifestEntry
from data.pipeline.alp_interface import ALPReader, NullALPReader


# ── Anatomy-stratified sampler ────────────────────────────────────────────────

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
    ):
        self.entries = entries
        self.total_steps = total_steps
        self.samples_per_epoch = samples_per_epoch

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
        if frac < 0.33: return 1
        if frac < 0.66: return 2
        return 3

    def current_alpha(self) -> float:
        """
        Alpha for ALP_k = alpha * S_k + (1-alpha) * H_k
        Stage 1: alpha=1.0 (saliency only)
        Stage 2: alpha=0.7
        Stage 3: alpha=0.4 (mix saliency + hardness)
        """
        return {1: 1.0, 2: 0.7, 3: 0.4}[self._stage(self.current_step)]

    def current_mask_ratio(self) -> float:
        """Mask ratio increases with difficulty stage."""
        return {1: 0.40, 2: 0.65, 3: 0.80}[self._stage(self.current_step)]

    def current_n_frames(self) -> int:
        """Clip length increases with stage."""
        return {1: 8, 2: 16, 3: 32}[self._stage(self.current_step)]

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
    ):
        self.curriculum = CurriculumSampler(
            entries, global_step, total_steps, samples_per_epoch * 4
        )
        self.samples_per_epoch = samples_per_epoch
        self.entries = entries
        self.anatomy_weights = anatomy_weights or {}
        self._current_pool_entries: Optional[List] = None

    def update_step(self, step: int):
        self.curriculum.update_step(step)

    def current_alpha(self) -> float:
        return self.curriculum.current_alpha()

    def current_mask_ratio(self) -> float:
        return self.curriculum.current_mask_ratio()

    def current_n_frames(self) -> int:
        return self.curriculum.current_n_frames()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[int]:
        # Get curriculum-filtered pool indices
        pool_idx = list(self.curriculum._pool)
        pool_entries = [self.entries[i] for i in pool_idx]

        # Anatomy-stratified sub-sample
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
            sampled_local.extend(random.sample(fam_to_local[fam], q))

        # Map back to global indices
        sampled_global = [pool_idx[li] for li in sampled_local]
        random.shuffle(sampled_global)
        return iter(sampled_global[:self.samples_per_epoch])


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
