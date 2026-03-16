"""
oura/data/pipeline/alp_interface.py  ·  ALPReader protocol
===========================================================

Purpose
-------
This module defines the *read-only* interface that data/pipeline/ components
use to access Adaptive Learning Priority (ALP) scores at dataset load time.

The problem it solves
---------------------
samplers.py (HardnessAwareSampler) and dataset.py (ImageSSLDataset) both
need per-sample ALP scores at load time to bias masking and sampling.
Those scores are *written* by the training loop (train/alp.py →
ALPScoreCache).

Without this file the dependency graph has a cycle:
    data/pipeline/ → train/alp.py → data/pipeline/  ← CYCLE

The fix: data/pipeline/ depends only on the ALPReader Protocol defined
here.  train/alp.py (ALPScoreCache) implements ALPReader without data/
knowing about it.  The training loop wires them together at runtime.

Dependency graph after this file:
    data/pipeline/alp_interface.py   ← stdlib/typing only
    data/pipeline/samplers.py        ← imports ALPReader from here
    data/pipeline/dataset.py         ← imports ALPReader from here
    train/alp.py                     ← implements ALPReader (no import here)
    train/datamodule.py              ← creates ALPScoreCache, passes it in

Usage
-----
In dataset.py:
    from oura.data.pipeline.alp_interface import ALPReader, NullALPReader

    class ImageSSLDataset(Dataset):
        def __init__(self, ..., alp_reader: ALPReader = NullALPReader()):
            self.alp = alp_reader

        def __getitem__(self, idx):
            entry = self.entries[idx]
            alp_scores = self.alp.get(entry.sample_id, alpha=self.alpha)
            ...

In samplers.py:
    from oura.data.pipeline.alp_interface import ALPReader, NullALPReader

    class HardnessAwareSampler(Sampler):
        def __init__(self, ..., alp_reader: ALPReader = NullALPReader()):
            self.alp = alp_reader

        def __iter__(self):
            hardness = [
                self.alp.aggregate_hardness(self.entries[i].sample_id)
                for i in pool_indices
            ]
            ...

In train/datamodule.py (the wiring site):
    from oura.train.alp import ALPScoreCache
    cache = ALPScoreCache(disk_cache_dir=cscs.alp_cache_dir())
    dataset = ImageSSLDataset(..., alp_reader=cache)
    sampler = HardnessAwareSampler(..., alp_reader=cache)
    # The same cache object is passed to both and written to by the
    # training loop via HardnessFeedback.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ALPReader(Protocol):
    """
    Read-only interface for per-sample ALP scores.

    Implemented by:
        oura.train.alp.ALPScoreCache  (production, LRU + disk)
        oura.data.pipeline.alp_interface.NullALPReader  (testing / cold-start)

    All methods must be safe to call from multiple DataLoader worker
    processes simultaneously (i.e. they must not mutate shared state in
    a non-thread-safe way).
    """

    def get(
        self,
        sample_id: str,
        alpha: float = 1.0,
        n_patches: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Return ALP_k scores for this sample as a float32 ndarray of shape
        (n_patches,), or None if this sample has not been scored yet.

        Parameters
        ----------
        sample_id : str
            Unique sample identifier (USManifestEntry.sample_id).
        alpha : float
            Blend coefficient: ALP_k = alpha * S_k + (1-alpha) * H_k.
            Provided by CurriculumSampler.current_alpha() at call time.
        n_patches : int or None
            Expected patch count for the current crop resolution. If the
            cached scores have a different length, the implementation
            should resize or return None.

        Returns
        -------
        np.ndarray of shape (n_patches,), float32, values in [0, 1].
        Returns None if the sample is not in the cache (cold-start).
        """
        ...

    def aggregate_hardness(self, sample_id: str) -> float:
        """
        Return a scalar aggregate hardness score for this sample.
        Used by HardnessAwareSampler to weight sampling probability.

        Returns 0.5 (neutral) if the sample has not been scored.
        """
        ...


class NullALPReader:
    """
    No-op implementation of ALPReader.

    Used as the default when no ALP cache has been configured — e.g.
    during cold-start (first epoch before any scores exist), in unit
    tests, or when ALP is deliberately disabled.

    get() always returns None → transforms fall back to uniform masking.
    aggregate_hardness() returns 0.5 → uniform sampling weight.
    """

    def get(
        self,
        sample_id: str,
        alpha: float = 1.0,
        n_patches: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        return None

    def aggregate_hardness(self, sample_id: str) -> float:
        return 0.5

    def __repr__(self) -> str:
        return "NullALPReader()"
