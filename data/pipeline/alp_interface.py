"""
/data/pipeline/alp_interface.py  ·  ALPReader protocol
===========================================================

This module defines the *read-only* interface that data/pipeline/ components
use to access Adaptive Learning Priority (ALP) scores at dataset load time.

"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ALPReader(Protocol):
    """
    Read-only interface for per-sample ALP scores.

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
