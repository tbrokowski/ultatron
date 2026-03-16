"""
alp_feedback.py — Adaptive Learning Priority (ALP) Feedback Loop
=================================================================

Implements the self-adaptive masking curriculum:

    ALP_k = α · S_k + (1-α) · H_k

where:
    S_k  = teacher saliency for patch/region k
           (computed from teacher attention maps, high = important region)
    H_k  = student hardness for patch/region k
           (latent prediction error, high = region student struggles with)
    α    = stage-dependent blend controlled by CurriculumSampler.current_alpha()

Data flow
---------
  Training step
    1. Forward image teacher → collect attention maps → S_k
    2. Forward student + predictor → compute per-patch L2 error → H_k
    3. ALP_k = α·S_k + (1-α)·H_k per patch
    4. HardnessFeedback.update(sample_ids, alp_scores)  ← writes to ALPScoreCache
       HardnessFeedback.update_saliency(sample_ids, saliency_maps)

  Next DataLoader iteration
    5. ImageSSLDataset.__getitem__ calls ALPScoreCache.get(sample_id)
       → passes ALP scores to ImageSSLTransform
       → transform biases patch masking toward high-ALP patches
    6. HardnessAwareSampler reads aggregate per-sample hardness
       → up-weights harder samples in __iter__

Multi-GPU usage
---------------
On distributed training, each rank maintains its own cache shard.
Scores are only written for samples processed on that rank; the sampler
reads from the local shard which naturally covers its partition.
For cross-rank sharing, call ALPScoreCache.sync_across_ranks() once per epoch.
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Sampler
from data.pipeline.alp_interface import ALPReader, NullALPReader

log = logging.getLogger(__name__)


# ── ALP Score Cache ────────────────────────────────────────────────────────────

class ALPScoreCache:
    """
    Thread-safe in-memory cache of per-sample ALP scores.

    Cache entry: {sample_id → ALPEntry}

    ALPEntry holds:
        saliency   : float32 ndarray (n_patches,)  — teacher attention saliency S_k
        hardness   : float32 ndarray (n_patches,)  — student prediction error H_k
        alp        : float32 ndarray (n_patches,)  — ALP_k = α·S_k + (1-α)·H_k
        step       : int                            — training step when last updated
        hit_count  : int                            — how many times sample was loaded

    When alpha changes (curriculum stage transition) the ALP field is re-derived
    on access without re-fetching saliency/hardness separately.

    Parameters
    ----------
    max_entries : int
        LRU cache capacity. Entries exceeding this are evicted by least-recently used.
    disk_cache_dir : str or None
        If set, newly computed scores are persisted to disk as .npz files so they
        survive restarts. Path: {disk_cache_dir}/{sample_id}.npz
    """

    def __init__(
        self,
        max_entries: int = 1_000_000,
        disk_cache_dir: Optional[str] = None,
    ):
        self.max_entries = max_entries
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None
        if self.disk_cache_dir:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # OrderedDict used as LRU (move-to-end on access)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def update_saliency(
        self,
        sample_ids: List[str],
        saliency_maps: Tensor,   # (B, n_patches) or (B, H_p, W_p)
        step: int = 0,
    ):
        """Write teacher saliency scores. Called after each image teacher forward."""
        sal = saliency_maps.detach().cpu().float()
        if sal.ndim == 3:
            sal = sal.flatten(1)  # (B, n_patches)
        with self._lock:
            for sid, s in zip(sample_ids, sal):
                self._write(sid, saliency=s.numpy(), step=step)

    def update_hardness(
        self,
        sample_ids: List[str],
        per_patch_errors: Tensor,  # (B, n_patches) — (pred - target)² per patch
        step: int = 0,
    ):
        """Write student prediction error. Called after masked patch distillation."""
        err = per_patch_errors.detach().cpu().float()
        if err.ndim == 3:
            err = err.flatten(1)
        with self._lock:
            for sid, h in zip(sample_ids, err):
                self._write(sid, hardness=h.numpy(), step=step)

    def _write(
        self,
        sample_id: str,
        saliency: Optional[np.ndarray] = None,
        hardness: Optional[np.ndarray] = None,
        step: int = 0,
    ):
        """Internal: update or create cache entry. Lock must be held."""
        if sample_id in self._cache:
            entry = self._cache.pop(sample_id)  # remove for LRU promotion
        else:
            entry = {"saliency": None, "hardness": None, "step": 0, "hit_count": 0}
            if len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)  # evict LRU

        if saliency is not None:
            entry["saliency"] = saliency
        if hardness is not None:
            entry["hardness"] = hardness
        entry["step"] = max(entry["step"], step)
        # Invalidate derived ALP so it's recomputed on next get()
        entry.pop("alp", None)

        self._cache[sample_id] = entry  # re-insert at end (MRU)

        # Persist to disk
        if self.disk_cache_dir is not None:
            path = self.disk_cache_dir / f"{sample_id}.npz"
            saves = {}
            if entry["saliency"] is not None: saves["saliency"] = entry["saliency"]
            if entry["hardness"] is not None: saves["hardness"] = entry["hardness"]
            if saves:
                np.savez(path, **saves)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(
        self,
        sample_id: str,
        alpha: float = 1.0,
        n_patches: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Return ALP_k scores for this sample, or None if not cached.

        ALP_k = α · S_k + (1-α) · H_k
        Both S_k and H_k are softmax-normalised before blending so they
        are on the same scale regardless of magnitude.

        Parameters
        ----------
        sample_id : str
        alpha     : float  — current α from CurriculumSampler.current_alpha()
        n_patches : int or None — expected number of patches (for validation)
        """
        with self._lock:
            entry = self._cache.get(sample_id)

        if entry is None:
            # Try disk fallback
            entry = self._load_from_disk(sample_id)
            if entry is None:
                self._misses += 1
                return None

        self._hits += 1

        # Move to MRU
        with self._lock:
            if sample_id in self._cache:
                self._cache.move_to_end(sample_id)
                self._cache[sample_id]["hit_count"] += 1

        sal = entry.get("saliency")
        hard = entry.get("hardness")

        if sal is None and hard is None:
            return None

        def _normalise(x: np.ndarray) -> np.ndarray:
            # Softmax normalisation for stable blending
            x = x - x.max()
            e = np.exp(x)
            return e / (e.sum() + 1e-8)

        if sal is not None and hard is not None:
            alp = alpha * _normalise(sal) + (1 - alpha) * _normalise(hard)
        elif sal is not None:
            alp = _normalise(sal)
        else:
            alp = _normalise(hard)

        if n_patches is not None and len(alp) != n_patches:
            # Resize (e.g. if image was resized between caching and loading)
            alp = np.interp(
                np.linspace(0, 1, n_patches),
                np.linspace(0, 1, len(alp)),
                alp,
            )

        return alp.astype(np.float32)

    def _load_from_disk(self, sample_id: str) -> Optional[dict]:
        if self.disk_cache_dir is None:
            return None
        path = self.disk_cache_dir / f"{sample_id}.npz"
        if not path.exists():
            return None
        try:
            data = np.load(path)
            entry = {
                "saliency": data["saliency"] if "saliency" in data else None,
                "hardness": data["hardness"] if "hardness" in data else None,
                "step": 0,
                "hit_count": 0,
            }
            # Warm the in-memory cache
            with self._lock:
                self._cache[sample_id] = entry
            return entry
        except Exception as e:
            log.warning(f"Failed to load ALP cache from disk for {sample_id}: {e}")
            return None

    # ── Aggregate per-sample hardness (for HardnessAwareSampler) ─────────────

    def sample_aggregate_hardness(self, sample_id: str) -> float:
        """
        Returns mean hardness score for this sample (scalar).
        Used by HardnessAwareSampler to bias sampling toward hard samples.
        Returns 0.5 (neutral) if not cached.
        """
        with self._lock:
            entry = self._cache.get(sample_id)
        if entry is None or entry.get("hardness") is None:
            return 0.5
        return float(np.mean(entry["hardness"]))

    # ── Sync across ranks ─────────────────────────────────────────────────────

    def sync_across_ranks(self):
        """
        Placeholder for distributed sync.
        In practice: use torch.distributed all_gather on serialised score dicts,
        or rely on the shared disk cache with NFS/CSCS Lustre.
        On CSCS, Lustre is shared across nodes so disk_cache_dir sync is free.
        """
        log.debug("ALPScoreCache.sync_across_ranks(): using disk cache as sync medium.")

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (f"ALPScoreCache(entries={len(self)}, "
                f"hit_rate={self.hit_rate:.2%}, "
                f"disk={self.disk_cache_dir})")


# ── Hardness Feedback (called from training loop) ──────────────────────────────

class HardnessFeedback:
    """
    Called by the training step to push per-patch prediction errors
    back into the ALPScoreCache.

    Typical usage inside phase1_step / phase3_step
    -----------------------------------------------
        # After computing student patch predictions:
        patch_errors = (student_patches - teacher_patches).pow(2).mean(-1)  # (B, N_patches)
        feedback.update(batch["sample_ids"], patch_errors, global_step)

        # After teacher forward (for saliency):
        attn = img_branch.teacher.get_attention_maps()  # (B, H, N, N)
        saliency = attn.mean(1).mean(1)  # (B, N) — mean over heads and query tokens
        feedback.update_saliency(batch["sample_ids"], saliency, global_step)
    """

    def __init__(self, cache: ALPScoreCache):
        self.cache = cache
        self._step_count = 0

    def update(
        self,
        sample_ids: List[str],
        per_patch_errors: Tensor,  # (B, N_patches)
        step: int,
    ):
        """Write student prediction errors for this batch."""
        self.cache.update_hardness(sample_ids, per_patch_errors, step)
        self._step_count += 1

    def update_saliency(
        self,
        sample_ids: List[str],
        attention_maps: Tensor,    # (B, n_heads, N, N) or (B, N)
        step: int,
    ):
        """
        Derive and write teacher saliency from attention maps.
        If shape is (B, n_heads, N, N): average over heads and [CLS] attention
        to get (B, N_patch) saliency per patch.
        """
        if attention_maps.ndim == 4:
            # (B, H, N, N) → take CLS token row (index 0), avg over heads
            sal = attention_maps[:, :, 0, 1:].mean(1)  # (B, N_patches)
        else:
            sal = attention_maps  # already (B, N_patches)
        self.cache.update_saliency(sample_ids, sal, step)


# ── Hardness-Aware Sampler ─────────────────────────────────────────────────────

class HardnessAwareSampler(Sampler):
    """
    Extends curriculum + anatomy sampling with ALP-based sample weighting.

    Samples with high mean hardness are drawn more frequently, implementing
    the OPENUS curriculum principle of prioritising informative, hard regions.

    Parameters
    ----------
    base_sampler : CombinedSampler
        Provides the curriculum-filtered, anatomy-stratified pool.
    alp_cache    : ALPScoreCache
        Source of per-sample aggregate hardness scores.
    hardness_weight : float
        Mix coefficient: 0.0 = base sampler only, 1.0 = pure hardness weighting.
        Increases with stage (Stage1=0.0, Stage2=0.3, Stage3=0.6).
    temperature  : float
        Softmax temperature for hardness-based weights. Lower = sharper.
    """

    def __init__(
        self,
        base_sampler,                        # CombinedSampler
        alp_cache: ALPReader,  # ALPScoreCache at runtime
        entries,                             # List[USManifestEntry]
        hardness_weight: float = 0.3,
        temperature: float = 1.0,
    ):
        self.base_sampler = base_sampler
        self.alp_cache = alp_cache
        self.entries = entries
        self.hardness_weight = hardness_weight
        self.temperature = temperature

    def update_step(self, step: int):
        self.base_sampler.update_step(step)

    def current_alpha(self) -> float:
        return self.base_sampler.current_alpha()

    def current_mask_ratio(self) -> float:
        return self.base_sampler.current_mask_ratio()

    def current_n_frames(self) -> int:
        return self.base_sampler.current_n_frames()

    def __len__(self) -> int:
        return len(self.base_sampler)

    def __iter__(self) -> Iterator[int]:
        # Get base pool indices from curriculum + anatomy sampler
        base_indices = list(self.base_sampler)

        if self.hardness_weight <= 0.0:
            return iter(base_indices)

        # Compute hardness weights for pool
        hardness = np.array([
            self.alp_cache.sample_aggregate_hardness(self.entries[i].sample_id)
            for i in base_indices
        ], dtype=np.float32)

        # Softmax over pool
        hardness_norm = hardness / self.temperature
        hardness_norm -= hardness_norm.max()
        weights_hard = np.exp(hardness_norm)
        weights_hard /= weights_hard.sum()

        # Uniform weights for base sampler
        weights_uniform = np.ones(len(base_indices), dtype=np.float32)
        weights_uniform /= weights_uniform.sum()

        # Blend
        w = self.hardness_weight
        weights = (1 - w) * weights_uniform + w * weights_hard

        # Sample without replacement (approximate via weighted multinomial)
        n = min(len(self.base_sampler), len(base_indices))
        chosen_local = np.random.choice(
            len(base_indices), size=n, replace=False,
            p=weights / weights.sum(),
        )
        chosen = [base_indices[i] for i in chosen_local]

        import random
        random.shuffle(chosen)
        return iter(chosen)


# ── Global singleton (optional convenience) ────────────────────────────────────

_global_cache: Optional[ALPScoreCache] = None


def get_alp_cache() -> ALPScoreCache:
    global _global_cache
    if _global_cache is None:
        _global_cache = ALPScoreCache()
    return _global_cache


def configure_alp_cache(
    max_entries: int = 1_000_000,
    disk_cache_dir: Optional[str] = None,
) -> ALPScoreCache:
    global _global_cache
    _global_cache = ALPScoreCache(max_entries, disk_cache_dir)
    return _global_cache
