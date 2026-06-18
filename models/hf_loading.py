"""
models/hf_loading.py  ·  Offline-first HuggingFace model loading
=================================================================

Gated models (DINOv3, V-JEPA2, …) still trigger a Hub auth check on
from_pretrained() even when weights are cached.  On compute nodes without a
token that fails with 401.  When a complete snapshot exists in cache_dir we
load with local_files_only=True to skip the network round-trip.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List, Optional, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


def _fallback_cache_dirs() -> List[Path]:
    try:
        from data.infra.cscs_paths import HF_CACHE_STORE
        return [Path(HF_CACHE_STORE)]
    except ImportError:
        return []


def _search_cache_dirs(hf_cache_dir: Optional[str | Path]) -> List[Path]:
    """Primary cache first, then shared Capstor store."""
    dirs: List[Path] = []
    if hf_cache_dir:
        dirs.append(Path(hf_cache_dir))
    for fb in _fallback_cache_dirs():
        if fb not in dirs:
            dirs.append(fb)
    return dirs


def _cache_snapshot_dir(hf_id: str, cache_dir: str | Path) -> Optional[Path]:
    """Return the newest snapshot dir for hf_id under cache_dir, or None."""
    root = Path(cache_dir) / ("models--" + hf_id.replace("/", "--")) / "snapshots"
    if not root.is_dir():
        return None
    candidates = sorted(
        (p for p in root.iterdir() if p.is_dir() and (p / "config.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_cached_model_dir(
    hf_id: str,
    hf_cache_dir: Optional[str | Path] = None,
) -> Optional[Path]:
    """Return the cache root that holds a complete snapshot for hf_id."""
    for cache_dir in _search_cache_dirs(hf_cache_dir):
        snap = _cache_snapshot_dir(hf_id, cache_dir)
        if snap is None:
            continue
        has_weights = any(snap.glob("*.safetensors")) or any(snap.glob("*.bin"))
        if has_weights:
            return cache_dir
    return None


def hf_model_is_cached(hf_id: str, cache_dir: Optional[str | Path]) -> bool:
    """True when config + weight files exist in the HF hub cache layout."""
    return find_cached_model_dir(hf_id, cache_dir) is not None


def list_cached_hf_ids(
    hf_cache_dir: Optional[str | Path] = None,
    *,
    prefix: str = "",
) -> list[str]:
    """List hub model IDs with complete snapshots under known cache dirs."""
    found: list[str] = []
    for cache_dir in _search_cache_dirs(hf_cache_dir):
        root = Path(cache_dir)
        if not root.is_dir():
            continue
        for model_dir in sorted(root.glob("models--*")):
            hf_id = model_dir.name.replace("models--", "").replace("--", "/", 1)
            if prefix and not hf_id.startswith(prefix):
                continue
            if hf_model_is_cached(hf_id, cache_dir):
                found.append(hf_id)
    return sorted(set(found))


def resolve_dinov3_variant(
    variant: str,
    hf_cache_dir: Optional[str | Path] = None,
) -> str:
    """
    Pick a DINOv3 registry key that is actually cached offline.

    Capstor store typically has vit-l / vit-s / vit-7b but not vit-b; when the
    requested variant is missing we fall back to the closest cached size.
    """
    from models.image_backbones.dinov3 import _DINOV3_HF_IDS

    if variant not in _DINOV3_HF_IDS:
        return variant

    hf_id = _DINOV3_HF_IDS[variant]
    if hf_model_is_cached(hf_id, hf_cache_dir):
        return variant

    fallback_order = {
        "dinov3_b": ("dinov3_l", "dinov3_s", "dinov3_hplus", "dinov3_splus"),
        "dinov3_hplus": ("dinov3_l", "dinov3_b", "dinov3_s"),
        "dinov3_splus": ("dinov3_s", "dinov3_b", "dinov3_l"),
    }
    for alt in fallback_order.get(variant, ("dinov3_l", "dinov3_s")):
        alt_id = _DINOV3_HF_IDS.get(alt)
        if alt_id and hf_model_is_cached(alt_id, hf_cache_dir):
            log.warning(
                "%s not in HF cache (%s); using cached %s instead",
                variant, hf_id, alt,
            )
            return alt

    cached = list_cached_hf_ids(hf_cache_dir, prefix="facebook/dinov3")
    raise FileNotFoundError(
        f"DINOv3 variant {variant!r} ({hf_id}) is not cached locally. "
        f"Cached DINOv3 models: {cached or 'none'}. "
        "Download on a login node or switch comparison config to dinov3_l."
    )


def _hf_token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_pretrained(
    model_cls: type[T],
    hf_id: str,
    *,
    hf_cache_dir: Optional[str | Path] = None,
    local_files_only: Optional[bool] = None,
    **kwargs: Any,
) -> T:
    """
    Load a HuggingFace model, preferring a local Capstor/scratch cache.

    Parameters
    ----------
    local_files_only : if None (default), auto-enabled when cache is complete
    """
    load_kwargs = dict(kwargs)
    cached_at = find_cached_model_dir(hf_id, hf_cache_dir)

    if local_files_only is None:
        local_files_only = cached_at is not None

    if cached_at is not None:
        load_kwargs["cache_dir"] = str(cached_at)

    if local_files_only:
        load_kwargs["local_files_only"] = True
        log.info("Loading %s offline from cache (%s)", hf_id, load_kwargs.get("cache_dir", "HF_HOME"))
    else:
        token = _hf_token()
        if token:
            load_kwargs["token"] = token
        log.info("Loading %s from HuggingFace Hub (cache=%s)", hf_id, hf_cache_dir or "default")

    try:
        return model_cls.from_pretrained(hf_id, **load_kwargs)
    except Exception as exc:
        if local_files_only or "401" in str(exc) or "gated" in str(exc).lower():
            cached = list_cached_hf_ids(hf_cache_dir)
            hint = (
                f" Cached models under {hf_cache_dir or 'HF_HOME'}: "
                f"{[c for c in cached if hf_id.split('/')[0] in c][:8] or 'none'}."
            )
            raise RuntimeError(
                f"Failed to load {hf_id} offline: {exc}.{hint}"
            ) from exc
        raise
