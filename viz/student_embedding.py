"""
viz/student_embedding.py  ·  Student model embedding extraction + UMAP/t-SNE viz
==================================================================================

Extract frozen global embeddings from a Hiera student checkpoint and plot
anatomy-family UMAP/t-SNE panels (foundation-model representation analysis).

Usage (library)
---------------
    from finetune.backbones.student_encoder import StudentEncoder
    from viz.student_embedding import StudentEmbeddingExtractor, run_embedding_viz

    enc = StudentEncoder("checkpoints/latest.pt", use_ema=True, device="cuda")
    extractor = StudentEmbeddingExtractor(enc, dm, device="cuda")
    run_embedding_viz(extractor, output_dir="viz_outputs/run1", method="both")
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.schema.manifest import normalize_anatomy
from data.pipeline.student_datamodule import SOURCE_MASK_PATCH_STRIDE, upsample_mask_grid
from viz import features as feat_viz

log = logging.getLogger(__name__)


def build_balanced_indices(
    entries: list,
    samples_per_family: int,
    seed: int = 42,
) -> tuple[list[int], dict[str, int]]:
    """
    Pick dataset indices with up to ``samples_per_family`` entries per anatomy family.

    Every family present in ``entries`` is included (even rare ones with <N samples).
    """
    rng = np.random.default_rng(seed)
    by_fam: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        fam = normalize_anatomy(getattr(entry, "anatomy_family", "other"))
        by_fam[fam].append(idx)

    selected: list[int] = []
    counts: dict[str, int] = {}
    for fam in sorted(by_fam):
        idxs = by_fam[fam]
        k = min(len(idxs), samples_per_family)
        if len(idxs) < samples_per_family:
            log.info(
                "Anatomy %s: using all %d samples (requested %d per family)",
                fam, len(idxs), samples_per_family,
            )
        chosen = rng.choice(idxs, size=k, replace=False)
        selected.extend(chosen.tolist())
        counts[fam] = int(k)

    rng.shuffle(selected)
    log.info(
        "Balanced index plan: %d families, %d total samples (≤%d per family)",
        len(counts), len(selected), samples_per_family,
    )
    return selected, counts


def stratified_subsample(
    features: np.ndarray,
    anatomies: list[str],
    dataset_ids: Optional[list[str]] = None,
    sample_ids: Optional[list[str]] = None,
    stream_types: Optional[list[str]] = None,
    max_samples: int = 10_000,
    min_per_family: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], Optional[list[str]], Optional[list[str]], Optional[list[str]], dict]:
    """
    Cap total points while preserving anatomy-family coverage.

    Families with fewer than ``min_per_family`` samples are dropped (with a warning).
    """
    rng = np.random.default_rng(seed)
    anatomies = [normalize_anatomy(a) for a in anatomies]
    n = len(features)

    by_fam: dict[str, list[int]] = defaultdict(list)
    for i, fam in enumerate(anatomies):
        by_fam[fam].append(i)

    dropped = [fam for fam, idxs in by_fam.items() if len(idxs) < min_per_family]
    kept_fams = {fam: idxs for fam, idxs in by_fam.items() if len(idxs) >= min_per_family}

    if dropped:
        log.warning(
            "Dropping %d anatomy families with <%d samples: %s",
            len(dropped), min_per_family, sorted(dropped),
        )

    if not kept_fams:
        raise ValueError(
            f"No anatomy family has >={min_per_family} samples "
            f"(extracted {n} total)."
        )

    all_idxs = list(range(n))
    if len(all_idxs) <= max_samples:
        selected = all_idxs
    else:
        n_fams = len(kept_fams)
        per_fam = max(min_per_family, max_samples // n_fams)
        selected: list[int] = []
        for fam in sorted(kept_fams):
            idxs = kept_fams[fam]
            k = min(len(idxs), per_fam)
            selected.extend(rng.choice(idxs, size=k, replace=False).tolist())
        if len(selected) > max_samples:
            selected = rng.choice(selected, size=max_samples, replace=False).tolist()

    selected = sorted(selected)
    sub_feats = features[selected]
    sub_anat = [anatomies[i] for i in selected]
    sub_ds = [dataset_ids[i] for i in selected] if dataset_ids is not None else None
    sub_sids = [sample_ids[i] for i in selected] if sample_ids is not None else None
    sub_streams = [stream_types[i] for i in selected] if stream_types is not None else None

    stats = {
        "n_before": n,
        "n_after": len(selected),
        "per_anatomy": dict(Counter(sub_anat)),
        "dropped_families": sorted(dropped),
    }
    if sub_streams is not None:
        stats["per_stream"] = dict(Counter(sub_streams))
    return sub_feats, sub_anat, sub_ds, sub_sids, sub_streams, stats


class StudentEmbeddingExtractor:
    """
    One-pass frozen feature extraction from a student Hiera backbone.

    Parameters
    ----------
    encoder : StudentEncoder
    dm : USFoundationDataModule  (must be setup() already)
    device : str
    embedding_space : ``"global"`` (1152-d) or ``"global_proj"`` (1024-d align space)
    """

    def __init__(
        self,
        encoder,
        dm,
        device: str = "cuda",
        embedding_space: str = "global_proj",
        student_patch_stride: int = 4,
    ):
        self.encoder = encoder
        self.dm = dm
        self.device = device
        self.embedding_space = embedding_space
        self.student_patch_stride = student_patch_stride
        self._mask_upsample = SOURCE_MASK_PATCH_STRIDE // student_patch_stride
        self.encoder.to(device)
        self.encoder.eval()

    def _to_student_padding_mask(self, pmask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Stride-16 collator masks → Hiera stride-4 grid (matches training collator)."""
        if pmask is None or self._mask_upsample <= 1:
            return pmask
        return upsample_mask_grid(pmask, self._mask_upsample)

    def _select_embedding(self, out: dict) -> torch.Tensor:
        if self.embedding_space == "global_proj":
            if "global_proj" in out:
                return out["global_proj"]
            log.warning("global_proj unavailable — falling back to global (1152-d)")
        elif self.embedding_space != "global":
            raise ValueError(
                f"embedding_space must be 'global' or 'global_proj', got {self.embedding_space!r}"
            )
        return out["global"]

    def _loader_for_split(self, split: str, stream: str):
        if split == "val":
            return self.dm.val_loader(stream=stream)
        if split == "train":
            if stream == "image":
                return self.dm.image_loader()
            if stream == "video":
                return self.dm.video_loader()
            raise ValueError(f"train split does not support stream={stream!r}")
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    def _dataset_and_entries(self, split: str, stream: str):
        self.dm.setup()
        if stream == "image":
            return self.dm._image_dataset, self.dm._image_entries
        if stream == "video":
            return self.dm._video_dataset, self.dm._video_entries
        raise ValueError(f"unsupported stream={stream!r}")

    def _forward_batch(self, batch: dict, stream: str) -> np.ndarray:
        if stream == "image":
            crops = batch["global_crops"][:, 0].to(self.device, non_blocking=True)
            pmask = batch.get("global_pmasks")
            if pmask is not None:
                pmask = self._to_student_padding_mask(pmask[:, 0].to(self.device, non_blocking=True))
            x = crops.unsqueeze(1)
        else:
            x = batch["full_clips"].to(self.device, non_blocking=True)
            pmask = batch.get("padding_masks")
            if pmask is not None:
                pmask = self._to_student_padding_mask(pmask.to(self.device, non_blocking=True))

        out = self.encoder.backbone(x, padding_mask=pmask)
        emb = self._select_embedding(out).float()
        return F.normalize(emb, dim=-1).cpu().numpy()

    @torch.no_grad()
    def extract_balanced(
        self,
        split: str = "train",
        stream: str = "image",
        samples_per_family: int = 500,
        batch_size: int = 32,
        seed: int = 42,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """
        Extract embeddings with fixed per-anatomy sampling from the manifest.

        Guarantees every anatomy family in the stream is represented.
        """
        if split != "train":
            raise ValueError(
                "Balanced extraction uses train-stream datasets; "
                f"got split={split!r}. Use split='train'."
            )

        dataset, entries = self._dataset_and_entries(split, stream)
        indices, family_plan = build_balanced_indices(entries, samples_per_family, seed=seed)

        if stream == "image":
            from data.pipeline.collators import ImageSSLCollator
            collator = ImageSSLCollator()
        else:
            from data.pipeline.collators import VideoSSLCollator
            collator = VideoSSLCollator()
            batch_size = min(batch_size, getattr(self.dm, "video_batch_size", 1))

        all_feats: list[np.ndarray] = []
        all_anatomies: list[str] = []
        all_dataset_ids: list[str] = []
        all_sample_ids: list[str] = []

        for start in range(0, len(indices), batch_size):
            batch_idxs = indices[start : start + batch_size]
            samples = [dataset[i] for i in batch_idxs]
            batch = collator(samples)
            emb = self._forward_batch(batch, stream)

            all_feats.append(emb)
            all_anatomies.extend(
                normalize_anatomy(a)
                for a in batch.get("anatomy_families", ["other"] * emb.shape[0])
            )
            all_dataset_ids.extend(batch.get("dataset_ids", [""] * emb.shape[0]))
            all_sample_ids.extend(batch.get("sample_ids", [""] * emb.shape[0]))

        features = np.concatenate(all_feats, axis=0)
        metadata = {
            "stream": stream,
            "split": split,
            "balance_anatomy": True,
            "samples_per_family": samples_per_family,
            "family_plan": family_plan,
            "n_families": len(family_plan),
            "embedding_space": self.embedding_space,
            "n_extracted": len(features),
            "embed_dim": int(features.shape[1]),
            "dataset_ids": all_dataset_ids,
            "sample_ids": all_sample_ids,
            "per_anatomy_raw": dict(Counter(all_anatomies)),
        }
        log.info(
            "Balanced extract: %d families, %d %s features, D=%d",
            len(family_plan), len(features), stream, features.shape[1],
        )
        return features, all_anatomies, metadata

    @torch.no_grad()
    def extract(
        self,
        split: str = "train",
        stream: str = "image",
        max_samples: int = 50_000,
        balance_anatomy: bool = True,
        samples_per_family: int = 500,
        batch_size: int = 32,
        seed: int = 42,
    ) -> tuple[np.ndarray, list[str], dict[str, Any]]:
        """
        Extract L2-normalised global embeddings.

        Default (``balance_anatomy=True``): sample ``samples_per_family`` images
        per anatomy family from the pretrain train manifest — every family included.

        Sequential mode (``balance_anatomy=False``): walk the dataloader up to
        ``max_samples`` (may miss rare families).
        """
        if balance_anatomy:
            return self.extract_balanced(
                split=split,
                stream=stream,
                samples_per_family=samples_per_family,
                batch_size=batch_size,
                seed=seed,
            )

        loader = self._loader_for_split(split, stream)
        loader_iter = iter(loader)

        all_feats: list[np.ndarray] = []
        all_anatomies: list[str] = []
        all_dataset_ids: list[str] = []
        all_sample_ids: list[str] = []
        n_seen = 0

        while n_seen < max_samples:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            emb = self._forward_batch(batch, stream)

            bsz = emb.shape[0]
            all_feats.append(emb)
            all_anatomies.extend(
                normalize_anatomy(a)
                for a in batch.get("anatomy_families", ["other"] * bsz)
            )
            all_dataset_ids.extend(batch.get("dataset_ids", [""] * bsz))
            all_sample_ids.extend(batch.get("sample_ids", [""] * bsz))
            n_seen += bsz

        if not all_feats:
            raise RuntimeError(f"No samples extracted for stream={stream!r}")

        features = np.concatenate(all_feats, axis=0)[:max_samples]
        anatomies = all_anatomies[:max_samples]
        dataset_ids = all_dataset_ids[:max_samples]
        sample_ids = all_sample_ids[:max_samples]

        metadata = {
            "stream": stream,
            "split": split,
            "balance_anatomy": False,
            "embedding_space": self.embedding_space,
            "n_extracted": len(features),
            "embed_dim": int(features.shape[1]),
            "dataset_ids": dataset_ids,
            "sample_ids": sample_ids,
            "per_anatomy_raw": dict(Counter(anatomies)),
        }
        log.info(
            "Extracted %d %s features, D=%d (space=%s)",
            len(features), stream, features.shape[1], self.embedding_space,
        )
        return features, anatomies, metadata


def save_embedding_cache(
    path: Path,
    features: np.ndarray,
    anatomy_families: list[str],
    dataset_ids: Optional[list[str]] = None,
    sample_ids: Optional[list[str]] = None,
    stream_types: Optional[list[str]] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Persist embeddings for re-plotting without re-inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "features": features,
        "anatomy_families": np.array(anatomy_families, dtype=object),
    }
    if dataset_ids is not None:
        payload["dataset_ids"] = np.array(dataset_ids, dtype=object)
    if sample_ids is not None:
        payload["sample_ids"] = np.array(sample_ids, dtype=object)
    if stream_types is not None:
        payload["stream_types"] = np.array(stream_types, dtype=object)
    if extra:
        payload["meta"] = np.array([json.dumps(extra)], dtype=object)
    np.savez_compressed(str(path), **payload)
    log.info("Cached embeddings → %s", path)
    return path


def load_embedding_cache(path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Load embeddings.npz written by ``save_embedding_cache``."""
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)
    features = data["features"]
    anatomies = data["anatomy_families"].tolist()
    meta: dict = {}
    if "meta" in data:
        meta = json.loads(str(data["meta"][0]))
    dataset_ids = data["dataset_ids"].tolist() if "dataset_ids" in data else None
    if dataset_ids is not None:
        meta["dataset_ids"] = dataset_ids
    sample_ids = data["sample_ids"].tolist() if "sample_ids" in data else None
    if sample_ids is not None:
        meta["sample_ids"] = sample_ids
    stream_types = data["stream_types"].tolist() if "stream_types" in data else None
    if stream_types is not None:
        meta["stream_types"] = stream_types
    elif meta.get("stream") in ("image", "video"):
        meta["stream_types"] = [meta["stream"]] * len(features)
    log.info("Loaded %d cached embeddings from %s", len(features), path)
    return features, anatomies, meta


def _extract_streams(
    extractor: StudentEmbeddingExtractor,
    *,
    split: str,
    stream: str,
    balance_anatomy: bool,
    samples_per_family: int,
    max_samples: int,
    batch_size: int,
    video_batch_size: Optional[int] = None,
    seed: int,
) -> tuple[np.ndarray, list[str], dict]:
    """Extract one stream or concatenate image + video for ``stream='both'``."""
    vid_bs = video_batch_size if video_batch_size is not None else batch_size
    if stream == "both":
        img_feats, img_anat, img_meta = extractor.extract(
            split=split, stream="image",
            balance_anatomy=balance_anatomy,
            samples_per_family=samples_per_family,
            max_samples=max_samples,
            batch_size=batch_size,
            seed=seed,
        )
        vid_feats, vid_anat, vid_meta = extractor.extract(
            split=split, stream="video",
            balance_anatomy=balance_anatomy,
            samples_per_family=samples_per_family,
            max_samples=max_samples,
            batch_size=vid_bs,
            seed=seed,
        )
        features = np.concatenate([img_feats, vid_feats], axis=0)
        anatomies = img_anat + vid_anat
        stream_types = ["image"] * len(img_feats) + ["video"] * len(vid_feats)
        meta = {
            **img_meta,
            "stream": "both",
            "stream_types": stream_types,
            "n_image": len(img_feats),
            "n_video": len(vid_feats),
            "family_plan_image": img_meta.get("family_plan", {}),
            "family_plan_video": vid_meta.get("family_plan", {}),
            "per_anatomy_raw": dict(Counter(anatomies)),
            "n_extracted": len(features),
        }
        log.info(
            "Combined extract: %d image + %d video = %d total",
            len(img_feats), len(vid_feats), len(features),
        )
        return features, anatomies, meta

    features, anatomies, meta = extractor.extract(
        split=split,
        stream=stream,
        balance_anatomy=balance_anatomy,
        samples_per_family=samples_per_family,
        max_samples=max_samples,
        batch_size=batch_size,
        seed=seed,
    )
    meta = {**meta, "stream_types": [stream] * len(features)}
    return features, anatomies, meta


def run_embedding_viz(
    extractor: Optional[StudentEmbeddingExtractor] = None,
    *,
    features: Optional[np.ndarray] = None,
    anatomy_families: Optional[list[str]] = None,
    output_dir: str | Path = "embedding_viz",
    split: str = "train",
    stream: str = "both",
    balance_anatomy: bool = True,
    samples_per_family: int = 500,
    max_samples: Optional[int] = None,
    min_per_family: int = 1,
    plot_max_points: int = 50_000,
    batch_size: int = 32,
    video_batch_size: Optional[int] = None,
    method: str = "both",
    cache_features: bool = True,
    cache_path: Optional[str | Path] = None,
    skip_extract: bool = False,
    checkpoint_meta: Optional[dict] = None,
    seed: int = 42,
) -> dict:
    """
    End-to-end: extract (or load cache) → subsample → UMAP/t-SNE + similarity matrix.

    Parameters
    ----------
    method : ``"umap"``, ``"tsne"``, or ``"both"``
    skip_extract : load ``embeddings.npz`` from ``cache_path`` / ``output_dir``
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = Path(cache_path) if cache_path else out_dir / "embeddings.npz"

    extract_meta: dict = {}
    dataset_ids: Optional[list[str]] = None
    stream_types: Optional[list[str]] = None

    if skip_extract:
        if not npz_path.exists():
            raise FileNotFoundError(f"Cache not found: {npz_path}")
        features, anatomy_families, extract_meta = load_embedding_cache(npz_path)
        dataset_ids = extract_meta.get("dataset_ids")
        stream_types = extract_meta.get("stream_types")
    else:
        if extractor is None:
            raise ValueError("extractor is required unless skip_extract=True")
        features, anatomy_families, extract_meta = _extract_streams(
            extractor,
            split=split,
            stream=stream,
            balance_anatomy=balance_anatomy,
            samples_per_family=samples_per_family,
            max_samples=max_samples or 50_000,
            batch_size=batch_size,
            video_batch_size=video_batch_size,
            seed=seed,
        )
        dataset_ids = extract_meta.get("dataset_ids")
        stream_types = extract_meta.get("stream_types")

    if stream_types is None:
        stream_types = [extract_meta.get("stream", stream)] * len(features)

    # Balanced extraction already covers every family; only cap for plotting.
    plot_cap = max_samples if max_samples is not None else len(features)
    sub_feats, sub_anat, sub_ds, sub_sids, sub_streams, sub_stats = stratified_subsample(
        features,
        anatomy_families,
        dataset_ids=dataset_ids,
        sample_ids=extract_meta.get("sample_ids"),
        stream_types=stream_types,
        max_samples=plot_cap,
        min_per_family=min_per_family,
        seed=seed,
    )

    if cache_features and not skip_extract:
        cache_meta = {**extract_meta, "subsample_stats": sub_stats}
        if checkpoint_meta:
            cache_meta["checkpoint"] = checkpoint_meta
        save_embedding_cache(
            npz_path, features, anatomy_families,
            dataset_ids=extract_meta.get("dataset_ids"),
            sample_ids=extract_meta.get("sample_ids"),
            stream_types=stream_types,
            extra=cache_meta,
        )

    dr_cap = min(len(sub_feats), plot_max_points)
    if len(sub_feats) > plot_max_points:
        log.info(
            "UMAP/t-SNE will use %d / %d points (plot_max_points=%d)",
            dr_cap, len(sub_feats), plot_max_points,
        )

    methods = ["umap", "tsne"] if method == "both" else [method]
    figures: dict[str, str] = {}

    for m in methods:
        fname = f"{m}_anatomy.png"
        feat_viz.plot_anatomy_modality_embedding(
            sub_feats, sub_anat, sub_streams or ["image"] * len(sub_anat),
            method=m,
            max_points=dr_cap,
            save_path=str(out_dir / fname),
            seed=seed,
        )
        figures[m] = str(out_dir / fname)
        log.info("Wrote %s", out_dir / fname)

    sim_path = out_dir / "anatomy_similarity_matrix.png"
    feat_viz.plot_similarity_matrix(sub_feats, sub_anat, save_path=str(sim_path))
    figures["similarity_matrix"] = str(sim_path)

    summary = {
        "output_dir": str(out_dir),
        "split": split,
        "stream": stream,
        "balance_anatomy": extract_meta.get("balance_anatomy", balance_anatomy),
        "samples_per_family": extract_meta.get("samples_per_family", samples_per_family),
        "n_families": extract_meta.get("n_families", len(sub_stats["per_anatomy"])),
        "plot_max_points": dr_cap,
        "embedding_space": extract_meta.get("embedding_space", "unknown"),
        "embed_dim": int(sub_feats.shape[1]),
        "n_extracted": extract_meta.get("n_extracted", len(features)),
        "n_cached": len(features),
        "n_plotted": len(sub_feats),
        "per_anatomy": sub_stats["per_anatomy"],
        "per_stream": sub_stats.get("per_stream"),
        "dropped_families": sub_stats["dropped_families"],
        "methods": methods,
        "figures": figures,
        "cache_path": str(npz_path) if cache_features else None,
    }
    if checkpoint_meta:
        summary["checkpoint"] = checkpoint_meta

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info("Summary → %s", summary_path)
    return summary
