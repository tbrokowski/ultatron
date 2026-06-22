"""
finetune/segmentation_viz.py  ·  Shared GT vs pred segmentation figures
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from finetune.base import FinetuneExperiment

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

_EXPERIMENT_SLUGS = {
    "camus": "camus",
    "camus_lv_segmentation": "camus",
    "busi": "busi",
    "busi_seg": "busi",
    "busi_multitask": "busi",
    "busi_breast_segmentation": "busi",
    "tn3k": "tn3k",
    "tn3k_thyroid_segmentation": "tn3k",
    "busbra": "busbra",
    "busbra_breast_segmentation": "busbra",
}


def dataset_slug_for_experiment(exp: "FinetuneExperiment") -> str:
    name = exp.EXPERIMENT_NAME.lower()
    if name in _EXPERIMENT_SLUGS:
        return _EXPERIMENT_SLUGS[name]
    # Fall back to the experiment name with common suffixes stripped.
    for suffix in ("_breast_segmentation", "_thyroid_segmentation", "_lv_segmentation"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def deterministic_viz_index(n_samples: int, dataset_slug: str) -> int:
    if n_samples <= 0:
        raise ValueError(f"Cannot pick viz sample from empty dataset ({dataset_slug})")
    seed = int(hashlib.md5(dataset_slug.encode()).hexdigest(), 16) % (2**31)
    rng = np.random.RandomState(seed)
    return int(rng.randint(0, n_samples))


def _sample_to_batch(sample: dict) -> dict:
    batch: dict = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        elif isinstance(value, (str, int, float, bool)):
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


@torch.no_grad()
def collect_seg_sample(
    exp: "FinetuneExperiment",
    split: str = "test",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return (image, gt_mask, pred_mask, sample_id) for one deterministic test sample."""
    from models.heads.finetune_seg import forward_seg_head

    loader = exp.build_dataloader(split)
    dataset = loader.dataset
    slug = dataset_slug_for_experiment(exp)
    idx = deterministic_viz_index(len(dataset), slug)
    sample = dataset[idx]
    batch = _sample_to_batch(sample)

    device = exp.device
    batch = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    pmask = batch.get("padding_mask")
    if pmask is not None and isinstance(pmask, torch.Tensor):
        feats = exp.encoder.encode_image(batch["image"], padding_mask=pmask)
        logits = forward_seg_head(exp.head, feats, padding_mask=pmask)
    else:
        feats = exp.encoder.encode_image(batch["image"])
        logits = forward_seg_head(exp.head, feats)

    target = batch["mask"]
    pred = F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    pred_np = (torch.sigmoid(pred) > 0.5).cpu().numpy()[0, 0]
    gt_np = (target > 0.5).cpu().numpy()[0, 0]
    img_np = (batch["image"].cpu().permute(0, 2, 3, 1).numpy()[0] * 255).astype(np.uint8)

    sample_id = batch.get("sample_id", [str(idx)])[0]
    if not isinstance(sample_id, str):
        sample_id = str(sample_id)

    log.info(
        "[%s] Segmentation viz sample: dataset=%s index=%d sample_id=%s",
        exp.EXPERIMENT_NAME,
        slug,
        idx,
        sample_id,
    )
    return img_np, gt_np, pred_np, sample_id


def save_segmentation_figure(exp: "FinetuneExperiment") -> Path | None:
    """Save GT vs pred overlay to {run_dir}/figures/{model}_segmentation.png."""
    if exp.encoder is None:
        log.warning("[%s] No encoder — skipping segmentation figure", exp.EXPERIMENT_NAME)
        return None

    try:
        from viz.core import save_figure
        from viz.segmentation import plot_segmentation_overlay
    except ImportError:
        log.warning("viz module not available — skipping segmentation figure")
        return None

    slug = dataset_slug_for_experiment(exp)
    model_name = f"{exp.encoder.name}_{exp.cfg.head_type}"
    out_dir = exp.output_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_segmentation.png"

    image, gt_mask, pred_mask, sample_id = collect_seg_sample(exp)
    title = f"{slug.upper()} — {model_name} — {sample_id}"
    fig = plot_segmentation_overlay(
        image,
        pred_mask,
        gt_mask=gt_mask,
        title=title,
    )
    save_figure(fig, out_path)
    log.info("[%s] Segmentation figure → %s", exp.EXPERIMENT_NAME, out_path)
    return out_path
