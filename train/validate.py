"""
validate.py  ·  Validation and linear probe evaluation for Ultatron
===============================================================

Usage (from repo root):
    python train/validate.py --config configs/experiments/run1.yaml \\
                             --checkpoint /path/to/latest.pt

    # Modes: val_loss | linear_probe | full (default)
    python train/validate.py --config configs/experiments/run1.yaml \\
                             --checkpoint ... --mode linear_probe

Modes
-----
  val_loss       : forward pass on val set; reports image CLS cosine loss
  linear_probe   : train a linear head on frozen features; report AUC per anatomy family
  full           : val_loss + linear_probe (default)

Linear probe protocol
---------------------
  1. Extract frozen teacher CLS tokens (one pass, no grad).
  2. Train a LogisticRegression (scikit-learn) on the labelled subset.
  3. Report AUC per anatomy family and macro-average.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Repo root for imports
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from models import ModelConfig, build_image_branch, build_video_branch
from models.branches.image_branch import ImageBranch
from models.branches.shared import CrossBranchDistillation
from data.pipeline.datamodule import USFoundationDataModule
from data.pipeline.transforms import (
    ImageSSLTransformConfig, VideoSSLTransformConfig, FreqMaskConfig,
)
from data.infra.cscs_paths import CSCSConfig

log = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    """Load YAML config with _base_ inheritance (same as scripts/train.py)."""
    path = Path(path)
    if not path.is_absolute():
        path = _repo_root / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    bases = cfg.pop("_base_", [])
    merged = {}
    for base_path in bases:
        base_cfg = _load_config(str(_repo_root / base_path))
        _deep_merge(merged, base_cfg)
    _deep_merge(merged, cfg)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _get_padding_mask(batch: dict, crop_idx: int = 0) -> Optional[torch.Tensor]:
    """Extract (B, ph, pw) padding mask for a specific global crop index."""
    pm = batch.get("global_pmasks")
    return pm[:, crop_idx] if pm is not None else None


def _cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Symmetric cosine distance: mean(1 - cos(a,b))."""
    a_n = F.normalize(a.float(), dim=-1)
    b_n = F.normalize(b.float(), dim=-1)
    return (1 - (a_n * b_n).sum(-1)).mean()


# ── Val loss ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_val_loss(
    img_branch: ImageBranch,
    dm,
    device: str,
    max_batches: int = 200,
) -> dict:
    """
    Compute mean CLS cosine loss on the validation set.
    Returns dict with loss_cls, loss_7b_align.
    """
    img_branch.student.eval()
    img_branch.teacher.eval()

    val_loader = dm.val_loader(stream="image")
    total_cls = 0.0
    total_7b  = 0.0
    n_batches = 0

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        batch = to_device(batch, device)

        t_pmask = _get_padding_mask(batch, 1)
        s_pmask = _get_padding_mask(batch, 0)

        t_out = img_branch.forward_teacher(
            batch["global_crops"][:, 1], padding_mask=t_pmask
        )
        s_out = img_branch.forward_student(
            batch["global_crops"][:, 0], padding_mask=s_pmask,
        )

        total_cls += _cosine_loss(s_out["cls"], t_out["cls"]).item()
        n_batches += 1

    img_branch.student.train()

    return {
        "val_loss_cls":      total_cls / max(n_batches, 1),
        "val_loss_7b_align": total_7b  / max(n_batches, 1),
        "n_val_batches":     n_batches,
    }


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    img_branch: ImageBranch,
    dm,
    device: str,
    split: str = "val",
    max_samples: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract frozen teacher CLS tokens from val (or train) set.

    Returns
    -------
    features       : (N, D) float32 numpy array
    labels         : (N,) int numpy array  (-1 = unlabelled)
    anatomy_list   : list[str] length N
    """
    img_branch.teacher.eval()

    loader = dm.val_loader(stream="image") if split == "val" else dm.image_loader()
    all_feats    = []
    all_labels   = []
    all_anatomies = []
    n_samples    = 0

    for batch in loader:
        if n_samples >= max_samples:
            break
        batch = to_device(batch, device)
        t_pmask = _get_padding_mask(batch, 0)
        out = img_branch.forward_teacher(
            batch["global_crops"][:, 0], padding_mask=t_pmask
        )
        cls = out["cls"].float().cpu().numpy()
        all_feats.append(cls)
        all_labels.extend(batch["cls_labels"].cpu().tolist())
        all_anatomies.extend(batch["anatomy_families"])
        n_samples += cls.shape[0]

    features = np.concatenate(all_feats, axis=0)[:max_samples]
    labels   = np.array(all_labels[:max_samples])
    anatomies = all_anatomies[:max_samples]

    return features, labels, anatomies


# ── Linear probe ──────────────────────────────────────────────────────────────

def linear_probe_eval(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_anatomies: list[str],
    val_features: np.ndarray,
    val_labels: np.ndarray,
    val_anatomies: list[str],
) -> dict:
    """
    Train a logistic regression on labelled train features.
    Report AUC per anatomy family and macro-average.

    Requires scikit-learn.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        from sklearn.multiclass import OneVsRestClassifier
    except ImportError:
        log.warning("scikit-learn not available — skipping linear probe.")
        return {"linear_probe": "skipped (no sklearn)"}

    # Filter to labelled samples
    train_mask = train_labels >= 0
    val_mask   = val_labels   >= 0

    if train_mask.sum() < 10 or val_mask.sum() < 10:
        return {"linear_probe": "skipped (insufficient labelled samples)"}

    X_tr, y_tr = train_features[train_mask], train_labels[train_mask]
    X_val, y_val = val_features[val_mask], val_labels[val_mask]
    anatomy_val = [a for a, m in zip(val_anatomies, val_mask) if m]

    # Normalise
    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, C=0.316, solver="lbfgs",
                              multi_class="multinomial", n_jobs=-1)
    clf.fit(X_tr, y_tr)

    unique_classes = sorted(set(y_val.tolist()))
    if len(unique_classes) < 2:
        return {"linear_probe": "skipped (single class in val)"}

    proba = clf.predict_proba(X_val)

    # Overall AUC
    try:
        if len(unique_classes) == 2:
            auc_overall = roc_auc_score(y_val, proba[:, 1])
        else:
            auc_overall = roc_auc_score(y_val, proba, multi_class="ovr",
                                         average="macro")
    except Exception:
        auc_overall = float("nan")

    # Per-anatomy AUC
    auc_per_anatomy = {}
    for fam in set(anatomy_val):
        idx = [i for i, a in enumerate(anatomy_val) if a == fam]
        if len(idx) < 5:
            continue
        y_fam  = y_val[idx]
        p_fam  = proba[idx]
        try:
            unique_fam = sorted(set(y_fam.tolist()))
            if len(unique_fam) < 2: continue
            if len(unique_fam) == 2:
                a = roc_auc_score(y_fam, p_fam[:, unique_classes.index(unique_fam[1])])
            else:
                a = roc_auc_score(y_fam, p_fam[:, [unique_classes.index(c) for c in unique_fam]],
                                  multi_class="ovr", average="macro")
            auc_per_anatomy[fam] = round(a, 4)
        except Exception:
            pass

    return {
        "auc_macro":       round(auc_overall, 4),
        "auc_per_anatomy": auc_per_anatomy,
        "n_train":         int(train_mask.sum()),
        "n_val":           int(val_mask.sum()),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/experiments/run1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode",       default="full",
                        choices=["val_loss", "linear_probe", "full"])
    parser.add_argument("--no-7b",      action="store_true")
    parser.add_argument("--max-samples", type=int, default=50_000)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    cfg      = _load_config(args.config)
    cscs     = CSCSConfig.from_env()
    hf_cache = str(cscs.store_path("hf_cache"))

    log.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    step = ckpt.get("global_step", 0)
    log.info(f"  Step: {step}")

    model_cfg = ModelConfig.from_dict(cfg.get("model", {}))
    model_cfg.hf_cache_dir = hf_cache
    if args.no_7b:
        model_cfg.frozen_teacher = None
    img_branch = build_image_branch(model_cfg, device=device)
    vid_branch = build_video_branch(model_cfg, device=device)
    img_branch = img_branch.to(device)
    vid_branch = vid_branch.to(device)
    def _strip_ddp(sd: dict) -> dict:
        """Strip 'module.' prefix inserted by DDP wrapping, if present."""
        return {(k[len("module."):] if k.startswith("module.") else k): v
                for k, v in sd.items()}

    img_branch.student.load_state_dict(_strip_ddp(ckpt["img_student"]), strict=True)
    img_branch.teacher.load_state_dict(_strip_ddp(ckpt["img_teacher"]), strict=True)
    vid_branch.student.load_state_dict(_strip_ddp(ckpt["vid_student"]), strict=True)
    vid_branch.teacher.load_state_dict(_strip_ddp(ckpt["vid_teacher"]), strict=True)

    _mp = Path(cfg["manifest"]["path"])
    if not _mp.is_absolute():
        _mp = _repo_root / _mp
    manifest_path = str(_mp) if _mp.exists() else str(cscs.manifest_path(_mp.name))
    root_remap = cfg["manifest"].get("root_remap")
    if root_remap is None:
        root_remap = cscs.remap_dict()
    total_steps = cfg.get("curriculum", {}).get("total_training_steps", 20_000)
    img_raw = dict(cfg["transforms"]["image"])
    vid_raw = dict(cfg["transforms"]["video"])
    img_freq = img_raw.pop("freq_mask", {})
    vid_freq = vid_raw.pop("freq_mask", {})
    img_tcfg = ImageSSLTransformConfig(
        **img_raw, freq_mask=FreqMaskConfig(**img_freq) if img_freq else FreqMaskConfig()
    )
    vid_tcfg = VideoSSLTransformConfig(
        **vid_raw, freq_mask=FreqMaskConfig(**vid_freq) if vid_freq else FreqMaskConfig()
    )
    dm = USFoundationDataModule(
        manifest_path=manifest_path,
        image_batch_size=cfg["loaders"]["image_batch_size"],
        video_batch_size=cfg["loaders"]["video_batch_size"],
        num_workers=cfg["loaders"]["num_workers"],
        pin_memory=cfg["loaders"]["pin_memory"],
        patch_size=cfg["transforms"]["patch_size"],
        total_training_steps=total_steps,
        image_samples_per_epoch=cfg["curriculum"]["image_samples_per_epoch"],
        video_samples_per_epoch=cfg["curriculum"]["video_samples_per_epoch"],
        anatomy_weights=cfg.get("anatomy_weights", {}),
        root_remap=root_remap,
        image_cfg=img_tcfg,
        video_cfg=vid_tcfg,
    )
    dm.setup()

    results = {"step": step, "checkpoint": args.checkpoint}

    # ── Val loss ───────────────────────────────────────────────────────────────
    if args.mode in ("val_loss", "full"):
        log.info("Computing validation loss ...")
        val_metrics = compute_val_loss(img_branch, dm, device)
        results.update(val_metrics)
        log.info(f"  Val CLS loss: {val_metrics['val_loss_cls']:.4f}")

    # ── Linear probe ──────────────────────────────────────────────────────────
    if args.mode in ("linear_probe", "full"):
        log.info("Extracting train features for linear probe ...")
        tr_feat, tr_lab, tr_anat = extract_features(
            img_branch, dm, device, split="train",
            max_samples=min(args.max_samples, 100_000)
        )
        log.info("Extracting val features ...")
        val_feat, val_lab, val_anat = extract_features(
            img_branch, dm, device, split="val",
            max_samples=args.max_samples
        )
        log.info("Running linear probe ...")
        probe_results = linear_probe_eval(
            tr_feat, tr_lab, tr_anat,
            val_feat, val_lab, val_anat,
        )
        results["linear_probe"] = probe_results
        log.info(f"  Macro AUC: {probe_results.get('auc_macro', 'N/A')}")
        if "auc_per_anatomy" in probe_results:
            for fam, auc in sorted(probe_results["auc_per_anatomy"].items()):
                log.info(f"    {fam:25s}: {auc:.4f}")

    # ── Output ─────────────────────────────────────────────────────────────────
    output_str = json.dumps(results, indent=2)
    print(output_str)

    if args.output:
        out_path = Path(args.output)
    else:
        log_dir  = cscs.scratch_path("logs") / "current_run" / "results"
        out_path = log_dir / f"validation_step_{step:08d}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_str)
    log.info(f"Results written → {out_path}")


if __name__ == "__main__":
    main()
