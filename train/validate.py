"""
validate.py  ·  Validation and linear probe evaluation for Ultatron
===============================================================

Usage:
    # Run full validation on a checkpoint
    python validate.py --config configs/data_config.yaml \\
                       --checkpoint $SCRATCH/checkpoints/current_run/phase3_end.pt

    # Anatomy-stratified linear probe only
    python validate.py --config configs/data_config.yaml \\
                       --checkpoint ... --mode linear_probe

    # Compute val loss only (fast, run during training)
    python validate.py --config configs/data_config.yaml \\
                       --checkpoint ... --mode val_loss

Modes
-----
  val_loss       : forward pass on val set; reports image CLS cosine loss + 7B alignment
  linear_probe   : train a linear head on frozen features; report AUC per anatomy family
  full           : val_loss + linear_probe (default)

Linear probe protocol
---------------------
  1. Extract frozen teacher CLS tokens from the val set (one pass, no grad).
  2. Train a LogisticRegression (scikit-learn) on the labelled subset.
  3. Report AUC per anatomy family and macro-average.
  
Output
------
  Prints a JSON summary to stdout.
  Writes results/validation_step_{step}.json to the log directory.
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

sys.path.insert(0, str(Path(__file__).parent))

from image_branch import ImageBranch, build_image_branch
from video_branch import VideoBranch, build_video_branch
from train import (
    load_config, build_datamodule, cosine_loss, CrossBranchDistillation,
    PrototypeHead, to_device,
)
from cscs_paths import CSCSConfig
from training_integration import _get_padding_mask

log = logging.getLogger(__name__)


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

        total_cls += cosine_loss(s_out["cls"], t_out["cls"]).item()

        if img_branch.teacher7b is not None:
            t7b = img_branch.forward_teacher7b(batch["global_crops"][:, 1])
            if t7b is not None:
                proj = img_branch.proj_7b
                t_proj = proj(t7b["cls"].float().to(device))
                total_7b += cosine_loss(s_out["cls"], t_proj).item()

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
    parser.add_argument("--config",     default="configs/data_config.yaml")
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
    cfg      = load_config(args.config)
    cscs     = CSCSConfig.from_env()
    hf_cache = str(cscs.store_path("hf_cache"))

    log.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    step = ckpt.get("global_step", 0)
    log.info(f"  Step: {step}")

    # Build models
    img_branch = build_image_branch(
        load_7b_teacher=not args.no_7b,
        dtype=torch.bfloat16, device=device, hf_cache_dir=hf_cache
    )
    vid_branch = build_video_branch(
        dtype=torch.bfloat16, device=device, hf_cache_dir=hf_cache
    )

    # Load weights
    img_branch.student.vit.load_state_dict(ckpt["img_student"])
    img_branch.teacher.vit.load_state_dict(ckpt["img_teacher"])
    vid_branch.student.model.load_state_dict(ckpt["vid_student"])
    img_branch = img_branch.to(device)

    dm = build_datamodule(cfg, cscs)
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
