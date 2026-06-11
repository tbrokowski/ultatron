"""
eval/linear_probe.py  ·  Anatomy-stratified linear probe
=============================================================

The primary SSL fitness signal.

Protocol
--------
1. Extract frozen teacher CLS tokens from train and val sets in one pass
   (no gradient, no augmentation).
2. Train a LogisticRegression (sklearn) on the labelled training subset.
3. Report AUC per anatomy family and macro-average.
4. Write results to a JSON file

This is a formalisation of the linear probe code in scripts/validate.py

Usage
-----
    from eval.linear_probe import LinearProbe

    probe = LinearProbe(img_branch, dm, device)
    results = probe.run(max_train_samples=100_000, max_val_samples=50_000)
    probe.save(results, output_path="results/step_100000.json")

CLI:
    python -m eval.linear_probe \\
        --config configs/data/data_config.yaml \\
        --checkpoint checkpoints/phase3_end.pt \\
        --output results/linear_probe.json

Output JSON schema
------------------
{
  "step":          int,
  "auc_macro":     float,
  "per_anatomy":   {family: auc},
  "n_families":    int,
  "n_train":       int,
  "n_val":         int,
  "elapsed_sec":   float
}
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from eval.metrics import stratified_auc

log = logging.getLogger(__name__)


class LinearProbe:
    """
    Anatomy-stratified linear probe on frozen teacher CLS tokens.

    Parameters
    ----------
    img_branch : ImageBranch  (teacher is used; student is ignored)
    dm         : USFoundationDataModule  (must be setup() already)
    device     : str
    """

    def __init__(self, img_branch, dm, device: str = "cuda"):
        self.img_branch = img_branch
        self.dm         = dm
        self.device     = device

    # ── Feature extraction ────────────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(
        self,
        split: str,            # "train" or "val"
        max_samples: int = 50_000,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        One-pass frozen feature extraction.

        Returns
        -------
        features  : (N, D) float32
        labels    : (N,)   int32   (-1 = unlabelled)
        anatomies : list of N anatomy family strings
        """
        self.img_branch.teacher.eval()

        loader = (
            self.dm.val_loader(stream="image")
            if split == "val"
            else self.dm.image_loader()
        )

        all_feats     = []
        all_labels    = []
        all_anatomies = []
        n_seen        = 0

        for batch in loader:
            if n_seen >= max_samples:
                break

            # Move pixel values to device
            crops   = batch["global_crops"][:, 0].to(self.device, non_blocking=True)
            pmask   = batch.get("global_pmasks")
            if pmask is not None:
                pmask = pmask[:, 0].to(self.device, non_blocking=True)

            out = self.img_branch.forward_teacher(crops, padding_mask=pmask)
            cls = out["cls"].float().cpu().numpy()

            all_feats.append(cls)
            all_labels.extend(batch["cls_labels"].cpu().tolist())
            all_anatomies.extend(batch.get("anatomy_families", ["unknown"] * cls.shape[0]))
            n_seen += cls.shape[0]

        features  = np.concatenate(all_feats, axis=0)[:max_samples]
        labels    = np.array(all_labels[:max_samples], dtype=np.int32)
        anatomies = all_anatomies[:max_samples]

        log.info(f"Extracted {len(features)} {split} features, D={features.shape[1]}")
        return features, labels, anatomies

    # ── Linear probe ──────────────────────────────────────────────────────────

    def run(
        self,
        max_train_samples: int = 100_000,
        max_val_samples:   int = 50_000,
        C: float = 0.316,           # regularisation strength (10^-0.5)
        max_iter: int = 1_000,
        min_samples_per_family: int = 5,
    ) -> dict:
        """
        Run the full linear probe protocol.

        Returns a results dict compatible with the output JSON schema.
        """
        t0 = time.time()

        log.info("Extracting train features ...")
        tr_feat, tr_lab, tr_anat = self.extract_features("train", max_train_samples)

        log.info("Extracting val features ...")
        val_feat, val_lab, val_anat = self.extract_features("val", max_val_samples)

        # Filter to labelled samples for fitting
        tr_mask  = tr_lab  >= 0
        val_mask = val_lab >= 0

        if tr_mask.sum() < 10 or val_mask.sum() < 10:
            log.warning("Insufficient labelled samples for linear probe.")
            return {
                "auc_macro": float("nan"),
                "per_anatomy": {},
                "n_families": 0,
                "n_train": int(tr_mask.sum()),
                "n_val":   int(val_mask.sum()),
                "note": "insufficient labelled samples",
            }

        X_tr,  y_tr  = tr_feat[tr_mask],  tr_lab[tr_mask]
        X_val, y_val = val_feat[val_mask], val_lab[val_mask]
        anat_val     = [a for a, m in zip(val_anat, val_mask) if m]

        # Normalise features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # Fit logistic regression
        log.info(f"Fitting LogisticRegression on {len(X_tr)} samples ...")
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            C=C, max_iter=max_iter,
            solver="lbfgs", multi_class="multinomial",
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)

        proba = clf.predict_proba(X_val)   # (N_val, C)

        # Anatomy-stratified AUC
        results = stratified_auc(y_val, proba, anat_val, min_samples_per_family)
        results["n_train"]     = int(tr_mask.sum())
        results["n_val"]       = int(val_mask.sum())
        results["elapsed_sec"] = round(time.time() - t0, 1)

        log.info(f"Linear probe AUC macro: {results['auc_macro']:.4f}  "
                 f"({results['n_families']} anatomy families)")
        for fam, auc in sorted(results["per_anatomy"].items()):
            log.info(f"  {fam:25s}: {auc:.4f}")

        return results

    # ── Output ────────────────────────────────────────────────────────────────

    @staticmethod
    def save(results: dict, output_path: str | Path, step: int = 0):
        """Write results to a JSON file."""
        out  = {"step": step, **results}
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        log.info(f"Linear probe results → {path}")
        return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    """Load YAML with _base_ inheritance (matches scripts/train.py logic)."""
    import yaml
    repo = Path(__file__).resolve().parent.parent
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = repo / cfg.pop("_base_")
        with open(base_path) as f:
            base = yaml.safe_load(f)
        def _deep_merge(base: dict, override: dict) -> dict:
            out = dict(base)
            for k, v in override.items():
                if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out
        cfg = _deep_merge(base, cfg)
    return cfg


def _main():
    parser = argparse.ArgumentParser(
        description="Ultatron anatomy-stratified linear probe"
    )
    parser.add_argument("--config",      required=True,
                        help="Experiment YAML (e.g. configs/run1/minimal_run1.yaml)")
    parser.add_argument("--checkpoint",  required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--manifest",    default=None,
                        help="Override manifest path from config (JSONL file)")
    parser.add_argument("--output",      default=None,
                        help="Output JSON path (default: auto-generated alongside ckpt)")
    parser.add_argument("--max-train",   type=int, default=100_000)
    parser.add_argument("--max-val",     type=int, default=50_000)
    parser.add_argument("--no-7b",       action="store_true",
                        help="Skip DINOv3-7B frozen teacher (faster, less memory)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    import yaml
    from models import ModelConfig, build_image_branch
    from data.infra.cscs_paths import CSCSConfig
    from data.pipeline.datamodule import USFoundationDataModule
    from data.pipeline.transforms import (
        ImageSSLTransformConfig, VideoSSLTransformConfig, FreqMaskConfig,
    )

    cfg  = _load_config(args.config)
    cscs = CSCSConfig.from_env()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    step = ckpt.get("global_step", 0)
    phase_tag = ckpt.get("phase", "")
    log.info("  step=%d  phase=%s", step, phase_tag)

    # ── Build image branch ────────────────────────────────────────────────────
    hf_cache = str(cscs.store_path("hf_cache"))
    model_cfg = ModelConfig.from_dict(cfg.get("model", {}))
    model_cfg.hf_cache_dir = hf_cache
    if args.no_7b:
        model_cfg.frozen_teacher = None

    img_branch = build_image_branch(model_cfg, device=device)
    if "img_student" in ckpt:
        img_branch.student.load_state_dict(ckpt["img_student"])
        img_branch.teacher.load_state_dict(ckpt["img_teacher"])
        log.info("Loaded img_student / img_teacher weights.")
    else:
        log.warning("Checkpoint has no img_student key — using randomly initialised weights.")

    # ── Build DataModule ──────────────────────────────────────────────────────
    total_steps = cfg.get("curriculum", {}).get("total_training_steps", 300_000)

    img_raw  = dict(cfg["transforms"]["image"])
    vid_raw  = dict(cfg["transforms"]["video"])
    img_freq = img_raw.pop("freq_mask", {})
    vid_freq = vid_raw.pop("freq_mask", {})
    img_tcfg = ImageSSLTransformConfig(
        **img_raw,
        freq_mask=FreqMaskConfig(**img_freq) if img_freq else FreqMaskConfig(),
    )
    vid_tcfg = VideoSSLTransformConfig(
        **vid_raw,
        freq_mask=FreqMaskConfig(**vid_freq) if vid_freq else FreqMaskConfig(),
    )

    if args.manifest:
        manifest_path = args.manifest
    else:
        _mcfg = Path(cfg["manifest"]["path"])
        repo  = Path(__file__).resolve().parent.parent
        if not _mcfg.is_absolute():
            _mcfg = repo / _mcfg
        manifest_path = str(_mcfg if _mcfg.exists() else cscs.manifest_path(_mcfg.name))

    root_remap = cfg["manifest"].get("root_remap")
    if root_remap is None:
        root_remap = cscs.remap_dict()

    dm = USFoundationDataModule(
        manifest_path           = manifest_path,
        image_batch_size        = cfg["loaders"]["image_batch_size"],
        video_batch_size        = cfg["loaders"]["video_batch_size"],
        num_workers             = cfg["loaders"].get("num_workers", 4),
        pin_memory              = cfg["loaders"].get("pin_memory", True),
        patch_size              = cfg["transforms"]["patch_size"],
        total_training_steps    = total_steps,
        image_samples_per_epoch = cfg["curriculum"]["image_samples_per_epoch"],
        video_samples_per_epoch = cfg["curriculum"]["video_samples_per_epoch"],
        anatomy_weights         = cfg.get("anatomy_weights", {}),
        root_remap              = root_remap,
        image_cfg               = img_tcfg,
        video_cfg               = vid_tcfg,
    )
    dm.setup()

    # ── Run linear probe ──────────────────────────────────────────────────────
    probe   = LinearProbe(img_branch, dm, device=device)
    results = probe.run(
        max_train_samples = args.max_train,
        max_val_samples   = args.max_val,
    )
    results["phase"] = phase_tag

    # ── Output path ───────────────────────────────────────────────────────────
    output = args.output
    if output is None:
        out_dir = ckpt_path.parent / "results"
        label   = phase_tag if phase_tag else f"step{step:08d}"
        output  = out_dir / f"linear_probe_{label}.json"

    LinearProbe.save(results, output, step=step)
    print(f"\nLinear probe AUC macro: {results.get('auc_macro', float('nan')):.4f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    _main()
