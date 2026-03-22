#!/usr/bin/env python3
"""
scripts/finetune.py  ·  Ultatron Phase 4 — downstream head fine-tuning
=======================================================================

Loads a pre-trained checkpoint (phase3_end.pt or latest.pt), freezes the
teacher backbone, then trains and evaluates task heads for:

    1. BUSI       — tumour segmentation (Dice) + 3-class classification (AUC)
    2. EchoNet    — ejection fraction regression (MAE, R²)
    3. LUS-patient — TB / Pneumonia / COVID (patient-level AUC)

Experiments run sequentially on a single GPU (rank 0 only).

Usage
-----
    python scripts/finetune.py \\
        --checkpoint /capstor/scratch/cscs/tbrokowski/ultrasound/checkpoints/run1/phase3_end.pt \\
        --train-config configs/experiments/run1.yaml \\
        --output-dir results/run1_finetune/

    # Evaluate only (skip training, load saved heads):
    python scripts/finetune.py --checkpoint ... --eval-only

    # Override individual dataset roots:
    python scripts/finetune.py --checkpoint ... --echonet-root /path/to/EchoNet-Dynamic
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml


log = logging.getLogger(__name__)


def _load_config(path: str) -> dict:
    """Load YAML with _base_ inheritance."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    with open(p) as f:
        cfg = yaml.safe_load(f)
    bases = cfg.pop("_base_", [])
    merged: dict = {}
    for base_path in bases:
        _deep_merge(merged, _load_config(base_path))
    _deep_merge(merged, cfg)
    return merged


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _load_finetune_cfg(yaml_path: str) -> dict:
    """Load a finetune YAML (no _base_ inheritance needed)."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Ultatron Phase 4: downstream head fine-tuning"
    )
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to SSL pre-training checkpoint (.pt)")
    parser.add_argument("--train-config", default="configs/experiments/run1.yaml",
                        help="Training YAML config (for backbone architecture)")
    parser.add_argument("--output-dir",   default=None,
                        help="Output directory for results and head checkpoints. "
                             "Defaults to <checkpoint_dir>/finetune/")
    parser.add_argument("--device",       default="cuda")

    # Dataset root overrides (fall back to values in finetune YAML configs)
    parser.add_argument("--busi-root",    default=None,
                        help="Override BUSI dataset root")
    parser.add_argument("--echonet-root", default=None,
                        help="Override EchoNet-Dynamic dataset root")
    parser.add_argument("--benin-root",   default=None,
                        help="Override Benin-LUS dataset root")
    parser.add_argument("--rsa-root",     default=None,
                        help="Override RSA-LUS dataset root")

    # Experiment selection
    parser.add_argument("--experiments",  nargs="+",
                        choices=["busi", "echonet", "lus"],
                        default=["busi", "echonet", "lus"],
                        help="Which experiments to run (default: all)")
    parser.add_argument("--eval-only",    action="store_true",
                        help="Skip training, load saved best_head.pt and evaluate")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    repo = _find_repo_root()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "finetune"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load architecture config ──────────────────────────────────────────────
    train_cfg = _load_config(args.train_config)
    model_cfg_dict = train_cfg.get("model", {})

    # Resolve HF cache (prefer scratch)
    try:
        from data.infra.cscs_paths import CSCSConfig
        cscs = CSCSConfig.from_env()
        _scratch_hf = cscs.scratch_path("hf_cache")
        _store_hf   = cscs.store_path("hf_cache")
        hf_cache = str(_scratch_hf if _scratch_hf.exists() else _store_hf)
    except Exception:
        hf_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    model_cfg_dict["hf_cache_dir"] = hf_cache
    model_cfg_dict["frozen_teacher"] = None   # not needed for finetune

    # ── Build backbone branches (single GPU, no DDP) ──────────────────────────
    from models import ModelConfig, build_image_branch, build_video_branch
    model_cfg = ModelConfig.from_dict(model_cfg_dict)

    log.info(f"Building image backbone: {model_cfg.image_backbone}")
    img_branch = build_image_branch(model_cfg, device=args.device)

    log.info(f"Building video backbone: {model_cfg.video_backbone}")
    vid_branch = build_video_branch(model_cfg, device=args.device)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    step  = ckpt.get("global_step", "?")
    phase = ckpt.get("current_phase", "?")
    log.info(f"  → step={step}  phase={phase}")

    img_branch.teacher.load_state_dict(ckpt["img_teacher"], strict=True)
    vid_branch.teacher.load_state_dict(ckpt["vid_teacher"], strict=True)
    log.info("Backbone weights loaded.")

    # ── Load finetune YAML configs ────────────────────────────────────────────
    ft_cfg_dir = repo / "configs" / "finetune"

    busi_raw     = _load_finetune_cfg(str(ft_cfg_dir / "busi.yaml"))
    echonet_raw  = _load_finetune_cfg(str(ft_cfg_dir / "echonet.yaml"))
    lus_raw      = _load_finetune_cfg(str(ft_cfg_dir / "lus_patient.yaml"))

    # ── Resolve dataset roots (CLI overrides config YAML) ────────────────────
    busi_root    = args.busi_root    or busi_raw.get("dataset_root",    "")
    echonet_root = args.echonet_root or echonet_raw.get("dataset_root", "")
    benin_root   = args.benin_root   or lus_raw.get("dataset_root_benin", "")
    rsa_root     = args.rsa_root     or lus_raw.get("dataset_root_rsa",   "")

    # ── Run experiments ───────────────────────────────────────────────────────
    all_results: dict = {"checkpoint": str(ckpt_path), "step": step, "phase": phase}

    if "busi" in args.experiments:
        log.info("=" * 60)
        log.info("Experiment 1/3: BUSI tumour segmentation + classification")
        log.info("=" * 60)
        _run_busi(img_branch, vid_branch, busi_root, busi_raw,
                  output_dir / "busi", args.device, args.eval_only, all_results)

    if "echonet" in args.experiments:
        log.info("=" * 60)
        log.info("Experiment 2/3: EchoNet-Dynamic EF regression")
        log.info("=" * 60)
        _run_echonet(img_branch, vid_branch, echonet_root, echonet_raw,
                     output_dir / "echonet", args.device, args.eval_only, all_results)

    if "lus" in args.experiments:
        log.info("=" * 60)
        log.info("Experiment 3/3: LUS patient TB/Pneumonia/COVID")
        log.info("=" * 60)
        _run_lus(img_branch, vid_branch, benin_root, rsa_root, lus_raw,
                 output_dir / "lus_patient", args.device, args.eval_only, all_results)

    # ── Write combined results ────────────────────────────────────────────────
    results_path = output_dir / "results_summary.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    log.info(f"\nResults summary → {results_path}")
    print("\n" + "=" * 60)
    print("FINETUNE RESULTS")
    print("=" * 60)
    print(json.dumps(all_results, indent=2))


# ── Per-experiment runners ────────────────────────────────────────────────────

def _run_busi(img_branch, vid_branch, data_root, raw_cfg,
              out_dir, device, eval_only, all_results):
    from finetune.experiments.busi import BUSIFinetune
    from finetune.base import FinetuneConfig

    if not data_root or not Path(data_root).exists():
        log.warning(f"[BUSI] data_root not found: {data_root!r}. Skipping.")
        all_results["busi"] = {"skipped": True, "reason": f"data_root not found: {data_root}"}
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FinetuneConfig.from_dict(raw_cfg.get("finetune", raw_cfg))
    exp = BUSIFinetune(data_root=data_root, output_dir=str(out_dir), cfg=cfg)
    exp.setup(img_branch, device=device, vid_branch=vid_branch)

    if eval_only:
        best = out_dir / "best_head.pt"
        if best.exists():
            exp.load_head(str(best))
        else:
            log.warning("[BUSI] eval-only but best_head.pt not found — running finetune.")
            exp.run()
    else:
        exp.run()

    results = exp.evaluate("test")
    log.info(f"[BUSI] {results}")
    all_results["busi"] = results


def _run_echonet(img_branch, vid_branch, data_root, raw_cfg,
                 out_dir, device, eval_only, all_results):
    from finetune.experiments.echonet import EchoNetFinetune
    from finetune.base import FinetuneConfig

    if not data_root or not Path(data_root).exists():
        log.warning(f"[EchoNet] data_root not found: {data_root!r}. Skipping.")
        all_results["echonet"] = {"skipped": True, "reason": f"data_root not found: {data_root}"}
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FinetuneConfig.from_dict(raw_cfg.get("finetune", raw_cfg))
    exp = EchoNetFinetune(data_root=data_root, output_dir=str(out_dir), cfg=cfg)
    exp.setup(img_branch, device=device, vid_branch=vid_branch)

    if eval_only:
        best = out_dir / "best_head.pt"
        if best.exists():
            exp.load_head(str(best))
        else:
            log.warning("[EchoNet] eval-only but best_head.pt not found — running finetune.")
            exp.run()
    else:
        exp.run()

    results = exp.evaluate("test")
    log.info(f"[EchoNet] {results}")
    all_results["echonet"] = results


def _run_lus(img_branch, vid_branch, benin_root, rsa_root, raw_cfg,
             out_dir, device, eval_only, all_results):
    from finetune.experiments.lus_patient import LUSPatientFinetune
    from finetune.base import FinetuneConfig

    has_benin = benin_root and Path(benin_root).exists()
    has_rsa   = rsa_root   and Path(rsa_root).exists()
    if not has_benin and not has_rsa:
        log.warning(f"[LUS] Neither benin_root nor rsa_root found. Skipping.")
        all_results["lus_patient"] = {"skipped": True,
                                      "reason": "dataset roots not found"}
        return

    # Fall back to empty string if one is missing (dataset reads will warn + skip)
    benin_root = benin_root or ""
    rsa_root   = rsa_root   or ""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_raw = raw_cfg.get("finetune", raw_cfg)
    cfg = FinetuneConfig.from_dict(ft_raw)
    exp = LUSPatientFinetune(
        data_root_benin = benin_root,
        data_root_rsa   = rsa_root,
        output_dir      = str(out_dir),
        cfg             = cfg,
        n_frames        = ft_raw.get("n_frames", 8),
        img_size        = ft_raw.get("img_size",  224),
    )
    exp.setup(img_branch, device=device, vid_branch=vid_branch)

    if eval_only:
        best = out_dir / "best_head.pt"
        if best.exists():
            exp.load_head(str(best))
        else:
            log.warning("[LUS] eval-only but best_head.pt not found — running finetune.")
            exp.run()
    else:
        exp.run()

    results = exp.evaluate("test")
    log.info(f"[LUS] {results}")
    all_results["lus_patient"] = results


if __name__ == "__main__":
    main()
