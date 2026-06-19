#!/usr/bin/env python3
"""
scripts/finetune.py  ·  Ultatron Phase 4 — downstream head fine-tuning
=======================================================================

Loads a pre-trained checkpoint (phase3_end.pt or latest.pt), freezes the
teacher backbone, then trains and evaluates task heads for:

    1. BUSI       — tumour segmentation (Dice) + 3-class classification (AUC)
    2. EchoNet    — ejection fraction regression (MAE, R²)
    3. LUS-patient — TB binary (patient-level MIL + MLP, Benin + RSA)
    4. LUS-video    — 7 finding binary MLP heads (clip/image level, Benin + RSA)

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

    # Comparison sweep — subset of backbones:
    python scripts/finetune.py --comparison-config configs/finetune/comparison_representative.yaml \\
        --backbones biomedclip usfm echocare
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _prepare_finetune_dirs(results_dir: Path, repo: Path) -> tuple[Path, Path]:
    """Results dir (repo finetune tree) + matching Capstor checkpoint dir."""
    from finetune.paths import mirror_checkpoint_dir

    results_dir = Path(results_dir)
    ckpt_dir = mirror_checkpoint_dir(results_dir, repo)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, ckpt_dir


def _resolve_hf_cache() -> str:
    """Prefer Capstor store/scratch HF cache with models (see setup_hf_cache.sh)."""
    try:
        from data.infra.cscs_paths import configure_hf_environment
        return str(configure_hf_environment())
    except Exception:
        return os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))


def main():
    parser = argparse.ArgumentParser(
        description="Ultatron Phase 4: downstream head fine-tuning"
    )
    parser.add_argument("--checkpoint",   default=None,
                        help="Path to SSL pre-training checkpoint (.pt). "
                             "Required unless --comparison-config is set.")
    parser.add_argument("--comparison-config", default=None,
                        help="Run multi-backbone comparison sweep from YAML "
                             "(see configs/finetune/comparison.yaml)")
    parser.add_argument("--train-config", default="configs/experiments/run1.yaml",
                        help="Training YAML config (for backbone architecture)")
    parser.add_argument("--output-dir",   default=None,
                        help="Output directory for metrics, logs, and visualizations. "
                             "Task-head weights go to Capstor checkpoints/Finetune/.")
    parser.add_argument("--device",       default="cuda")

    # Dataset root overrides (fall back to values in finetune YAML configs)
    parser.add_argument("--busi-root",         default=None)
    parser.add_argument("--echonet-root",      default=None)
    parser.add_argument("--camus-root",        default=None)
    parser.add_argument("--echonet-ped-root",  default=None)
    parser.add_argument("--echonet-lvh-root",  default=None)
    parser.add_argument("--mimic-lvvol-root",  default=None)
    parser.add_argument("--cardiacudc-root",   default=None)
    parser.add_argument("--echocp-root",       default=None)
    parser.add_argument("--benin-root",        default=None)
    parser.add_argument("--rsa-root",          default=None)
    parser.add_argument("--busbra-root",       default=None)
    parser.add_argument("--tn3k-root",         default=None)
    parser.add_argument("--fetal-planes-db-root", default=None)

    _ALL_EXPERIMENTS = [
        "busi", "busi_multitask", "busbra", "echonet", "camus", "echonet_ped", "echonet_lvh",
        "mimic_lvvol", "cardiacudc", "echocp", "fetal_planes_db", "lus", "lus_video", "tn3k",
    ]
    parser.add_argument("--experiments", nargs="+", default=None,
                        choices=_ALL_EXPERIMENTS,
                        help="Which experiments to run (comparison: subset of config list; "
                             "legacy: default all)")
    parser.add_argument("--eval-only",    action="store_true",
                        help="Skip training, load saved best_head.pt and evaluate")
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Comparison mode only: run a subset of backbone keys "
                             "from the comparison config (default: all). "
                             "Example: --backbones biomedclip usfm echocare")
    parser.add_argument("--parallel-experiments", action="store_true",
                        help="Comparison mode: run one experiment per GPU in parallel "
                             "(default when multiple GPUs are visible)")
    parser.add_argument("--no-parallel-experiments", action="store_true",
                        help="Comparison mode: run experiments sequentially on one GPU")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Comparison mode: cap parallel GPU workers (default: all visible)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.comparison_config:
        run_comparison(args)
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --comparison-config is set")

    repo = _find_repo_root()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        from finetune.paths import finetune_results_root
        output_dir = finetune_results_root(repo) / "legacy"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load architecture config ──────────────────────────────────────────────
    train_cfg = _load_config(args.train_config)
    model_cfg_dict = train_cfg.get("model", {})

    hf_cache = _resolve_hf_cache()
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

    def _cfg(name: str) -> dict:
        return _load_finetune_cfg(str(ft_cfg_dir / f"{name}.yaml"))

    busi_raw        = _cfg("busi")
    echonet_raw     = _cfg("echonet")
    echonet_ped_raw = _cfg("echonet_pediatric")
    echonet_lvh_raw = _cfg("echonet_lvh")
    mimic_lvvol_raw = _cfg("mimic_lvvol")
    cardiacudc_raw  = _cfg("cardiacudc")
    echocp_raw      = _cfg("echocp")
    lus_raw         = _cfg("lus_patient")
    lus_video_raw   = _cfg("lus_video")

    # ── Resolve dataset roots (CLI overrides config YAML) ────────────────────
    busi_root        = args.busi_root        or busi_raw.get("dataset_root", "")
    echonet_root     = args.echonet_root     or echonet_raw.get("dataset_root", "")
    echonet_ped_root = args.echonet_ped_root or echonet_ped_raw.get("dataset_root", "")
    echonet_lvh_root = args.echonet_lvh_root or echonet_lvh_raw.get("dataset_root", "")
    mimic_lvvol_root = args.mimic_lvvol_root or mimic_lvvol_raw.get("dataset_root", "")
    cardiacudc_root  = args.cardiacudc_root  or cardiacudc_raw.get("dataset_root", "")
    echocp_root      = args.echocp_root      or echocp_raw.get("dataset_root", "")
    benin_root       = args.benin_root       or lus_raw.get("dataset_root_benin", "")
    rsa_root         = args.rsa_root         or lus_raw.get("dataset_root_rsa",   "")

    # ── Run experiments ───────────────────────────────────────────────────────
    all_results: dict = {"checkpoint": str(ckpt_path), "step": step, "phase": phase}
    exps = args.experiments or _ALL_EXPERIMENTS

    def _hdr(n: int, total: int, label: str):
        log.info("=" * 60)
        log.info(f"Experiment {n}/{total}: {label}")
        log.info("=" * 60)

    total = len(exps)

    if "busi" in exps:
        _hdr(exps.index("busi") + 1, total, "BUSI tumour segmentation + classification")
        _run_busi(img_branch, vid_branch, busi_root, busi_raw,
                  output_dir / "busi", args.device, args.eval_only, all_results)

    if "echonet" in exps:
        _hdr(exps.index("echonet") + 1, total, "EchoNet-Dynamic EF regression")
        _run_echonet(img_branch, vid_branch, echonet_root, echonet_raw,
                     output_dir / "echonet", args.device, args.eval_only, all_results)

    if "echonet_ped" in exps:
        _hdr(exps.index("echonet_ped") + 1, total, "EchoNet-Pediatric EF regression")
        _run_generic(
            "echonet_ped", "EchoNetPediatricFinetune",
            "finetune.experiments.echonet_pediatric",
            img_branch, vid_branch, echonet_ped_root, echonet_ped_raw,
            output_dir / "echonet_pediatric", args.device, args.eval_only, all_results,
        )

    if "echonet_lvh" in exps:
        _hdr(exps.index("echonet_lvh") + 1, total, "EchoNet-LVH wall thickness regression")
        _run_generic(
            "echonet_lvh", "EchoNetLVHFinetune",
            "finetune.experiments.echonet_lvh",
            img_branch, vid_branch, echonet_lvh_root, echonet_lvh_raw,
            output_dir / "echonet_lvh", args.device, args.eval_only, all_results,
        )

    if "mimic_lvvol" in exps:
        _hdr(exps.index("mimic_lvvol") + 1, total, "MIMIC-LVVol-A4C LVEF regression")
        _run_generic(
            "mimic_lvvol", "MIMICLVVolFinetune",
            "finetune.experiments.mimic_lvvol",
            img_branch, vid_branch, mimic_lvvol_root, mimic_lvvol_raw,
            output_dir / "mimic_lvvol", args.device, args.eval_only, all_results,
        )

    if "cardiacudc" in exps:
        _hdr(exps.index("cardiacudc") + 1, total, "CardiacUDC normal/disease classification")
        _run_generic(
            "cardiacudc", "CardiacUDCFinetune",
            "finetune.experiments.cardiacudc",
            img_branch, vid_branch, cardiacudc_root, cardiacudc_raw,
            output_dir / "cardiacudc", args.device, args.eval_only, all_results,
        )

    if "echocp" in exps:
        _hdr(exps.index("echocp") + 1, total, "EchoCP PFO classification")
        _run_generic(
            "echocp", "EchoCPFinetune",
            "finetune.experiments.echocp",
            img_branch, vid_branch, echocp_root, echocp_raw,
            output_dir / "echocp", args.device, args.eval_only, all_results,
        )

    if "lus" in exps:
        _hdr(exps.index("lus") + 1, total, "LUS patient TB (gated MIL + MLP)")
        _run_lus(img_branch, vid_branch, benin_root, rsa_root, lus_raw,
                 output_dir / "lus_patient", args.device, args.eval_only, all_results)

    if "lus_video" in exps:
        _hdr(exps.index("lus_video") + 1, total,
             "LUS clip-level findings (7 binary MLP heads)")
        _run_lus_video(img_branch, vid_branch, benin_root, rsa_root, lus_video_raw,
                       output_dir / "lus_video", args.device, args.eval_only, all_results)

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
    out_dir, ckpt_dir = _prepare_finetune_dirs(out_dir, _find_repo_root())

    cfg = FinetuneConfig.from_dict(raw_cfg.get("finetune", raw_cfg))
    exp = BUSIFinetune(
        data_root=data_root,
        output_dir=str(out_dir),
        cfg=cfg,
        checkpoint_dir=str(ckpt_dir),
    )
    exp.setup(img_branch, device=device, vid_branch=vid_branch)

    if eval_only:
        best = ckpt_dir / "best_head.pt"
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
    out_dir, ckpt_dir = _prepare_finetune_dirs(out_dir, _find_repo_root())

    cfg = FinetuneConfig.from_dict(raw_cfg.get("finetune", raw_cfg))
    exp = EchoNetFinetune(
        data_root=data_root,
        output_dir=str(out_dir),
        cfg=cfg,
        checkpoint_dir=str(ckpt_dir),
    )
    exp.setup(img_branch, device=device, vid_branch=vid_branch)

    if eval_only:
        best = ckpt_dir / "best_head.pt"
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
    from finetune.experiments.lus_patient import LUSPatientFinetune, lus_patient_kwargs_from_raw
    from finetune.base import FinetuneConfig

    include_rsa = bool(raw_cfg.get("include_rsa", False))
    has_benin = benin_root and Path(benin_root).exists()
    has_rsa   = include_rsa and rsa_root and Path(rsa_root).exists()
    if not has_benin and not has_rsa:
        log.warning("[LUS] Benin root not found (and RSA not enabled/found). Skipping.")
        all_results["lus_patient"] = {"skipped": True,
                                      "reason": "dataset roots not found"}
        return

    benin_root = benin_root or ""
    rsa_root   = rsa_root if include_rsa else ""

    out_dir = Path(out_dir)
    out_dir, ckpt_dir = _prepare_finetune_dirs(out_dir, _find_repo_root())

    ft_raw = raw_cfg.get("finetune", raw_cfg)
    cfg = FinetuneConfig.from_dict(ft_raw)
    exp = LUSPatientFinetune(
        output_dir=str(out_dir),
        cfg=cfg,
        checkpoint_dir=str(ckpt_dir),
        **lus_patient_kwargs_from_raw(raw_cfg, ft_raw),
    )
    exp.setup(img_branch, device=device, vid_branch=vid_branch)

    if eval_only:
        best = ckpt_dir / "best_head.pt"
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


def _run_lus_video(img_branch, vid_branch, benin_root, rsa_root, raw_cfg,
                   out_dir, device, eval_only, all_results):
    from finetune.experiments.lus_video import LUSVideoFinetune
    from finetune.base import FinetuneConfig

    has_benin = benin_root and Path(benin_root).exists()
    has_rsa   = rsa_root   and Path(rsa_root).exists()
    if not has_benin and not has_rsa:
        log.warning("[LUSVideo] Neither benin_root nor rsa_root found. Skipping.")
        all_results["lus_video"] = {"skipped": True,
                                    "reason": "dataset roots not found"}
        return

    benin_root = benin_root or ""
    rsa_root   = rsa_root   or ""

    out_dir = Path(out_dir)
    out_dir, ckpt_dir = _prepare_finetune_dirs(out_dir, _find_repo_root())

    ft_raw = raw_cfg.get("finetune", raw_cfg)
    cfg = FinetuneConfig.from_dict(ft_raw)
    exp = LUSVideoFinetune(
        data_root_benin = benin_root,
        data_root_rsa   = rsa_root,
        output_dir      = str(out_dir),
        cfg             = cfg,
        checkpoint_dir  = str(ckpt_dir),
        n_frames        = ft_raw.get("n_frames",   8),
        img_size        = ft_raw.get("img_size",   224),
        encode_bs       = ft_raw.get("encode_bs",  16),
    )
    exp.setup(img_branch, device=device, vid_branch=vid_branch)

    if eval_only:
        best = ckpt_dir / "best_head.pt"
        if best.exists():
            exp.load_head(str(best))
        else:
            log.warning("[LUSVideo] eval-only but best_head.pt not found — running finetune.")
            exp.run()
    else:
        exp.run()

    results = exp.evaluate("test")
    log.info(f"[LUSVideo] {results}")
    all_results["lus_video"] = results


def _run_generic(
    result_key: str,
    cls_name:   str,
    module:     str,
    img_branch, vid_branch,
    data_root:  str,
    raw_cfg:    dict,
    out_dir,
    device:     str,
    eval_only:  bool,
    all_results: dict,
):
    """
    Generic runner for single-root finetune experiments.

    Dynamically imports `cls_name` from `module`, instantiates with the
    standard (data_root, output_dir, cfg) signature, and runs or evaluates.
    """
    import importlib
    from finetune.base import FinetuneConfig

    if not data_root or not Path(data_root).exists():
        log.warning(f"[{result_key}] data_root not found: {data_root!r}. Skipping.")
        all_results[result_key] = {"skipped": True,
                                   "reason": f"data_root not found: {data_root}"}
        return

    out_dir = Path(out_dir)
    out_dir, ckpt_dir = _prepare_finetune_dirs(out_dir, _find_repo_root())

    mod = importlib.import_module(module)
    cls = getattr(mod, cls_name)
    cfg = FinetuneConfig.from_dict(raw_cfg.get("finetune", raw_cfg))
    exp = cls(
        data_root=data_root,
        output_dir=str(out_dir),
        cfg=cfg,
        checkpoint_dir=str(ckpt_dir),
    )
    exp.setup(img_branch, device=device, vid_branch=vid_branch)

    if eval_only:
        best = ckpt_dir / "best_head.pt"
        if best.exists():
            exp.load_head(str(best))
        else:
            log.warning(f"[{result_key}] eval-only but best_head.pt not found — running finetune.")
            exp.run()
    else:
        exp.run()

    results = exp.evaluate("test")
    log.info(f"[{result_key}] {results}")
    all_results[result_key] = results


# ── Comparison sweep ──────────────────────────────────────────────────────────

_EXPERIMENT_REGISTRY = {
    "busi":        ("finetune.experiments.busi",        "BUSIFinetune"),
    "busi_multitask": ("finetune.experiments.busi",     "BUSIMultitaskFinetune"),
    "busbra":      ("finetune.experiments.busbra",      "BUSBRAFinetune"),
    "camus":       ("finetune.experiments.camus",       "CAMUSFinetune"),
    "echonet":     ("finetune.experiments.echonet",     "EchoNetFinetune"),
    "echonet_ped": ("finetune.experiments.echonet_pediatric", "EchoNetPediatricFinetune"),
    "echonet_lvh": ("finetune.experiments.echonet_lvh", "EchoNetLVHFinetune"),
    "mimic_lvvol": ("finetune.experiments.mimic_lvvol", "MIMICLVVolFinetune"),
    "cardiacudc":  ("finetune.experiments.cardiacudc",  "CardiacUDCFinetune"),
    "echocp":      ("finetune.experiments.echocp",      "EchoCPFinetune"),
    "tn3k":            ("finetune.experiments.tn3k",            "TN3KFinetune"),
    "fetal_planes_db": ("finetune.experiments.fetal_planes_db", "FetalPlanesDBFinetune"),
    "lus_video":       ("finetune.experiments.lus_video",       "LUSVideoFinetune"),
}


def _resolve_dataset_roots(cfg: dict, repo: Path) -> dict[str, str]:
    """Merge comparison YAML roots with per-experiment finetune YAML defaults."""
    ft_cfg_dir = repo / "configs" / "finetune"
    roots = dict(cfg.get("dataset_roots", {}))

    _yaml_keys = {
        "busi":        ("busi",        "dataset_root"),
        "busi_multitask": ("busi_multitask", "dataset_root"),
        "busbra":      ("busbra",      "dataset_root"),
        "camus":       ("camus",       "dataset_root"),
        "echonet":     ("echonet",     "dataset_root"),
        "echonet_ped": ("echonet_pediatric", "dataset_root"),
        "echonet_lvh": ("echonet_lvh", "dataset_root"),
        "mimic_lvvol": ("mimic_lvvol", "dataset_root"),
        "cardiacudc":  ("cardiacudc",  "dataset_root"),
        "echocp":      ("echocp",      "dataset_root"),
        "tn3k":            ("tn3k",            "dataset_root"),
        "fetal_planes_db": ("fetal_planes_db", "dataset_root"),
    }
    for exp_key, (yaml_name, field) in _yaml_keys.items():
        if roots.get(exp_key):
            continue
        yaml_path = ft_cfg_dir / f"{yaml_name}.yaml"
        if yaml_path.exists():
            raw = _load_finetune_cfg(str(yaml_path))
            roots[exp_key] = raw.get(field, "")

    if not roots.get("benin") and (ft_cfg_dir / "lus_patient.yaml").exists():
        lus_raw = _load_finetune_cfg(str(ft_cfg_dir / "lus_patient.yaml"))
        roots["benin"] = lus_raw.get("dataset_root_benin", "")
        roots["rsa"]   = lus_raw.get("dataset_root_rsa",   "")
    # lus_video shares the same benin/rsa roots — no extra lookup needed

    return roots


# Comparison sweeps set head_type per backbone×experiment×head loop; smoke YAML must not override it.
_SMOKE_OVERRIDE_SKIP = frozenset({"head_type"})


def _apply_smoke_overrides(
    cfg,
    overrides: dict | None,
    exp_name: str | None = None,
    exp_overrides_map: dict | None = None,
) -> None:
    """Apply global and per-experiment smoke overrides to FinetuneConfig."""
    for src in (overrides, (exp_overrides_map or {}).get(exp_name or "", None)):
        if not src:
            continue
        for key, val in src.items():
            if key in _SMOKE_OVERRIDE_SKIP:
                continue
            if hasattr(cfg, key):
                setattr(cfg, key, val)


def _backbone_checkpoint_missing(spec: dict) -> bool:
    """True when an explicit checkpoint path is set but the file is absent."""
    ckpt = spec.get("checkpoint")
    if not ckpt:
        return False
    return not Path(ckpt).exists()


def _merge_cli_dataset_roots(roots: dict[str, str], args) -> dict[str, str]:
    """CLI --*-root overrides take precedence over comparison YAML."""
    merged = dict(roots)
    _single = {
        "busi":        args.busi_root,
        "echonet":     args.echonet_root,
        "camus":       args.camus_root,
        "echonet_ped": args.echonet_ped_root,
        "echonet_lvh": args.echonet_lvh_root,
        "mimic_lvvol": args.mimic_lvvol_root,
        "cardiacudc":  args.cardiacudc_root,
        "echocp":      args.echocp_root,
    }
    for key, val in _single.items():
        if val:
            merged[key] = val
    if args.benin_root:
        merged["benin"] = args.benin_root
    if args.rsa_root:
        merged["rsa"] = args.rsa_root
    if args.busbra_root:
        merged["busbra"] = args.busbra_root
    if args.tn3k_root:
        merged["tn3k"] = args.tn3k_root
    if args.fetal_planes_db_root:
        merged["fetal_planes_db"] = args.fetal_planes_db_root
    return merged


def _camus_variant_runs(cmp_cfg: dict | None) -> list[dict | None]:
    """Return CAMUS variant override dicts, or [None] for a single default run."""
    if not cmp_cfg:
        return [None]
    variants = cmp_cfg.get("camus_variants")
    if not variants:
        return [None]
    return list(variants)


def _run_comparison_experiment(
    exp_name:    str,
    encoder,
    head_type:   str,
    out_dir:     Path,
    device:      str,
    eval_only:   bool,
    roots:       dict[str, str],
    repo:        Path,
    smoke_overrides: dict | None = None,
    experiment_smoke_overrides: dict | None = None,
    camus_variant_overrides: dict | None = None,
) -> dict | None:
    """Run one backbone × experiment × head_type combination."""
    import importlib
    from finetune.base import FinetuneConfig

    out_dir = Path(out_dir)
    out_dir, ckpt_dir = _prepare_finetune_dirs(out_dir, repo)
    log.info("[%s/%s] results → %s  checkpoints → %s", exp_name, head_type, out_dir, ckpt_dir)

    if exp_name in ("lus", "lus_video"):
        benin_root = roots.get("benin", "")
        rsa_root   = roots.get("rsa", "")
        if exp_name == "lus":
            lus_raw = _load_finetune_cfg(str(repo / "configs" / "finetune" / "lus_patient.yaml"))
            include_rsa = bool(lus_raw.get("include_rsa", False))
            has_data = (benin_root and Path(benin_root).exists()) or \
                       (include_rsa and rsa_root and Path(rsa_root).exists())
        else:
            has_data = (benin_root and Path(benin_root).exists()) or \
                       (rsa_root and Path(rsa_root).exists())
        if not has_data:
            log.warning(f"[{exp_name}] dataset roots not found — skipping.")
            return {"skipped": True, "reason": "dataset roots not found"}

        if exp_name == "lus":
            from finetune.experiments.lus_patient import LUSPatientFinetune, lus_patient_kwargs_from_raw
            lus_raw = _load_finetune_cfg(str(repo / "configs" / "finetune" / "lus_patient.yaml"))
            ft_raw  = dict(lus_raw.get("finetune", lus_raw))
            ft_raw["head_type"] = head_type
            cfg = FinetuneConfig.from_dict(ft_raw)
            _apply_smoke_overrides(cfg, smoke_overrides, exp_name, experiment_smoke_overrides)
            cfg.head_type = head_type
            lus_kwargs = lus_patient_kwargs_from_raw(lus_raw, ft_raw)
            if not lus_kwargs["include_rsa"]:
                lus_kwargs["data_root_rsa"] = ""
            exp = LUSPatientFinetune(
                output_dir=str(out_dir),
                cfg=cfg,
                checkpoint_dir=str(ckpt_dir),
                **lus_kwargs,
            )
        else:
            from finetune.experiments.lus_video import LUSVideoFinetune
            lus_video_raw = _load_finetune_cfg(
                str(repo / "configs" / "finetune" / "lus_video.yaml")
            )
            ft_raw = dict(lus_video_raw.get("finetune", lus_video_raw))
            ft_raw["head_type"] = head_type
            cfg = FinetuneConfig.from_dict(ft_raw)
            _apply_smoke_overrides(cfg, smoke_overrides, exp_name, experiment_smoke_overrides)
            cfg.head_type = head_type
            exp = LUSVideoFinetune(
                data_root_benin = benin_root or "",
                data_root_rsa   = rsa_root or "",
                output_dir      = str(out_dir),
                cfg             = cfg,
                checkpoint_dir  = str(ckpt_dir),
                n_frames        = ft_raw.get("n_frames",  8),
                img_size        = ft_raw.get("img_size",  224),
                encode_bs       = ft_raw.get("encode_bs", 16),
            )
    else:
        if exp_name not in _EXPERIMENT_REGISTRY:
            log.warning(f"Unknown experiment {exp_name!r} — skipping.")
            return {"skipped": True, "reason": f"unknown experiment: {exp_name}"}

        data_root = roots.get(exp_name, "")
        if (not data_root or not Path(data_root).exists()) and exp_name == "busi_multitask":
            data_root = roots.get("busi", "")
        if not data_root or not Path(data_root).exists():
            log.warning(f"[{exp_name}] data_root not found: {data_root!r} — skipping.")
            return {"skipped": True, "reason": f"data_root not found: {data_root}"}

        module_name, cls_name = _EXPERIMENT_REGISTRY[exp_name]
        yaml_map = {
            "echonet_ped": "echonet_pediatric",
        }
        yaml_name = yaml_map.get(exp_name, exp_name)
        raw_cfg = _load_finetune_cfg(str(repo / "configs" / "finetune" / f"{yaml_name}.yaml"))
        ft_raw = dict(raw_cfg.get("finetune", raw_cfg))
        ft_raw["head_type"] = head_type
        if camus_variant_overrides:
            ft_raw.update({k: v for k, v in camus_variant_overrides.items() if k != "variant"})
            if "variant" in camus_variant_overrides:
                ft_raw["camus_variant"] = camus_variant_overrides["variant"]
        cfg = FinetuneConfig.from_dict(ft_raw)
        _apply_smoke_overrides(cfg, smoke_overrides, exp_name, experiment_smoke_overrides)
        cfg.head_type = head_type

        mod = importlib.import_module(module_name)
        cls = getattr(mod, cls_name)
        exp = cls(
            data_root=data_root,
            output_dir=str(out_dir),
            cfg=cfg,
            checkpoint_dir=str(ckpt_dir),
        )

    exp.setup(encoder=encoder, device=device)

    if eval_only:
        best = ckpt_dir / "best_head.pt"
        if best.exists():
            exp.load_head(str(best))
        else:
            log.warning(f"[{exp_name}/{head_type}] eval-only but best_head.pt missing — training.")
            exp.run()
    else:
        exp.run()

    results = exp.evaluate("test")
    try:
        exp.run_viz(results, out_dir)
    except Exception as exc:
        log.warning("[%s/%s] viz skipped: %s", exp_name, head_type, exc)
    log.info(f"[{exp_name}/{head_type}] {results}")
    return results


class _ExperimentLogFilter(logging.Filter):
    """Prefix log records with experiment name (parallel workers)."""

    def __init__(self, exp_name: str):
        super().__init__()
        self.exp_name = exp_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"[{self.exp_name}] {record.msg}"
        return True


_WORKER_STATE: dict = {}


def _init_comparison_worker(log_queue, gpu_queue) -> None:
    _WORKER_STATE["log_queue"] = log_queue
    _WORKER_STATE["gpu_queue"] = gpu_queue
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.INFO)


def _attach_experiment_log_filter(exp_name: str) -> None:
    root = logging.getLogger()
    for h in root.handlers:
        h.filters = [f for f in h.filters if not isinstance(f, _ExperimentLogFilter)]
        h.addFilter(_ExperimentLogFilter(exp_name))


def _start_parallel_log_listener(
    ctx: mp.context.BaseContext,
) -> tuple[mp.Queue, logging.handlers.QueueListener]:
    log_queue: mp.Queue = ctx.Queue(-1)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    listener = logging.handlers.QueueListener(log_queue, handler, respect_handler_level=True)
    listener.start()
    return log_queue, listener


def _resolve_num_gpus(args, cmp_cfg: dict) -> int:
    if args.num_gpus is not None:
        return max(1, args.num_gpus)
    if cmp_cfg.get("num_gpus") is not None:
        return max(1, int(cmp_cfg["num_gpus"]))
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def _should_parallelize_experiments(args, cmp_cfg: dict, n_experiments: int) -> bool:
    if args.no_parallel_experiments:
        return False
    if args.parallel_experiments or cmp_cfg.get("parallel_experiments") is True:
        num_gpus = _resolve_num_gpus(args, cmp_cfg)
        return num_gpus > 1 and n_experiments > 1
    if cmp_cfg.get("parallel_experiments") is False:
        return False
    num_gpus = _resolve_num_gpus(args, cmp_cfg)
    return num_gpus > 1 and n_experiments > 1


def _comparison_parallel_strategy(cmp_cfg: dict) -> str:
    """per_experiment: 1 GPU per experiment, all backbones on that GPU (default)."""
    return cmp_cfg.get("parallel_strategy", "per_experiment")


def _run_comparison_sequential_per_experiment(
    *,
    backbone_specs: list[dict],
    experiments: list[str],
    head_types_cfg: dict,
    default_heads: list[str],
    output_dir: Path,
    device: str,
    eval_only: bool,
    roots: dict[str, str],
    repo: Path,
    model_cfg_dict: dict,
    hf_cache: str,
    smoke_overrides: dict | None,
    experiment_smoke_overrides: dict | None = None,
    cmp_cfg: dict | None = None,
) -> None:
    """Single GPU: each experiment runs all backbones before the next experiment."""
    log.info(
        "Sequential per-experiment mode: %d experiment(s) on %s "
        "(all backbones per experiment)",
        len(experiments), device,
    )
    for exp_name in experiments:
        head_types = head_types_cfg.get(exp_name, default_heads)
        _run_backbones_for_experiment(
            exp_name=exp_name,
            device=device,
            backbone_specs=backbone_specs,
            output_dir=output_dir,
            eval_only=eval_only,
            roots=roots,
            repo=repo,
            model_cfg_dict=model_cfg_dict,
            hf_cache=hf_cache,
            head_types=head_types,
            smoke_overrides=smoke_overrides,
            experiment_smoke_overrides=experiment_smoke_overrides,
            camus_variants=_camus_variant_runs(cmp_cfg) if exp_name == "camus" else None,
        )


def _run_backbones_for_experiment(
    exp_name: str,
    device: str,
    backbone_specs: list[dict],
    output_dir: Path,
    eval_only: bool,
    roots: dict[str, str],
    repo: Path,
    model_cfg_dict: dict,
    hf_cache: str,
    head_types: list[str],
    smoke_overrides: dict | None,
    experiment_smoke_overrides: dict | None = None,
    camus_variants: list[dict] | None = None,
) -> None:
    """Train/eval all backbones for a single experiment on *device*."""
    from finetune.backbones.registry import build_encoder

    log.info("=" * 60)
    log.info("Experiment: %s  device=%s", exp_name, device)
    log.info("=" * 60)

    for backbone_spec in backbone_specs:
        bkey = backbone_spec.get("key", "?")
        spec = dict(backbone_spec)
        if _backbone_checkpoint_missing(spec):
            log.warning(
                "Skipping backbone %r — checkpoint not found: %s",
                bkey, spec.get("checkpoint"),
            )
            continue
        log.info("Backbone: %s", bkey)
        try:
            spec.setdefault("hf_cache_dir", hf_cache)
            encoder = build_encoder(spec, device=device, train_cfg=model_cfg_dict)
            encoder.to(device)
            encoder.eval()
        except Exception as exc:
            log.error("Failed to build encoder %r: %s", bkey, exc)
            continue

        variant_runs = camus_variants if exp_name == "camus" and camus_variants else [None]
        for variant_cfg in variant_runs:
            variant_name = variant_cfg.get("variant") if variant_cfg else None
            from models.heads.finetune_seg import (
                encoder_has_hierarchical_features,
                head_type_requires_hierarchy,
            )
            for head_type in head_types:
                if (
                    head_type_requires_hierarchy(head_type)
                    and not encoder_has_hierarchical_features(encoder)
                ):
                    log.warning(
                        "Skipping %s / %s — %s requires a hierarchical encoder",
                        bkey, head_type, head_type,
                    )
                    continue
                if variant_name:
                    run_dir = output_dir / bkey / exp_name / variant_name / head_type
                    label = f"{exp_name}/{variant_name}"
                else:
                    run_dir = output_dir / bkey / exp_name / head_type
                    label = exp_name
                log.info("  → %s / %s → %s", label, head_type, run_dir)
                try:
                    _run_comparison_experiment(
                        exp_name, encoder, head_type, run_dir,
                        device, eval_only, roots, repo,
                        smoke_overrides=smoke_overrides,
                        experiment_smoke_overrides=experiment_smoke_overrides,
                        camus_variant_overrides=variant_cfg,
                    )
                except Exception as exc:
                    log.error(
                        "Run failed (%s/%s/%s): %s",
                        bkey, label, head_type, exc,
                    )


def _comparison_experiment_worker(task: dict) -> str:
    """ProcessPool worker: all backbones for one experiment on one GPU."""
    exp_name = task["exp_name"]
    _attach_experiment_log_filter(exp_name)
    gpu_id = _WORKER_STATE["gpu_queue"].get()
    try:
        device = f"cuda:{gpu_id}"
        log.info("gpu=%s starting", gpu_id)
        _run_backbones_for_experiment(
            exp_name=exp_name,
            device=device,
            backbone_specs=task["backbone_specs"],
            output_dir=Path(task["output_dir"]),
            eval_only=task["eval_only"],
            roots=task["roots"],
            repo=Path(task["repo"]),
            model_cfg_dict=task["model_cfg_dict"],
            hf_cache=task["hf_cache"],
            head_types=task["head_types"],
            smoke_overrides=task.get("smoke_overrides"),
            experiment_smoke_overrides=task.get("experiment_smoke_overrides"),
            camus_variants=task.get("camus_variants"),
        )
        log.info("gpu=%s finished", gpu_id)
        return exp_name
    except Exception:
        log.exception("Experiment worker failed")
        raise
    finally:
        _WORKER_STATE["gpu_queue"].put(gpu_id)


def _filter_backbone_specs(backbones: list[dict], selected: list[str] | None) -> list[dict]:
    """Return all backbones, or only those whose ``key`` is in *selected*."""
    if not selected:
        return backbones
    available = {spec.get("key") for spec in backbones}
    unknown = [k for k in selected if k not in available]
    if unknown:
        log.error(
            "Unknown backbone(s): %s. Available: %s",
            ", ".join(unknown),
            ", ".join(sorted(k for k in available if k)),
        )
        sys.exit(1)
    selected_set = set(selected)
    filtered = [spec for spec in backbones if spec.get("key") in selected_set]
    log.info("Backbone subset: %s", ", ".join(selected))
    return filtered


def _run_comparison_sequential(
    *,
    backbone_specs: list[dict],
    experiments: list[str],
    head_types_cfg: dict,
    default_heads: list[str],
    output_dir: Path,
    device: str,
    eval_only: bool,
    roots: dict[str, str],
    repo: Path,
    model_cfg_dict: dict,
    hf_cache: str,
    smoke_overrides: dict | None,
    experiment_smoke_overrides: dict | None = None,
) -> None:
    for backbone_spec in backbone_specs:
        bkey = backbone_spec.get("key", "?")
        spec = dict(backbone_spec)
        if _backbone_checkpoint_missing(spec):
            log.warning(
                "Skipping backbone %r — checkpoint not found: %s",
                bkey, spec.get("checkpoint"),
            )
            continue
        log.info("=" * 60)
        log.info(f"Backbone: {bkey}")
        log.info("=" * 60)
        try:
            from finetune.backbones.registry import build_encoder
            spec.setdefault("hf_cache_dir", hf_cache)
            encoder = build_encoder(spec, device=device, train_cfg=model_cfg_dict)
            encoder.to(device)
            encoder.eval()
        except Exception as exc:
            log.error(f"Failed to build encoder {bkey!r}: {exc}")
            continue

        for exp_name in experiments:
            head_types = head_types_cfg.get(exp_name, default_heads)
            for head_type in head_types:
                run_dir = output_dir / bkey / exp_name / head_type
                log.info(f"  → {exp_name} / {head_type} → {run_dir}")
                try:
                    _run_comparison_experiment(
                        exp_name, encoder, head_type, run_dir,
                        device, eval_only, roots, repo,
                        smoke_overrides=smoke_overrides,
                        experiment_smoke_overrides=experiment_smoke_overrides,
                    )
                except Exception as exc:
                    log.error(f"Run failed ({bkey}/{exp_name}/{head_type}): {exc}")


def _run_comparison_parallel(
    *,
    args,
    experiments: list[str],
    head_types_cfg: dict,
    default_heads: list[str],
    backbone_specs: list[dict],
    output_dir: Path,
    eval_only: bool,
    roots: dict[str, str],
    repo: Path,
    model_cfg_dict: dict,
    hf_cache: str,
    smoke_overrides: dict | None,
    experiment_smoke_overrides: dict | None,
    cmp_cfg: dict,
) -> None:
    num_gpus = _resolve_num_gpus(args, cmp_cfg)
    max_workers = min(len(experiments), num_gpus)
    log.info(
        "Parallel per-experiment mode: %d experiment(s), %d GPU worker(s) "
        "(each GPU runs all backbones for one experiment)",
        len(experiments), max_workers,
    )

    ctx = mp.get_context("spawn")
    log_queue, listener = _start_parallel_log_listener(ctx)
    gpu_queue = ctx.Queue()
    for gpu_id in range(max_workers):
        gpu_queue.put(gpu_id)

    tasks = []
    for exp_name in experiments:
        tasks.append({
            "exp_name": exp_name,
            "backbone_specs": backbone_specs,
            "output_dir": str(output_dir),
            "eval_only": eval_only,
            "roots": roots,
            "repo": str(repo),
            "model_cfg_dict": model_cfg_dict,
            "hf_cache": hf_cache,
            "head_types": head_types_cfg.get(exp_name, default_heads),
            "smoke_overrides": smoke_overrides,
            "experiment_smoke_overrides": experiment_smoke_overrides,
            "camus_variants": _camus_variant_runs(cmp_cfg) if exp_name == "camus" else None,
        })

    failed: list[str] = []
    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_comparison_worker,
            initargs=(log_queue, gpu_queue),
        ) as pool:
            futures = {
                pool.submit(_comparison_experiment_worker, task): task["exp_name"]
                for task in tasks
            }
            for fut in as_completed(futures):
                exp_name = futures[fut]
                try:
                    fut.result()
                    log.info("Experiment complete: %s", exp_name)
                except Exception as exc:
                    failed.append(exp_name)
                    log.error("Experiment failed: %s — %s", exp_name, exc)
    finally:
        listener.stop()

    if failed:
        log.error("Parallel comparison had failures: %s", ", ".join(failed))


def run_comparison(args) -> None:
    """Sweep backbone × head_type × experiment and write comparison reports."""
    repo = _find_repo_root()
    cfg_path = Path(args.comparison_config)
    if not cfg_path.is_absolute():
        cfg_path = repo / cfg_path
    with open(cfg_path) as f:
        cmp_cfg = yaml.safe_load(f)

    train_cfg_full = _load_config(cmp_cfg["train_config"])
    model_cfg_dict = dict(train_cfg_full.get("model", {}))

    hf_cache = _resolve_hf_cache()
    model_cfg_dict["hf_cache_dir"] = hf_cache
    model_cfg_dict["frozen_teacher"] = None

    output_dir = Path(cmp_cfg.get("output_dir", "results/finetune/representative"))
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device    = args.device or cmp_cfg.get("device", "cuda")
    eval_only = args.eval_only or cmp_cfg.get("eval_only", False)
    roots     = _merge_cli_dataset_roots(
        _resolve_dataset_roots(cmp_cfg, repo), args
    )
    smoke_overrides = cmp_cfg.get("smoke_overrides")
    experiment_smoke_overrides = cmp_cfg.get("experiment_smoke_overrides")

    from finetune.report import generate_comparison_report

    head_types_cfg = cmp_cfg.get("head_types", {"default": ["linear"]})
    default_heads  = head_types_cfg.get("default", ["linear"])

    backbone_specs = _filter_backbone_specs(
        cmp_cfg.get("backbones", []), args.backbones
    )
    experiments = list(cmp_cfg.get("experiments", []))
    if args.experiments is not None:
        unknown = [e for e in args.experiments if e not in experiments]
        if unknown:
            log.error(
                "Experiment(s) not in comparison config: %s (configured: %s)",
                ", ".join(unknown),
                ", ".join(experiments),
            )
            sys.exit(1)
        experiments = [e for e in experiments if e in args.experiments]
        log.info("Experiment subset: %s", ", ".join(experiments))
    strategy = _comparison_parallel_strategy(cmp_cfg)
    n_backbones = len(backbone_specs)
    log.info(
        "Comparison layout: %d experiment(s) × %d backbone(s) — strategy=%s",
        len(experiments), n_backbones, strategy,
    )

    use_parallel = (
        strategy == "per_experiment"
        and _should_parallelize_experiments(args, cmp_cfg, len(experiments))
    )

    if use_parallel:
        _run_comparison_parallel(
            args=args,
            experiments=experiments,
            head_types_cfg=head_types_cfg,
            default_heads=default_heads,
            backbone_specs=backbone_specs,
            output_dir=output_dir,
            eval_only=eval_only,
            roots=roots,
            repo=repo,
            model_cfg_dict=model_cfg_dict,
            hf_cache=hf_cache,
            smoke_overrides=smoke_overrides,
            experiment_smoke_overrides=experiment_smoke_overrides,
            cmp_cfg=cmp_cfg,
        )
    elif strategy == "per_experiment":
        _run_comparison_sequential_per_experiment(
            backbone_specs=backbone_specs,
            experiments=experiments,
            head_types_cfg=head_types_cfg,
            default_heads=default_heads,
            output_dir=output_dir,
            device=device,
            eval_only=eval_only,
            roots=roots,
            repo=repo,
            model_cfg_dict=model_cfg_dict,
            hf_cache=hf_cache,
            smoke_overrides=smoke_overrides,
            experiment_smoke_overrides=experiment_smoke_overrides,
            cmp_cfg=cmp_cfg,
        )
    else:
        _run_comparison_sequential(
            backbone_specs=backbone_specs,
            experiments=experiments,
            head_types_cfg=head_types_cfg,
            default_heads=default_heads,
            output_dir=output_dir,
            device=device,
            eval_only=eval_only,
            roots=roots,
            repo=repo,
            model_cfg_dict=model_cfg_dict,
            hf_cache=hf_cache,
            smoke_overrides=smoke_overrides,
            experiment_smoke_overrides=experiment_smoke_overrides,
        )

    report = generate_comparison_report(output_dir)
    from finetune.report import generate_dashboard
    from finetune.results_collector import collect_all_sweeps

    dashboard_tree = collect_all_sweeps(from_logs=False)
    generate_dashboard(dashboard_tree)
    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print(f"\nResults written to: {output_dir}")
    print(f"Checkpoints on Capstor: /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/Finetune/")
    print(f"\nCharts written to: {output_dir / 'charts'}")
    print(f"Dashboard written to: results/finetune/_dashboard/")


if __name__ == "__main__":
    main()
