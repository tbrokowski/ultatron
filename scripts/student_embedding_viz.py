#!/usr/bin/env python3
"""
scripts/student_embedding_viz.py  ·  Student embedding UMAP/t-SNE CLI
======================================================================

Extract frozen global embeddings from a Hiera student checkpoint and plot
anatomy-family UMAP/t-SNE panels.

Usage
-----
    python scripts/student_embedding_viz.py \\
        --checkpoint /capstor/.../StudentPretrainPilot/latest.pt \\
        --config configs/student/student_pretrain_pilot.yaml \\
        --output-dir viz_outputs/student_pilot \\
        --split train --stream both \\
        --samples-per-family 500 \\
        --embedding-space global_proj
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger(__name__)

_DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "student" / "student_pretrain_pilot.yaml"


def _load_config(path: str) -> dict:
    import yaml

    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = PROJECT_ROOT / cfg.pop("_base_")
        with open(base_path) as f:
            base = yaml.safe_load(f)

        def _deep_merge(base_d: dict, override: dict) -> dict:
            out = dict(base_d)
            for k, v in override.items():
                if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out

        cfg = _deep_merge(base, cfg)
    return cfg


def _resolve_manifest(cfg: dict, manifest_override: str | None) -> str:
    from data.infra.cscs_paths import CSCSConfig

    if manifest_override:
        p = Path(manifest_override)
        return str(p if p.is_absolute() else PROJECT_ROOT / p)

    mcfg = Path(cfg["manifest"]["path"])
    if not mcfg.is_absolute():
        mcfg = PROJECT_ROOT / mcfg
    if mcfg.exists():
        return str(mcfg)

    cscs = CSCSConfig.from_env()
    return str(cscs.manifest_path(mcfg.name))


def _build_datamodule(cfg: dict, manifest_path: str):
    from data.infra.cscs_paths import CSCSConfig
    from data.pipeline.datamodule import USFoundationDataModule
    from data.pipeline.transforms import build_transform_configs

    cscs = CSCSConfig.from_env()
    img_cfg, vid_cfg = build_transform_configs(cfg["transforms"])

    manifest_cfg = cfg.get("manifest", {})
    root_remap = manifest_cfg.get("root_remap")
    if root_remap is None:
        root_remap = cscs.remap_dict()

    cur = cfg.get("curriculum", {})
    student_data = cfg.get("student_data", {})
    # Transform/collator grid (stride 16) — not student_data.patch_size (Hiera stride 4).
    transform_patch_size = cfg.get("transforms", {}).get("patch_size", 16)
    student_patch_stride = student_data.get("patch_size", 4)

    loaders = cfg.get("loaders", {})
    dm = USFoundationDataModule(
        manifest_path=manifest_path,
        image_batch_size=loaders.get("image_batch_size", 32),
        video_batch_size=loaders.get("video_batch_size", 1),
        num_workers=loaders.get("num_workers", 2),
        pin_memory=loaders.get("pin_memory", True),
        patch_size=transform_patch_size,
        total_training_steps=cur.get("total_training_steps", 15_000),
        image_samples_per_epoch=cur.get("image_samples_per_epoch", 50_000),
        video_samples_per_epoch=cur.get("video_samples_per_epoch", 10_000),
        curriculum_stage_fracs=cur.get("alp_stage_fracs"),
        alp_alpha_init=float(cur.get("alp_alpha_init", 0.1)),
        alp_alpha_final=float(cur.get("alp_alpha_final", 0.9)),
        alp_guidance_threshold_init=float(cur.get("alp_guidance_threshold_init", 0.1)),
        alp_guidance_threshold_final=float(cur.get("alp_guidance_threshold_final", 0.9)),
        alp_n_frames=cur.get("alp_n_frames"),
        exclude_datasets=manifest_cfg.get("exclude_datasets"),
        root_remap=root_remap,
        image_cfg=img_cfg,
        video_cfg=vid_cfg,
    )
    dm.setup()
    dm._student_patch_stride = student_patch_stride  # for embedding extractor
    return dm


def _parse_bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "on")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Student embedding UMAP/t-SNE visualisation by anatomy family",
    )
    parser.add_argument("--checkpoint", required=True, help="Student .pt checkpoint path")
    parser.add_argument(
        "--config", default=str(_DEFAULT_CONFIG),
        help="Student pretrain YAML (default: student_pretrain_pilot.yaml)",
    )
    parser.add_argument("--manifest", default=None, help="Override manifest JSONL path")
    parser.add_argument("--output-dir", default=None, help="Output directory for figures")
    parser.add_argument(
        "--split", choices=("train", "val"), default="train",
        help="Manifest split (default: train = pretrain image path)",
    )
    parser.add_argument(
        "--stream", choices=("image", "video", "both"), default="both",
        help="Data stream to extract (default: both = image + video on one plot)",
    )
    parser.add_argument(
        "--balance-anatomy", nargs="?", const=True, default=True,
        type=lambda x: True if x is None else _parse_bool(x),
        help="Sample per anatomy family from manifest (default: true)",
    )
    parser.add_argument(
        "--no-balance-anatomy", dest="balance_anatomy", action="store_false",
        help="Sequential dataloader extraction (may miss rare families)",
    )
    parser.add_argument(
        "--samples-per-family", type=int, default=500,
        help="Samples per anatomy family per stream when --balance-anatomy (default: 500)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap total points for plotting (default: all balanced samples)",
    )
    parser.add_argument(
        "--min-per-family", type=int, default=1,
        help="Min samples to keep a family in plots (default: 1)",
    )
    parser.add_argument(
        "--plot-max-points", type=int, default=50_000,
        help="Cap points for UMAP/t-SNE scatter",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Inference batch size (default: from config loaders)",
    )
    parser.add_argument(
        "--embedding-space", choices=("global", "global_proj"), default="global_proj",
    )
    parser.add_argument(
        "--use-ema", nargs="?", const=True, default=True,
        type=lambda x: True if x is None else _parse_bool(x),
        help="Load ema_student weights (default: true)",
    )
    parser.add_argument(
        "--method", choices=("umap", "tsne", "both"), default="both",
    )
    parser.add_argument(
        "--cache-features", nargs="?", const=True, default=True,
        type=lambda x: True if x is None else _parse_bool(x),
        help="Save embeddings.npz (default: true)",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Re-plot from cached embeddings.npz in output-dir",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = _load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = args.output_dir
    if output_dir is None:
        label = ckpt_path.stem
        output_dir = str(ckpt_path.parent / "embedding_viz" / label)
    output_dir = Path(output_dir)

    checkpoint_meta: dict = {}
    extractor = None

    if not args.skip_extract:
        from finetune.backbones.student_encoder import StudentEncoder
        from viz.student_embedding import StudentEmbeddingExtractor, run_embedding_viz

        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        checkpoint_meta = {
            "path": str(ckpt_path),
            "step": ckpt.get("step", ckpt.get("global_step")),
            "stage": ckpt.get("stage", ckpt.get("current_phase")),
            "use_ema": args.use_ema,
        }

        manifest_path = _resolve_manifest(cfg, args.manifest)
        log.info("Manifest: %s", manifest_path)
        dm = _build_datamodule(cfg, manifest_path)

        encoder = StudentEncoder(
            checkpoint=str(ckpt_path),
            use_ema=args.use_ema,
            device=device,
        )
        extractor = StudentEmbeddingExtractor(
            encoder, dm, device=device, embedding_space=args.embedding_space,
            student_patch_stride=getattr(dm, "_student_patch_stride", 4),
        )
        batch_size = args.batch_size or cfg.get("loaders", {}).get("image_batch_size", 32)
        video_batch_size = cfg.get("loaders", {}).get("video_batch_size", 1)
        if args.stream == "video":
            batch_size = args.batch_size or video_batch_size
    else:
        from viz.student_embedding import run_embedding_viz
        batch_size = args.batch_size or 32
        video_batch_size = None

    summary = run_embedding_viz(
        extractor,
        output_dir=output_dir,
        split=args.split,
        stream=args.stream,
        balance_anatomy=args.balance_anatomy,
        samples_per_family=args.samples_per_family,
        max_samples=args.max_samples,
        min_per_family=args.min_per_family,
        plot_max_points=args.plot_max_points,
        batch_size=batch_size,
        video_batch_size=video_batch_size,
        method=args.method,
        cache_features=args.cache_features,
        skip_extract=args.skip_extract,
        checkpoint_meta=checkpoint_meta or None,
        seed=args.seed,
    )

    print(f"\nPlotted {summary['n_plotted']} embeddings "
          f"({summary['embedding_space']}, {summary['embed_dim']}-d)")
    print(f"Anatomy families: {len(summary['per_anatomy'])}")
    print(f"Output: {output_dir}")
    for name, path in summary["figures"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
