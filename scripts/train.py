#!/usr/bin/env python3
"""
scripts/train.py  ·  Oura training entry point
===============================================

Thin CLI wrapper.  All logic lives in oura/train/trainer.py.

Usage (single node / testing):
    python scripts/train.py \\
        --config configs/experiments/full_oura.yaml

Usage (multi-node via torchrun, called by run_training_job.sh):
    torchrun \\
        --nnodes=$SLURM_NNODES \\
        --nproc_per_node=4 \\
        --node_rank=$SLURM_NODEID \\
        --master_addr=$MASTER_ADDR \\
        --master_port=$MASTER_PORT \\
        --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
        scripts/train.py \\
            --config configs/experiments/full_oura.yaml

Flags:
    --config   Path to experiment YAML (inherits from base configs)
    --resume   Path to checkpoint .pt (auto-detects latest.pt if not given)
    --phase    Force-start at this phase number (1–4)
    --no-7b    Skip loading the DINOv3-7B frozen teacher (~14 GB saved)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import yaml


def _setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _load_config(path: str) -> dict:
    """Load YAML config with simple _base_ inheritance."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    bases = cfg.pop("_base_", [])
    merged = {}
    for base_path in bases:
        base_cfg = _load_config(base_path)
        _deep_merge(merged, base_cfg)
    _deep_merge(merged, cfg)
    return merged


def _deep_merge(base: dict, override: dict):
    """Merge override into base in-place, recursing into nested dicts."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _build_finetune_experiments(cfg: dict, ckpt_dir: Path, log) -> list:
    """
    Instantiate Phase 4 finetune experiments from configs/finetune/*.yaml.
    Returns a (potentially empty) list.  Experiments that are missing a
    valid dataset_root are silently skipped with a warning.
    """
    import yaml
    repo = Path(__file__).resolve().parent.parent
    ft_cfg_dir = repo / "configs" / "finetune"
    results_dir = ckpt_dir.parent / "finetune"
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = []

    def _load_ft_yaml(name: str) -> dict:
        p = ft_cfg_dir / name
        if not p.exists():
            return {}
        with open(p) as f:
            return yaml.safe_load(f)

    # ── BUSI ─────────────────────────────────────────────────────────────────
    try:
        from finetune.experiments.busi import BUSIFinetune
        from finetune.base import FinetuneConfig
        raw = _load_ft_yaml("busi.yaml")
        root = raw.get("dataset_root", "")
        if root and Path(root).exists():
            exp = BUSIFinetune(
                data_root  = root,
                output_dir = str(results_dir / "busi"),
                cfg        = FinetuneConfig.from_dict(raw.get("finetune", raw)),
            )
            experiments.append(exp)
            log.info(f"[Phase4] BUSI experiment ready: {root}")
        else:
            log.warning(f"[Phase4] BUSI root not found ({root!r}), skipping.")
    except Exception as e:
        log.warning(f"[Phase4] BUSI experiment failed to load: {e}")

    # ── EchoNet ───────────────────────────────────────────────────────────────
    try:
        from finetune.experiments.echonet import EchoNetFinetune
        from finetune.base import FinetuneConfig
        raw = _load_ft_yaml("echonet.yaml")
        root = raw.get("dataset_root", "")
        if root and Path(root).exists():
            exp = EchoNetFinetune(
                data_root  = root,
                output_dir = str(results_dir / "echonet"),
                cfg        = FinetuneConfig.from_dict(raw.get("finetune", raw)),
            )
            experiments.append(exp)
            log.info(f"[Phase4] EchoNet experiment ready: {root}")
        else:
            log.warning(f"[Phase4] EchoNet root not found ({root!r}), skipping.")
    except Exception as e:
        log.warning(f"[Phase4] EchoNet experiment failed to load: {e}")

    # ── LUS patient ───────────────────────────────────────────────────────────
    try:
        from finetune.experiments.lus_patient import LUSPatientFinetune
        from finetune.base import FinetuneConfig
        raw    = _load_ft_yaml("lus_patient.yaml")
        ft_raw = raw.get("finetune", raw)
        benin  = raw.get("dataset_root_benin", "")
        rsa    = raw.get("dataset_root_rsa",   "")
        if (benin and Path(benin).exists()) or (rsa and Path(rsa).exists()):
            exp = LUSPatientFinetune(
                data_root_benin = benin,
                data_root_rsa   = rsa,
                output_dir      = str(results_dir / "lus_patient"),
                cfg             = FinetuneConfig.from_dict(ft_raw),
                n_frames        = ft_raw.get("n_frames", 8),
                img_size        = ft_raw.get("img_size",  224),
            )
            experiments.append(exp)
            log.info(f"[Phase4] LUS-patient experiment ready")
        else:
            log.warning(f"[Phase4] LUS roots not found, skipping.")
    except Exception as e:
        log.warning(f"[Phase4] LUS experiment failed to load: {e}")

    return experiments


def main():
    parser = argparse.ArgumentParser(description="Oura foundation model training")
    parser.add_argument("--config",  required=True,
                        help="Experiment config YAML (supports _base_ inheritance)")
    parser.add_argument("--resume",   default=None,
                        help="Checkpoint .pt to resume from")
    parser.add_argument("--ckpt-dir", default=None,
                        help="Directory to save checkpoints (overrides default current_run/)")
    parser.add_argument("--phase",    type=int, default=None,
                        help="Force-start at phase number (1–4)")
    parser.add_argument("--no-7b",    action="store_true",
                        help="Skip DINOv3-7B frozen teacher")
    parser.add_argument("--finetune", action="store_true",
                        help="Run Phase 4 finetune experiments (BUSI, EchoNet, LUS) "
                             "after Phase 3 on rank 0. Requires dataset roots in "
                             "configs/finetune/*.yaml")
    args = parser.parse_args()

    rank, world_size, local_rank = _setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"[rank{rank}] %(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    cfg = _load_config(args.config)
    if rank == 0:
        log.info(f"Config loaded from {args.config}")
        log.info(f"World size: {world_size}")

    # ── Imports (after sys.path is set) ───────────────────────────────────────
    from models import ModelConfig, build_image_branch, build_video_branch
    from models.branches.shared import CrossBranchDistillation, PrototypeHead
    from train.trainer import Trainer, TrainConfig
    from data.pipeline.datamodule import USFoundationDataModule
    from data.infra.cscs_paths import CSCSConfig
    from torch.nn.parallel import DistributedDataParallel as DDP

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    cscs     = CSCSConfig.from_env()
    # Prefer scratch HF cache (no store quota issues); fall back to store
    _scratch_hf = cscs.scratch_path("hf_cache")
    _store_hf   = cscs.store_path("hf_cache")
    hf_cache = str(_scratch_hf if _scratch_hf.exists() else _store_hf)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else cscs.checkpoints_dir(phase=1).parent / "current_run"
    log_dir  = cscs.scratch_path("logs") / "current_run"

    # ── Build models ──────────────────────────────────────────────────────────
    model_cfg = ModelConfig.from_dict(cfg.get("model", {}))
    model_cfg.hf_cache_dir = hf_cache
    if args.no_7b:
        model_cfg.frozen_teacher = None

    img_branch = build_image_branch(model_cfg, device=device)
    vid_branch = build_video_branch(model_cfg, device=device)

    img_dim = img_branch.embed_dim
    vid_dim = vid_branch.embed_dim
    dtype   = model_cfg.torch_dtype

    cross_distill = CrossBranchDistillation(
        img_dim, vid_dim, model_cfg.align_dim
    ).to(device=device, dtype=dtype)
    proto_head = PrototypeHead(
        model_cfg.align_dim, model_cfg.n_prototypes
    ).to(device=device, dtype=dtype)

    # ── DDP wrap ──────────────────────────────────────────────────────────────
    if world_size > 1:
        # Image branch: some params (e.g. DINOv3 registers) may be unused in Phase 1 loss.
        img_branch.student = DDP(
            img_branch.student, device_ids=[local_rank],
            find_unused_parameters=True,
        )
        vid_branch.student = DDP(
            vid_branch.student, device_ids=[local_rank],
            find_unused_parameters=True,
        )
        cross_distill = DDP(cross_distill, device_ids=[local_rank],
                            find_unused_parameters=True)
        proto_head    = DDP(proto_head,    device_ids=[local_rank],
                            find_unused_parameters=True)

    # ── Build datamodule ──────────────────────────────────────────────────────
    train_cfg = TrainConfig.from_dict(cfg.get("train", {}))
    total_steps = cfg.get("curriculum", {}).get("total_training_steps", 300_000)

    # Build datamodule inline using config values
    from data.pipeline.transforms import (
        ImageSSLTransformConfig, VideoSSLTransformConfig, FreqMaskConfig
    )
    img_raw   = dict(cfg["transforms"]["image"])
    vid_raw   = dict(cfg["transforms"]["video"])
    img_freq  = img_raw.pop("freq_mask", {})
    vid_freq  = vid_raw.pop("freq_mask", {})

    img_tcfg = ImageSSLTransformConfig(
        **img_raw,
        freq_mask=FreqMaskConfig(**img_freq) if img_freq else FreqMaskConfig(),
    )
    vid_tcfg = VideoSSLTransformConfig(
        **vid_raw,
        freq_mask=FreqMaskConfig(**vid_freq) if vid_freq else FreqMaskConfig(),
    )

    # Resolve manifest: use config path if it exists (absolute or repo-relative)
    _manifest_cfg = Path(cfg["manifest"]["path"])
    if not _manifest_cfg.is_absolute():
        _manifest_cfg = Path(__file__).resolve().parent.parent / _manifest_cfg
    if _manifest_cfg.exists():
        manifest_path = str(_manifest_cfg)
    else:
        manifest_path = str(cscs.manifest_path(Path(cfg["manifest"]["path"]).name))

    # Use config root_remap if present (e.g. {} to use manifest paths as-is from store).
    # Otherwise remap store → scratch for staged runs.
    root_remap = cfg["manifest"].get("root_remap")
    if root_remap is None:
        root_remap = cscs.remap_dict()

    dm = USFoundationDataModule(
        manifest_path           = manifest_path,
        image_batch_size        = cfg["loaders"]["image_batch_size"],
        video_batch_size        = cfg["loaders"]["video_batch_size"],
        num_workers             = cfg["loaders"]["num_workers"],
        pin_memory              = cfg["loaders"]["pin_memory"],
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

    # ── Phase 4 finetune experiments (optional, rank-0 only) ─────────────────
    finetune_experiments = []
    if args.finetune and rank == 0:
        finetune_experiments = _build_finetune_experiments(cfg, ckpt_dir, log)

    # ── Build trainer and run ─────────────────────────────────────────────────
    # Disable AMP GradScaler for bfloat16 (not implemented for bf16 on some CUDA; bf16 doesn't need scaling).
    use_amp_scaler = model_cfg.dtype != "bfloat16"
    trainer = Trainer(
        cfg                   = train_cfg,
        img_branch            = img_branch,
        vid_branch            = vid_branch,
        cross_distill         = cross_distill,
        proto_head            = proto_head,
        dm                    = dm,
        ckpt_dir              = ckpt_dir,
        log_dir               = log_dir,
        finetune_experiments  = finetune_experiments,
        rank                  = rank,
        local_rank            = local_rank,
        total_steps           = total_steps,
        no_7b                 = args.no_7b,
        use_amp_scaler        = use_amp_scaler,
    )

    if args.phase:
        trainer.current_phase = args.phase

    resume_path = args.resume or str(ckpt_dir / "latest.pt")
    if Path(resume_path).exists():
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":
    main()
