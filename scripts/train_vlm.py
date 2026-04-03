#!/usr/bin/env python3
"""
scripts/train_vlm.py  ·  Ultatron VLM GRPO training entry point
================================================================

Trains Qwen2.5-VL 7B with LoRA as the student, using the frozen Ultatron
backbone (DINOv3-L + V-JEPA2-L) as a domain-adapted visual encoder.
SAM2 is used as an agentic tool inside the iMCoT loop; Med-Gemini provides
accuracy rewards; GRPO optimises the LoRA + projector parameters.

Usage (single GPU):
    python scripts/train_vlm.py --config configs/vlm/run1_vlm.yaml

Usage (multi-GPU single-node via torchrun):
    torchrun --nproc_per_node=4 scripts/train_vlm.py \\
        --config configs/vlm/run1_vlm.yaml

Usage (multi-node CSCS, called by scripts/submit_vlm.sh):
    torchrun \\
        --nnodes=$SLURM_NNODES \\
        --nproc_per_node=4 \\
        --rdzv_backend=c10d \\
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\
        scripts/train_vlm.py \\
            --config configs/vlm/run1_vlm.yaml

Flags:
    --config        Path to YAML config (supports _base_ inheritance)
    --resume        Path to checkpoint directory to resume from
    --stage         Force-start at stage 1|2|3
    --eval-only     Run evaluation only (no training)
    --no-sam2       Disable SAM2 tool (Stage 1 rule-based only)
    --no-medgemini  Disable Med-Gemini judge (fall back to rule-based rewards)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist
import yaml


# ── Config loading (mirrors scripts/train.py) ─────────────────────────────────

def _load_config(path: str) -> dict:
    """Load YAML config with _base_ inheritance."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    bases = cfg.pop("_base_", [])
    merged = {}
    for base in bases:
        _deep_merge(merged, _load_config(base))
    _deep_merge(merged, cfg)
    return merged


def _deep_merge(base: dict, override: dict):
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ── Distributed setup ─────────────────────────────────────────────────────────

def _setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# ── Build helpers ─────────────────────────────────────────────────────────────

def _build_ultatron_backbone(cfg: dict, device: str, no_7b: bool = True):
    """
    Load the frozen Ultatron image branch (EMA teacher) from run1 checkpoint.
    Used to extract patch_tokens at rollout time.
    """
    backbone_cfg = cfg.get("backbone", {})
    ckpt_path    = backbone_cfg.get("checkpoint")

    if not ckpt_path or not Path(ckpt_path).exists():
        log.warning(f"Ultatron backbone checkpoint not found at {ckpt_path!r}. "
                    f"Backbone tokens will not be injected into the VLM.")
        return None

    from models.model_config import ModelConfig, build_image_branch
    model_cfg = ModelConfig.from_dict(cfg.get("model", {}))
    model_cfg.frozen_teacher = None   # never load 7B in VLM mode
    model_cfg.hf_cache_dir   = cfg.get("student", {}).get("hf_cache_dir")

    img_branch = build_image_branch(model_cfg, device=device)

    # Load checkpoint weights into img_branch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    img_keys = {k[len("img_branch."):]: v for k, v in state.items()
                if k.startswith("img_branch.")}
    if img_keys:
        missing, unexpected = img_branch.load_state_dict(img_keys, strict=False)
        log.info(f"Ultatron img_branch loaded from {ckpt_path}  "
                 f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        log.warning("No 'img_branch.*' keys found in checkpoint.")

    # Freeze everything
    for p in img_branch.parameters():
        p.requires_grad_(False)
    img_branch.eval()
    return img_branch


def _build_sam2_tool(cfg: dict):
    """Build frozen SAM2 tool + SAMTok bridge."""
    from vlm.tools.sam2_tool  import SAM2Tool
    from vlm.tools.samtok     import SAMTokBridge

    sam2_cfg = cfg.get("sam2", {})
    sam2 = SAM2Tool(
        model_cfg    = sam2_cfg.get("model_cfg"),
        checkpoint   = sam2_cfg.get("checkpoint"),
        device       = "cuda",
        hf_cache_dir = sam2_cfg.get("hf_cache_dir"),
    )
    samtok = SAMTokBridge.build(
        mode        = sam2_cfg.get("samtok_mode", "overlay"),
        samtok_ckpt = sam2_cfg.get("samtok_ckpt"),
        device      = "cuda",
    )
    return sam2, samtok


def _build_student(cfg: dict, projector, tool_registry, device: str):
    """Build Qwen2.5-VL student with LoRA + UltatronProjector."""
    from vlm.student import StudentModel, StudentConfig

    stu_cfg = StudentConfig.from_dict(cfg.get("student", {}))
    stu_cfg.device = device

    student = StudentModel(
        cfg            = stu_cfg,
        projector      = projector,
        tool_registry  = tool_registry,
        load_ref_model = True,
    )
    return student


def _build_reward(cfg: dict, no_medgemini: bool = False):
    """Build CompositeReward from config."""
    from vlm.rewards.composite import CompositeReward

    if no_medgemini:
        # Force rule-based mode
        reward_cfg = dict(cfg.get("reward", {}))
        mg_section = dict(reward_cfg.get("medgemini", {}))
        mg_section["mode"] = "rule"
        reward_cfg["medgemini"] = mg_section
        cfg = dict(cfg)
        cfg["reward"] = reward_cfg

    return CompositeReward.from_config(cfg)


def _build_data_module(cfg: dict, stage: int):
    """Build VLMDataModule for the given training stage."""
    from vlm.grpo.data import VLMDataModule

    data_cfg = cfg.get("data", {})
    manifest = data_cfg.get("manifest_path")
    if not manifest or not Path(manifest).exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest!r}. "
            f"Run: python scripts/build_manifest.py --config configs/run1/data_run1.yaml "
            f"--out <manifest_path>"
        )

    return VLMDataModule.for_stage(
        manifest_path   = manifest,
        stage           = stage,
        batch_size      = data_cfg.get("batch_size", 4),
        num_workers     = data_cfg.get("num_workers", 4),
        img_size        = data_cfg.get("img_size", 224),
        anatomy_weights = data_cfg.get("anatomy_weights", {}),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ultatron VLM GRPO training")
    parser.add_argument("--config",       required=True,
                        help="YAML config (supports _base_ inheritance)")
    parser.add_argument("--resume",       default=None,
                        help="Checkpoint directory to resume from")
    parser.add_argument("--stage",        type=int, default=None,
                        help="Force-start at stage 1|2|3")
    parser.add_argument("--eval-only",    action="store_true",
                        help="Run evaluation only")
    parser.add_argument("--no-sam2",      action="store_true",
                        help="Disable SAM2 tool (Stage 1 rule-based only)")
    parser.add_argument("--no-medgemini", action="store_true",
                        help="Disable Med-Gemini judge (rule-based rewards only)")
    args = parser.parse_args()

    rank, world_size, local_rank = _setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(
        level   = logging.INFO  if rank == 0 else logging.WARNING,
        format  = f"[rank{rank}] %(asctime)s %(levelname)s %(message)s",
    )
    global log
    log = logging.getLogger(__name__)

    cfg = _load_config(args.config)
    if rank == 0:
        log.info(f"Config: {args.config}   world_size={world_size}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    # ── Determine starting stage ──────────────────────────────────────────────
    start_stage = args.stage or 1

    # ── Build projector ───────────────────────────────────────────────────────
    from vlm.projector import UltatronProjector
    proj_cfg = cfg.get("projector", {})
    backbone_ckpt = cfg.get("backbone", {}).get("checkpoint")
    projector = UltatronProjector.from_checkpoint(
        ckpt_path = backbone_ckpt or "",
        device    = device,
        dtype     = torch.bfloat16,
        **{k: v for k, v in proj_cfg.items() if k in
           ("img_dim", "vid_dim", "qwen_dim", "mid_dim")},
    )
    if rank == 0:
        log.info("UltatronProjector initialised.")

    # ── Build SAM2 tool ───────────────────────────────────────────────────────
    if not args.no_sam2:
        sam2_tool, samtok_bridge = _build_sam2_tool(cfg)
    else:
        sam2_tool, samtok_bridge = None, None
        if rank == 0:
            log.info("SAM2 tool DISABLED (--no-sam2).")

    # ── Build tool registry ───────────────────────────────────────────────────
    # Note: tokenizer/processor not available yet; we inject after student build
    tool_registry_placeholder = None  # will be set below

    # ── Build student ─────────────────────────────────────────────────────────
    student = _build_student(cfg, projector, tool_registry=None, device=device)

    # Now wire the tool registry with the real tokenizer
    if sam2_tool is not None:
        from vlm.tools.registry import ToolRegistry
        tool_registry = ToolRegistry(
            sam2_tool     = sam2_tool,
            samtok_bridge = samtok_bridge,
            tokenizer     = student._processor.tokenizer,
            processor     = student._processor,
        )
        student.tool_registry = tool_registry
        if rank == 0:
            log.info("ToolRegistry wired to StudentModel.")

    # ── Build Ultatron backbone (frozen) ──────────────────────────────────────
    ultatron_backbone = _build_ultatron_backbone(cfg, device)

    # ── Build rewards ─────────────────────────────────────────────────────────
    reward_fn = _build_reward(cfg, no_medgemini=args.no_medgemini)
    if rank == 0:
        mode = cfg.get("reward", {}).get("medgemini", {}).get("mode", "auto")
        log.info(f"Reward: Med-Gemini mode={mode!r}  "
                 f"weights={cfg.get('reward', {}).get('weights', {})}")

    # ── Build data module ─────────────────────────────────────────────────────
    data_module = _build_data_module(cfg, stage=start_stage)

    # ── Build GRPO components ─────────────────────────────────────────────────
    from vlm.grpo.rollout import AgenticRollout
    from vlm.grpo.trainer import GRPOConfig, GRPOTrainer

    grpo_raw = cfg.get("grpo", {})
    grpo_cfg = GRPOConfig.from_dict(grpo_raw)
    if grpo_cfg.ckpt_dir is None or grpo_cfg.ckpt_dir == "null":
        grpo_cfg.ckpt_dir = str(
            Path("/tmp") / f"vlm_grpo_{cfg.get('experiment_name', 'run')}"
        )

    rollout = AgenticRollout(
        student        = student,
        reward_fn      = reward_fn,
        group_size     = grpo_cfg.group_size,
        max_sam2_calls = grpo_cfg.group_size,   # reuse group_size field
        device         = device,
    )

    trainer = GRPOTrainer(
        student   = student,
        rollout   = rollout,
        reward_fn = reward_fn,
        cfg       = grpo_cfg,
        rank      = rank,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        trainer.load_checkpoint(args.resume)
        if rank == 0:
            log.info(f"Resumed from {args.resume} at step {trainer.step}")
    elif args.stage:
        # Manually set step to beginning of requested stage
        if args.stage == 2:
            trainer.step = grpo_cfg.stage1_steps
        elif args.stage == 3:
            trainer.step = grpo_cfg.stage3_start
        if rank == 0:
            log.info(f"Force-start stage {args.stage} at step {trainer.step}")

    # ── Eval only ─────────────────────────────────────────────────────────────
    if args.eval_only:
        if rank == 0:
            log.info("Eval-only mode: running one rollout batch for sanity check.")
        val_dm = _build_data_module(cfg, stage=3)
        val_loader = val_dm.loader()
        samples = next(iter(val_loader))
        for s in samples[:2]:
            rb = rollout.rollout(s, ultatron_backbone)
            if rank == 0:
                log.info(f"  sample={s.sample_id}  "
                         f"rewards={[f'{r:.3f}' for r in rb.rewards]}  "
                         f"n_tool={rb.n_tool_calls}")
        return

    # ── Train ─────────────────────────────────────────────────────────────────
    if rank == 0:
        log.info(f"Starting GRPO training for {grpo_cfg.total_steps} steps "
                 f"(G={grpo_cfg.group_size}, stage={start_stage}).")
    trainer.train(data_module, ultatron_backbone)


if __name__ == "__main__":
    main()
