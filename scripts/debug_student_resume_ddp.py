#!/usr/bin/env python3
"""2-GPU DDP variant of debug_student_resume.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast

_ROOT = Path(__file__).resolve().parents[1]
_scripts = str(Path(__file__).resolve().parent)
sys.path = [p for p in sys.path if p not in ("", _scripts)]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("US_STUDENT_MODE", "pretrain")
os.environ.setdefault(
    "US_STUDENT_CONFIG",
    str(_ROOT / "configs" / "student" / "student_pretrain_pilot.yaml"),
)

from tests.dataset_adapters.student_training_smoke import (  # noqa: E402
    StudentSmokeTrainer,
    _build_datamodules,
    _init_dist,
    _is_main,
    _load_config,
    _resolve_hf_cache,
    _to_dev,
    _unwrap,
)
from train.student_phase_steps import student_stage1_step  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(
            "/capstor/store/cscs/swissai/a127/ultrasound/checkpoints/StudentPretrainPilot/step_01500.pt"
        ),
    )
    parser.add_argument("--step", type=int, default=1501)
    parser.add_argument("--no-load", action="store_true")
    parser.add_argument("--no-alp", action="store_true")
    args = parser.parse_args()

    _init_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    cfg = _load_config()
    _resolve_hf_cache(cfg)
    manifest = Path(cfg["manifest"]["path"])
    if not manifest.is_absolute():
        manifest = _ROOT / manifest

    student_dm, alp_feedback = _build_datamodules(cfg, manifest)
    trainer = StudentSmokeTrainer(cfg, device, alp_feedback=alp_feedback)
    if not args.no_load:
        trainer.load_checkpoint(args.ckpt)
    else:
        if _is_main():
            print("SKIP checkpoint load", flush=True)

    feedback = None if args.no_alp else alp_feedback

    student_dm.set_stage(1)
    student_dm.set_epoch(args.step)
    student_dm.update_curriculum(args.step)
    if dist.is_initialized():
        dist.barrier()

    batch = _to_dev(student_dm.next_batch(), device)
    lam = cfg.get("loss_weights", {})
    dtype = trainer.dtype
    use_amp = trainer.use_amp

    with autocast(dtype=dtype) if use_amp else torch.enable_grad():
        out = student_stage1_step(
            batch,
            _unwrap(trainer.student),
            trainer.ema_student,
            trainer.dino,
            _unwrap(trainer.proto),
            lam,
            global_step=args.step,
            alp_feedback=feedback,
        )

    loss = out["loss"]
    local_finite = torch.isfinite(loss).all().item()
    flag = torch.tensor([1 if local_finite else 0], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    global_finite = flag.item() != 0

    parts = []
    for k, v in out.items():
        if not k.startswith("loss"):
            continue
        if torch.is_tensor(v):
            ok = torch.isfinite(v).all().item()
        else:
            ok = __import__("math").isfinite(float(v))
        if not ok:
            parts.append(k)

    loss_val = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
    print(
        f"rank={dist.get_rank()} local_finite={local_finite} global_finite={global_finite} "
        f"loss={loss_val:.4f} bad={parts}",
        flush=True,
    )

    dist.barrier()
    if _is_main():
        print("DONE", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
