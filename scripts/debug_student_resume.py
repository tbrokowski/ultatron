#!/usr/bin/env python3
"""
Single-GPU diagnostic for student pretrain resume NaNs.

Runs the same load path as student_training_smoke, then executes one stage-1
forward pass with per-tensor finiteness checks.

Usage (inside EDF container, 1 GPU):
  export US_STUDENT_CONFIG=configs/student/student_pretrain_pilot.yaml
  export US_STUDENT_MODE=pretrain
  export PYTHONPATH=/users/tbrokowski/Ultatron
  python3 scripts/debug_student_resume.py \\
      --ckpt /capstor/store/cscs/swissai/a127/ultrasound/checkpoints/StudentPretrainPilot/step_01500.pt \\
      --step 1501
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

_ROOT = Path(__file__).resolve().parents[1]
_scripts = str(Path(__file__).resolve().parent)
# Running as scripts/foo.py puts scripts/ on sys.path and shadows the train/ package.
sys.path = [p for p in sys.path if p not in ("", _scripts)]
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("US_STUDENT_MODE", "pretrain")
os.environ.setdefault(
    "US_STUDENT_CONFIG",
    str(_ROOT / "configs" / "student" / "student_pretrain_pilot.yaml"),
)

from data.pipeline.student_datamodule import StudentDataModule
from data.schema.manifest import load_manifest
from models.branches.shared import PrototypeHead
from models.student.student_config import (
    StudentModelConfig,
    build_fusion_target_builder,
    build_student_encoder,
)
from models.student.teacher_wrappers import FrozenDINOTeacher
from train.alp import HardnessFeedback, configure_alp_cache
from train.student_phase_steps import student_stage1_step

# Reuse training helpers without importing the full smoke module (avoids DDP init).
from tests.dataset_adapters.student_training_smoke import (  # noqa: E402
    _build_datamodules,
    _load_config,
    _resolve_hf_cache,
    _student_model_config,
    _to_dev,
)


def _tensor_stats(name: str, t: torch.Tensor | None) -> str:
    if t is None:
        return f"{name}: None"
    if not torch.is_tensor(t):
        return f"{name}: type={type(t).__name__}"
    finite = torch.isfinite(t)
    n_nan = (~finite).sum().item()
    return (
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"finite={finite.all().item()} nan/inf={n_nan} "
        f"min={t[finite].min().item() if finite.any() else 'n/a':.4g} "
        f"max={t[finite].max().item() if finite.any() else 'n/a':.4g}"
    )


def _check_dict(prefix: str, d: dict) -> list[str]:
    lines: list[str] = []
    for k, v in d.items():
        if torch.is_tensor(v):
            lines.append(_tensor_stats(f"{prefix}.{k}", v))
        elif isinstance(v, float) and not math.isfinite(v):
            lines.append(f"{prefix}.{k}: non-finite float {v}")
    return lines


def _build_bundle(cfg: dict, device: str):
    model_cfg = _student_model_config(cfg)
    dtype = model_cfg.torch_dtype()
    lam = cfg.get("loss_weights", {})

    student = build_student_encoder(model_cfg, device=device).to(dtype=dtype)
    dino = FrozenDINOTeacher(
        backbone_key=model_cfg.dino_teacher_key,
        align_dim=model_cfg.align_dim,
        dtype=dtype,
        hf_cache_dir=model_cfg.hiera_hf_cache_dir,
    ).to(device=device, dtype=dtype)
    ema_student = copy.deepcopy(student)
    ema_student.is_ema_target = True
    for p in ema_student.parameters():
        p.requires_grad_(False)
    ema_student.eval()
    ema_student.to(device=device, dtype=dtype)
    proto = PrototypeHead(
        embed_dim=model_cfg.align_dim,
        n_prototypes=model_cfg.n_prototypes,
    ).to(device=device)
    fusion = build_fusion_target_builder(model_cfg, device=device)
    return student, ema_student, dino, proto, fusion, lam, dtype, model_cfg


def _load_ckpt(
    path: Path,
    device: str,
    student,
    ema_student,
    proto,
    fusion,
) -> dict:
    ckpt = torch.load(path, map_location=device)
    student.load_state_dict(ckpt["student"], strict=True)
    ema_student.load_state_dict(ckpt["ema_student"], strict=True)
    proto.load_state_dict(ckpt["proto"], strict=True)
    fusion.load_state_dict(ckpt["fusion"], strict=True)
    return ckpt


def _stage1_forward_trace(batch: dict, student, ema_student, dino, proto, lam, step: int, dtype, use_amp: bool):
    from train.student_phase_steps import (
        _dino_patch_masks,
        _ensure_image_batch,
        _lam_ema_eff,
        _student_crop_pmask,
        _student_f1_grid_pmask,
        _student_global,
        _student_patches,
        _teacher_image_pmask,
    )
    from models.losses.student_losses import (
        img_ema_global_loss,
        img_ema_patch_loss,
        img_global_distill_loss,
        img_masked_semantic_loss,
        img_patch_distill_loss,
        img_proto_loss,
    )

    lines: list[str] = []
    batch = _ensure_image_batch(batch)
    s_crop = batch["global_crops"][:, 0]
    t_crop = batch["global_crops"][:, 1]
    pmask_s = _student_crop_pmask(batch, 0)
    pmask_t = _student_crop_pmask(batch, 1)
    pmask_dino = _teacher_image_pmask(batch, pmask_t)
    patch_masks = batch.get("patch_masks")
    x_s = s_crop.unsqueeze(1)
    x_t = t_crop.unsqueeze(1)

    lines.append(_tensor_stats("batch.global_crops", batch["global_crops"]))

    ctx = autocast(dtype=dtype) if use_amp else torch.enable_grad()
    with torch.no_grad(), ctx:
        t_img = dino(t_crop, padding_mask=pmask_dino, return_attention=False)
        t_ema_out = ema_student(x_t, padding_mask=pmask_t)
    lines.extend(_check_dict("dino", t_img))
    lines.extend(_check_dict("ema", t_ema_out))

    with ctx:
        s_out = student(x_s, padding_mask=pmask_s)
    lines.extend(_check_dict("student", s_out))

    s_global = _student_global(s_out)
    s_patches = _student_patches(s_out, t=0)
    pmask_f1 = _student_f1_grid_pmask(s_out)
    unmasked_flat, masked_flat = _dino_patch_masks(s_out, patch_masks)

    t_cls_proj = t_img["cls_proj"]
    t_patch_proj = t_img["patch_proj"]
    t_ema_global = t_ema_out["global"]
    t_ema_f1 = t_ema_out["F1"][:, 0]

    losses = {
        "L_global": img_global_distill_loss(s_global, t_cls_proj),
        "L_patch": img_patch_distill_loss(s_patches, t_patch_proj, mask=unmasked_flat),
        "L_masked": (
            img_masked_semantic_loss(s_patches, t_patch_proj, masked_flat, pmask_f1)
            if masked_flat is not None
            else s_global.new_tensor(0.0)
        ),
        "L_proto": img_proto_loss(
            s_global.unsqueeze(1),
            t_cls_proj.detach().unsqueeze(1),
            proto.prototypes,
        ),
        "L_ema_global": img_ema_global_loss(s_out["global"], t_ema_global),
        "L_ema_patch": img_ema_patch_loss(
            s_out["F1"][:, 0], t_ema_f1, masked_positions=patch_masks, padding_mask=pmask_f1
        ),
    }
    lam_ema_eff = _lam_ema_eff(lam, step)
    total = (
        lam.get("lam_global", 1.0) * losses["L_global"]
        + lam.get("lam_patch", 1.0) * losses["L_patch"]
        + lam.get("lam_masked", 1.0) * losses["L_masked"]
        + lam.get("lam_proto", 0.5) * losses["L_proto"]
        + lam_ema_eff * (losses["L_ema_global"] + losses["L_ema_patch"])
    )
    losses["total"] = total

    for name, val in losses.items():
        if torch.is_tensor(val):
            lines.append(_tensor_stats(f"loss.{name}", val))
        else:
            lines.append(f"loss.{name}: {val}")

    return lines, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug student resume NaNs")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(
            "/capstor/store/cscs/swissai/a127/ultrasound/checkpoints/StudentPretrainPilot/step_01500.pt"
        ),
    )
    parser.add_argument("--step", type=int, default=1501)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-load", action="store_true", help="Skip checkpoint load (fresh weights)")
    parser.add_argument("--batch-index", type=int, default=0, help="Skip N batches from loader")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    cfg = _load_config()
    _resolve_hf_cache(cfg)
    device = args.device
    use_amp = bool(cfg.get("training", {}).get("use_amp", True))

    manifest_path = Path(cfg["manifest"]["path"])
    if not manifest_path.is_absolute():
        manifest_path = _ROOT / manifest_path
    print(f"manifest: {manifest_path}")
    print(f"ckpt: {args.ckpt}  load={not args.no_load}  step={args.step}  amp={use_amp}")

    student_dm, alp_feedback = _build_datamodules(cfg, manifest_path)
    student, ema_student, dino, proto, fusion, lam, dtype, model_cfg = _build_bundle(cfg, device)

    if not args.no_load:
        ckpt = _load_ckpt(args.ckpt, device, student, ema_student, proto, fusion)
        print(
            f"loaded ckpt step={ckpt['step']} n_finite={ckpt.get('n_finite')} "
            f"n_nonfinite={ckpt.get('n_nonfinite')}"
        )
    else:
        print("using freshly initialized student (no checkpoint)")

    student.train()
    student_dm.set_stage(1)
    student_dm.set_epoch(args.step)
    student_dm.update_curriculum(args.step)

    batch = None
    for i in range(args.batch_index + 1):
        batch = student_dm.next_batch()
    assert batch is not None
    batch = _to_dev(batch, device)

    print("\n=== Forward trace ===")
    trace_lines, losses = _stage1_forward_trace(
        batch, student, ema_student, dino, proto, lam, args.step, dtype, use_amp
    )
    for line in trace_lines:
        print(line)

    print("\n=== student_stage1_step (full) ===")
    with autocast(dtype=dtype) if use_amp else torch.enable_grad():
        out = student_stage1_step(
            batch, student, ema_student, dino, proto, lam,
            global_step=args.step, alp_feedback=alp_feedback,
        )
    for k, v in out.items():
        if k.startswith("loss"):
            if torch.is_tensor(v):
                print(_tensor_stats(f"out.{k}", v))
            else:
                ok = math.isfinite(float(v))
                print(f"out.{k}: {v}  finite={ok}")

    bad = [k for k, v in out.items() if k.startswith("loss") and not (
        (torch.is_tensor(v) and torch.isfinite(v).all()) or (isinstance(v, (int, float)) and math.isfinite(float(v)))
    )]
    if bad:
        print(f"\nRESULT: NON-FINITE in {bad}")
        sys.exit(1)
    print("\nRESULT: all losses finite on this batch")


if __name__ == "__main__":
    main()
