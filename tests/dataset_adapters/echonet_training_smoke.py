"""
echonet_training_smoke.py  ·  EchoNet-Dynamic training pipeline smoke tests
=======================================================================

DEPRECATED: Prefer running:

    python -m tests.dataset_adapters.training_smoke

which exercises all four training phases across multiple datasets using
USFoundationDataModule. This file is kept only for very targeted EchoNet
debugging.

Usage (from project root):

    python -m tests.dataset_adapters.echonet_training_smoke

This script:
  1. Builds a tiny EchoNet-Dynamic manifest (first 32 TRAIN entries).
  2. Runs an image-SSL DINOv3-S branch smoke test on a few samples.
  3. Runs a video-SSL V-JEPA2-L branch smoke test on a few samples.

All computations use float32 on CPU by default to maximise portability.
It is intended as a manual debugging tool, NOT part of automated tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data.adapters.echonet import EchoNetDynamicAdapter
from data.schema.manifest import ManifestWriter, USManifestEntry, load_manifest
from data.pipeline.dataset import ImageSSLDataset, VideoSSLDataset
from data.pipeline.transforms import (
    ImageSSLTransformConfig,
    VideoSSLTransformConfig,
    MASK_STRATEGY_FREQ,
)
from models.branches.image_branch import ImageBranch
from models.branches.video_branch import build_video_branch
from models.registry import build_image_backbone


DEFAULT_ECHONET_ROOT = Path(
    "/capstor/store/cscs/swissai/a127/ultrasound/raw/cardiac/EchoNet-Dynamic"
)
DEFAULT_MANIFEST = Path("echonet_training_manifest.jsonl")


def get_echonet_root() -> Path:
    env = os.environ.get("US_ECHONET_ROOT")
    root = Path(env) if env else DEFAULT_ECHONET_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"EchoNet-Dynamic root not found at {root}. "
            "Set US_ECHONET_ROOT to override or verify the dataset path."
        )
    return root


def build_tiny_manifest(root: Path, manifest_path: Path = DEFAULT_MANIFEST) -> Path:
    adapter = EchoNetDynamicAdapter(root)
    entries: List[USManifestEntry] = []
    for e in adapter.iter_entries():
        if e.split == "train":
            entries.append(e)
        if len(entries) >= 32:
            break

    if not entries:
        raise RuntimeError("No TRAIN entries found in EchoNet-Dynamic.")

    with ManifestWriter(manifest_path) as w:
        for e in entries:
            w.write(e)

    print(f"[SMOKE] Wrote {len(entries)} TRAIN entries to {manifest_path}")
    return manifest_path


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = nn.functional.normalize(x, dim=-1)
    y = nn.functional.normalize(y, dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


def dinov3_image_smoke(entries: List[USManifestEntry], device: str = "cpu") -> None:
    print("[SMOKE] === DINOv3-S image branch (EchoNet frames) ===")

    cfg = ImageSSLTransformConfig(
        n_global_crops=2,
        n_local_crops=2,
        max_global_crop_px=112,
        min_crop_px=32,
        mask_strategy=MASK_STRATEGY_FREQ,
    )
    ds = ImageSSLDataset(entries, cfg=cfg)
    ds_small = Subset(ds, range(min(4, len(ds))))

    loader = DataLoader(ds_small, batch_size=2, shuffle=False)

    # Build DINOv3-S backbone and image branch
    dtype = torch.float32
    student = build_image_backbone("dinov3_s", dtype=dtype)
    teacher = build_image_backbone("dinov3_s", dtype=dtype)
    branch = ImageBranch(student=student, teacher=teacher).to(device=device, dtype=dtype)

    opt = torch.optim.AdamW(branch.parameters(), lr=1e-4)

    branch.train()
    n_batches = 0
    for batch in loader:
        n_batches += 1
        if n_batches > 2:
            break

        global_crops = batch["global_crops"]  # (B, 2, C, H, W)
        g0 = global_crops[:, 0].to(device=device, dtype=dtype)
        g1 = global_crops[:, 1].to(device=device, dtype=dtype)

        opt.zero_grad()
        s_out = branch.forward_student(g0)
        t_out = branch.forward_teacher(g1)

        s_cls = s_out["cls"]          # (B, D)
        t_cls = t_out["cls"]          # (B, D)
        loss = cosine_loss(s_cls, t_cls)
        loss.backward()
        opt.step()

        print(
            f"[SMOKE][DINOv3] batch={n_batches} "
            f"student_cls.shape={tuple(s_cls.shape)} "
            f"loss={loss.item():.4f}"
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss in DINOv3 image smoke test.")

    print("[SMOKE] DINOv3-S image branch PASS")


def vjepa2_video_smoke(entries: List[USManifestEntry], device: str = "cpu") -> None:
    print("[SMOKE] === V-JEPA2-L video branch (EchoNet clips) ===")

    cfg = VideoSSLTransformConfig(
        n_frames=8,
        max_crop_px=112,
        min_crop_px=32,
        mask_strategy=MASK_STRATEGY_FREQ,
    )
    ds = VideoSSLDataset(entries, cfg=cfg)
    ds_small = Subset(ds, range(min(4, len(ds))))
    loader = DataLoader(ds_small, batch_size=2, shuffle=False)

    dtype = torch.float32
    # build_video_branch already moves to the requested device and dtype
    branch = build_video_branch(dtype=dtype, device=device)

    opt = torch.optim.AdamW(branch.parameters(), lr=1e-4)
    branch.train()

    n_batches = 0
    for batch in loader:
        n_batches += 1
        if n_batches > 2:
            break

        full = batch["full_clip"].to(device=device, dtype=dtype)         # (B, T, C, H, W)
        vis = batch["visible_clip"].to(device=device, dtype=dtype)
        tube_mask = batch["tube_mask"].to(device=device)
        padding_mask = batch.get("padding_mask")
        if padding_mask is not None:
            padding_mask = padding_mask.to(device=device)
        valid_frames = batch.get("valid_frames")
        if valid_frames is not None:
            valid_frames = valid_frames.to(device=device)

        opt.zero_grad()
        t_out = branch.forward_teacher(full, padding_mask=padding_mask, valid_frames=valid_frames)
        s_out = branch.forward_student(
            vis,
            tube_mask=tube_mask,
            padding_mask=padding_mask,
            valid_frames=valid_frames,
        )

        clip_t = t_out["clip_cls"]
        clip_s = s_out["clip_cls"]
        loss = cosine_loss(clip_s, clip_t)
        loss.backward()
        opt.step()

        print(
            f"[SMOKE][VJEPA2] batch={n_batches} "
            f"clip_cls.shape={tuple(clip_s.shape)} "
            f"loss={loss.item():.4f}"
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss in V-JEPA2 video smoke test.")

    print("[SMOKE] V-JEPA2-L video branch PASS")


def main() -> None:
    device = os.environ.get("US_SMOKE_DEVICE", "cpu")
    root = get_echonet_root()

    print(f"[SMOKE] Using EchoNet-Dynamic root: {root}")
    print(f"[SMOKE] Using device: {device}")

    manifest_path = DEFAULT_MANIFEST
    if not manifest_path.exists():
        manifest_path = build_tiny_manifest(root, manifest_path)
    entries = load_manifest(manifest_path, split="train")
    entries = entries[:32]
    print(f"[SMOKE] Loaded {len(entries)} TRAIN entries from manifest.")

    dinov3_image_smoke(entries, device=device)
    vjepa2_video_smoke(entries, device=device)

    print("[SMOKE] All EchoNet-Dynamic training smoke tests PASS.")


if __name__ == "__main__":
    main()

