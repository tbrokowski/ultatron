"""
collators.py  ·  Batch collation for image and video SSL streams
================================================================

Native-resolution collation
----------------------------
Crops are extracted at native pixel resolution (no forced resize),
Both collators variable crop sizes by padding to the per-batch maximum and returning attention masks.

ImageSSLCollator
  global_crops   : (B, n_global, C, H_max, W_max) — zero-padded, C inferred from input
  global_pmasks  : (B, n_global, ph_max, pw_max) bool — True = real patch
  local_crops    : (B, n_local,  C, h_max, w_max)
  local_pmasks   : (B, n_local,  ph_max, pw_max) bool
  patch_masks    : (B, ph_max, pw_max) bool — freq-energy mask (crop[0])
                   Padding positions are False (= unmasked / ignored by loss).

VideoSSLCollator
  full_clips     : (B, T_max, C, H_max, W_max)
  visible_clips  : (B, T_max, C, H_max, W_max)
  tube_masks     : (B, T_max, ph_max, pw_max)
  padding_masks  : (B, ph_max, pw_max) bool — True = real patch
  valid_frames   : (B, T_max) bool

Model contract
--------------
The ViT forward() must accept an optional padding_mask argument and zero-out
(or skip via attention bias) tokens where padding_mask is False.  The standard
approach is to add -inf to the attention logits at padding positions before
the softmax 

patch_masks (frequency-energy mask) and padding_masks (padding indicator) are
separate tensors.  The loss should only be computed at positions where BOTH
  padding_mask  == True   (real image content)
  patch_mask    == True   (high-energy-loss patch selected by freq masking)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ── helpers ───────────────────────────────────────────────────────────────────

def _pad_crop_to(crop: Tensor, target_H: int, target_W: int) -> Tensor:
    """Pad a (C, H, W) crop to (C, target_H, target_W) with zeros on right/bottom."""
    _, H, W = crop.shape
    return F.pad(crop, (0, target_W - W, 0, target_H - H), value=0.0)


def _pad_pmask_to(pmask: Tensor, target_ph: int, target_pw: int) -> Tensor:
    """Pad a (ph, pw) bool mask to (target_ph, target_pw) with False on right/bottom."""
    ph, pw = pmask.shape
    out = torch.zeros(target_ph, target_pw, dtype=torch.bool)
    out[:ph, :pw] = pmask
    return out


def _pad_freq_mask_to(fmask: Tensor, target_ph: int, target_pw: int) -> Tensor:
    """
    Pad a (ph, pw) frequency-energy bool mask with False.
    Padding positions are never selected for the loss.
    """
    ph, pw = fmask.shape
    out = torch.zeros(target_ph, target_pw, dtype=torch.bool)
    out[:ph, :pw] = fmask
    return out


# ── Image SSL collator ────────────────────────────────────────────────────────

class ImageSSLCollator:
    """
    Collates variable-resolution native-crop samples into padded batch tensors.

    Batch keys produced
    -------------------
    global_crops    (B, n_g, C, H_max, W_max)   float32
    global_pmasks   (B, n_g, ph_max, pw_max)     bool   True=real patch
    local_crops     (B, n_l, C, h_max, w_max)   float32
    local_pmasks    (B, n_l, ph_max_l, pw_max_l) bool
    patch_masks     (B, ph_max, pw_max)           bool   freq-energy mask
    seg_masks       (B, 1, H, W) or None
    cls_labels      (B,) long
    tiers           (B,) long
    is_promptable   (B,) bool
    dataset_ids     list[str]
    anatomy_families list[str]
    sample_ids      list[str]
    task_types      list[str]
    """

    def __call__(self, samples: List[Dict]) -> Dict[str, Any]:
        patch_size = 16   # must match transform patch_size

        # ── Determine per-batch max dimensions ────────────────────────────────
        n_global = len(samples[0]["global_crops"])
        n_local  = len(samples[0]["local_crops"])

        # Global: find max H and W across all samples and all global crops
        max_gH = max_gW = 0
        for s in samples:
            for crop in s["global_crops"]:
                _, H, W = crop.shape
                max_gH  = max(max_gH, H)
                max_gW  = max(max_gW, W)
        max_gph = max_gH // patch_size
        max_gpw = max_gW // patch_size

        # Local
        max_lH = max_lW = 0
        for s in samples:
            for crop in s["local_crops"]:
                _, H, W = crop.shape
                max_lH  = max(max_lH, H)
                max_lW  = max(max_lW, W)
        max_lph = max_lH // patch_size
        max_lpw = max_lW // patch_size

        B = len(samples)
        C = samples[0]["global_crops"][0].shape[0]   # infer channels from first crop

        global_crops  = torch.zeros(B, n_global, C, max_gH, max_gW)
        global_pmasks = torch.zeros(B, n_global, max_gph, max_gpw, dtype=torch.bool)
        local_crops   = torch.zeros(B, n_local,  C, max_lH, max_lW)
        local_pmasks  = torch.zeros(B, n_local,  max_lph, max_lpw, dtype=torch.bool)
        patch_masks   = torch.zeros(B, max_gph, max_gpw, dtype=torch.bool)

        for i, s in enumerate(samples):
            for j, (crop, pm) in enumerate(zip(s["global_crops"], s["global_pmasks"])):
                global_crops[i, j]  = _pad_crop_to(crop, max_gH, max_gW)
                global_pmasks[i, j] = _pad_pmask_to(pm, max_gph, max_gpw)

            for j, (crop, pm) in enumerate(zip(s["local_crops"], s["local_pmasks"])):
                local_crops[i, j]  = _pad_crop_to(crop, max_lH, max_lW)
                local_pmasks[i, j] = _pad_pmask_to(pm, max_lph, max_lpw)

            patch_masks[i] = _pad_freq_mask_to(s["patch_mask"], max_gph, max_gpw)

        # ── Seg masks (optional, pad to max spatial size) ─────────────────────
        raw_segs = [s.get("seg_mask") for s in samples]
        if any(m is not None for m in raw_segs):
            valid = [m for m in raw_segs if m is not None]
            _, _, sH, sW = valid[0].shape
            padded_segs  = []
            for m in raw_segs:
                if m is None:
                    padded_segs.append(torch.zeros(1, 1, sH, sW))
                else:
                    padded_segs.append(m.unsqueeze(0) if m.ndim == 3 else m.unsqueeze(0))
            seg_masks = torch.cat(padded_segs, dim=0)
        else:
            seg_masks = None

        return {
            "global_crops":     global_crops,     # B × n_g × 1 × H_max × W_max
            "global_pmasks":    global_pmasks,     # B × n_g × ph_max × pw_max
            "local_crops":      local_crops,       # B × n_l × 1 × h_max × w_max
            "local_pmasks":     local_pmasks,      # B × n_l × ph_max_l × pw_max_l
            "patch_masks":      patch_masks,       # B × ph_max × pw_max  (freq mask)
            "dataset_ids":      [s["dataset_id"]       for s in samples],
            "anatomy_families": [s["anatomy_family"]   for s in samples],
            "tiers":            torch.tensor([s["tier"]          for s in samples], dtype=torch.long),
            "sample_ids":       [s["sample_id"]        for s in samples],
            "seg_masks":        seg_masks,
            "cls_labels":       torch.tensor([s.get("cls_label", -1) for s in samples], dtype=torch.long),
            "task_types":       [s["task_type"]        for s in samples],
            "is_promptable":    torch.tensor([s.get("is_promptable", False) for s in samples]),
        }


# ── Video SSL collator ────────────────────────────────────────────────────────

class VideoSSLCollator:
    """
    Collates variable-resolution, variable-length video samples.

    Batch keys produced
    -------------------
    full_clips      (B, T_max, C, H_max, W_max)   C inferred from input frames
    visible_clips   (B, T_max, C, H_max, W_max)
    tube_masks      (B, T_max, ph_max, pw_max)  bool
    padding_masks   (B, ph_max, pw_max)          bool  True=real patch
    valid_frames    (B, T_max)                   bool  True=real frame
    """

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, samples: List[Dict]) -> Dict[str, Any]:
        patch_size = 16
        B     = len(samples)
        max_T = max(s["n_frames"] for s in samples)

        # Find max spatial dimensions across batch
        max_H = max_W = 0
        for s in samples:
            _, _, H, W = s["full_clip"].shape
            max_H = max(max_H, H)
            max_W = max(max_W, W)
        max_ph = max_H // patch_size
        max_pw = max_W // patch_size

        C = samples[0]["full_clip"].shape[1]

        full_clips    = torch.full((B, max_T, C, max_H, max_W), self.pad_value)
        visible_clips = torch.full((B, max_T, C, max_H, max_W), self.pad_value)
        tube_masks    = torch.zeros(B, max_T, max_ph, max_pw, dtype=torch.bool)
        padding_masks = torch.zeros(B, max_ph, max_pw, dtype=torch.bool)
        valid_frames  = torch.zeros(B, max_T, dtype=torch.bool)

        for i, s in enumerate(samples):
            T = s["n_frames"]
            _, _, H, W = s["full_clip"].shape
            ph, pw = H // patch_size, W // patch_size

            # Spatial pad each frame
            for t in range(T):
                full_clips[i, t, :, :H, :W]    = s["full_clip"][t]
                visible_clips[i, t, :, :H, :W] = s["visible_clip"][t]
            tube_masks[i, :T, :ph, :pw] = s["tube_mask"]
            valid_frames[i, :T]         = True

            # Padding mask from sample (or infer from clip dimensions)
            if "padding_mask" in s:
                padding_masks[i, :ph, :pw] = s["padding_mask"]
            else:
                padding_masks[i, :ph, :pw] = True   # whole frame is real

        return {
            "full_clips":       full_clips,
            "visible_clips":    visible_clips,
            "tube_masks":       tube_masks,
            "padding_masks":    padding_masks,
            "valid_frames":     valid_frames,
            "dataset_ids":      [s["dataset_id"]       for s in samples],
            "anatomy_families": [s["anatomy_family"]   for s in samples],
            "tiers":            torch.tensor([s["tier"]     for s in samples], dtype=torch.long),
            "sample_ids":       [s["sample_id"]        for s in samples],
            "fps":              torch.tensor([s.get("fps", 25.0)   for s in samples]),
            "is_cine":          torch.tensor([s.get("is_cine", False) for s in samples]),
            "task_types":       [s["task_type"]        for s in samples],
        }


# ── Dual-stream batch container ───────────────────────────────────────────────

@dataclass
class DualStreamBatch:
    image_batch: Dict[str, Any]
    video_batch: Dict[str, Any]

    @property
    def device(self):
        return self.image_batch["global_crops"].device

    def to(self, device) -> "DualStreamBatch":
        def _move(d):
            return {k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in d.items()}
        return DualStreamBatch(_move(self.image_batch), _move(self.video_batch))
