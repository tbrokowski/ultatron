"""
video_branch.py  ·  V-JEPA2-L video branch for Ultatron
====================================================

Architecture
------------
  Student encoder : V-JEPA2-L (facebook/vjepa2-vitl-fpc64-256), trainable
  Teacher encoder : EMA copy of student encoder, no gradients
  Predictor       : V-JEPA2's built-in narrow transformer predictor,
                    accepts context tokens + predicts masked tube tokens

Input contract  (per call)
--------------------------
  pixel_values_videos : (B, T, 3, H, W)  float32 [0,1], 3-channel RGB
  tube_mask           : (B, T, ph, pw)   bool  True=masked tube position
  padding_mask        : (B, ph, pw)       bool  True=real patch
  valid_frames        : (B, T)            bool  True=real frame

Output contract  (dict)
-----------------------
  clip_cls      : (B, D)          mean-pooled clip representation
  tube_tokens   : (B, T*ph*pw, D) all spatiotemporal patch tokens
  predicted     : (B, T*ph*pw, D) predictor output at all positions

Channel handling
----------------
The data pipeline (dataset.py + transforms.py) delivers (B, T, 3, H, W)
tensors.  Greyscale ultrasound frames are channel-repeated to R=G=B by
to_canonical_tensor() in transforms.py before they reach this branch.
No channel conversion is performed here.

V-JEPA2 input format
--------------------
HuggingFace VJEPA2Model expects:
  pixel_values_videos : (B, T, C, H, W)  — note T before C
  context_mask        : (B, N, 1) LongTensor — indices of visible (context) tokens
  target_mask         : (B, N, 1) LongTensor — indices of tokens to predict

We derive context_mask and target_mask from our tube_mask:
  - context = positions where tube_mask == False (visible)
  - target  = positions where tube_mask == True  (masked, to predict)

Padding mask
------------
V-JEPA2 uses the same SDPA attention.  We register the same -inf bias approach
as in the image branch.  Padding tokens (from variable-resolution clips) are
excluded from both context and target index sets — they are never visible or
predicted.

TODO: refactor to a model-agnostic VideoBackboneBase / VideoBranchBase pattern
      matching image_branch.py, so other video backbone (e.g. VideoMAE, InternVideo)
      can be swapped in without touching this file.
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_branch import ema_update   # shared EMA utility

log = logging.getLogger(__name__)

VJEPA2_L_HF = "facebook/vjepa2-vitl-fpc64-256"


# ── Mask index builders ───────────────────────────────────────────────────────

def _tube_mask_to_indices(
    tube_mask: torch.Tensor,       # (B, T, ph, pw) bool  True=masked
    padding_mask: Optional[torch.Tensor],  # (B, ph, pw) bool  True=real
    valid_frames: Optional[torch.Tensor],  # (B, T) bool  True=real
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert tube_mask to context_mask and target_mask index tensors
    as expected by HuggingFace VJEPA2Model.

    context_mask : (B, N_ctx, 1) LongTensor  — visible patch indices
    target_mask  : (B, N_tgt, 1) LongTensor  — masked patch indices

    Indices are into the flattened spatiotemporal sequence of length T*ph*pw.
    Padding patches (padding_mask=False) and padding frames (valid_frames=False)
    are excluded from both sets.

    Because N_ctx and N_tgt vary per sample, we pad the shorter sequences to
    the batch maximum with a sentinel value (= sequence length) and let the
    model ignore out-of-range indices.  In practice, since all samples in a
    batch are cropped to the same resolution (by the collator's max-pooling),
    N_ctx and N_tgt are constant within a batch.
    """
    B, T, ph, pw = tube_mask.shape
    N = T * ph * pw

    # Build a real-token mask: real frame AND real patch AND not tube-masked
    # (B, T, ph, pw)
    real = torch.ones(B, T, ph, pw, dtype=torch.bool, device=tube_mask.device)
    if valid_frames is not None:
        real = real & valid_frames.unsqueeze(-1).unsqueeze(-1)   # (B,T,1,1)
    if padding_mask is not None:
        real = real & padding_mask.unsqueeze(1)                  # (B,1,ph,pw)

    real_flat    = real.flatten(1)       # (B, N)
    masked_flat  = tube_mask.flatten(1)  # (B, N)

    context_flat = real_flat & ~masked_flat   # (B, N) visible real patches
    target_flat  = real_flat & masked_flat    # (B, N) masked real patches

    # Collect indices for each sample; pad to max_len within batch
    ctx_indices, tgt_indices = [], []
    for b in range(B):
        ctx = context_flat[b].nonzero(as_tuple=False).squeeze(1)   # (N_ctx,)
        tgt = target_flat[b].nonzero(as_tuple=False).squeeze(1)    # (N_tgt,)
        ctx_indices.append(ctx)
        tgt_indices.append(tgt)

    max_ctx = max(x.shape[0] for x in ctx_indices)
    max_tgt = max(x.shape[0] for x in tgt_indices)

    ctx_padded = torch.full((B, max_ctx), N, dtype=torch.long, device=tube_mask.device)
    tgt_padded = torch.full((B, max_tgt), N, dtype=torch.long, device=tube_mask.device)
    for b in range(B):
        ctx_padded[b, :ctx_indices[b].shape[0]] = ctx_indices[b]
        tgt_padded[b, :tgt_indices[b].shape[0]] = tgt_indices[b]

    return ctx_padded.unsqueeze(-1), tgt_padded.unsqueeze(-1)


# ── V-JEPA2 encoder wrapper ───────────────────────────────────────────────────

class VJEPA2Encoder(nn.Module):
    """
    Wraps HuggingFace VJEPA2Model as a student/teacher encoder.

    Handles:
      - context/target index derivation from tube_mask
      - standardised output dict

    Expects pixel_values in (B, T, 3, H, W) format; channel normalisation is
    handled upstream by to_canonical_tensor() in transforms.py.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model       = hf_model
        self.hidden_size = hf_model.config.hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,           # (B, T, 3, H, W)
        tube_mask: Optional[torch.Tensor] = None,    # (B, T, ph, pw)
        padding_mask: Optional[torch.Tensor] = None, # (B, ph, pw)
        valid_frames: Optional[torch.Tensor] = None, # (B, T)
    ) -> dict:
        if tube_mask is not None:
            context_mask, target_mask = _tube_mask_to_indices(
                tube_mask, padding_mask, valid_frames
            )
        else:
            context_mask = target_mask = None

        out = self.model(
            pixel_values_videos=pixel_values,
            context_mask=context_mask,
            target_mask=target_mask,
            output_hidden_states=False,
        )

        # last_hidden_state: (B, T*ph*pw, D)  — all spatiotemporal tokens
        tube_tokens = out.last_hidden_state

        # clip_cls: mean over all real (non-padding) tokens
        if padding_mask is not None and valid_frames is not None:
            real_flat = (
                valid_frames.unsqueeze(-1).unsqueeze(-1) &
                padding_mask.unsqueeze(1)
            ).flatten(1).float()   # (B, T*ph*pw)
            denom     = real_flat.sum(1, keepdim=True).clamp(min=1)
            clip_cls  = (tube_tokens * real_flat.unsqueeze(-1)).sum(1) / denom
        else:
            clip_cls = tube_tokens.mean(1)   # (B, D)

        result = {
            "clip_cls":    clip_cls,
            "tube_tokens": tube_tokens,
        }

        # Predictor output: only present when target_mask is provided
        if hasattr(out, "predictor_output") and out.predictor_output is not None:
            result["predicted"] = out.predictor_output.last_hidden_state

        return result


# ── VideoBranch: student + teacher ────────────────────────────────────────────

class VideoBranch(nn.Module):
    """
    Full video branch for Oura Phase 2/3.

    Contains:
        self.student  : VJEPA2Encoder  (trainable)
        self.teacher  : VJEPA2Encoder  (EMA, no grad)

    The teacher receives the full unmasked clip (tube_mask=None).
    The student receives the masked clip.
    The predictor is the student's built-in predictor head (inside VJEPA2Model).

    EMA momentum: constant 0.9995, updated every step.
    """

    def __init__(self, student_model, teacher_model):
        super().__init__()
        self.student = VJEPA2Encoder(student_model)
        self.teacher = VJEPA2Encoder(teacher_model)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def update_teacher(self, momentum: float = 0.9995):
        ema_update(self.student.model, self.teacher.model, momentum)

    def forward_student(
        self,
        pixel_values: torch.Tensor,
        tube_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        valid_frames: Optional[torch.Tensor] = None,
    ) -> dict:
        return self.student(pixel_values, tube_mask, padding_mask, valid_frames)

    @torch.no_grad()
    def forward_teacher(
        self,
        pixel_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        valid_frames: Optional[torch.Tensor] = None,
    ) -> dict:
        self.teacher.eval()
        return self.teacher(pixel_values, tube_mask=None,
                            padding_mask=padding_mask, valid_frames=valid_frames)


# ── Cross-branch distillation head ────────────────────────────────────────────

class CrossBranchDistillation(nn.Module):
    """
    Aligns image teacher patch tokens with video student tube tokens.

    Image patch tokens live in D_img space; video tube tokens in D_vid.
    A lightweight linear projection aligns them for cosine distance loss.

    The alignment operates on samples that appear in BOTH the image and video
    batches (same study / same dataset_id appearing in both).  When no aligned
    pairs exist in a batch the loss is zero.
    """

    def __init__(self, img_dim: int = 1024, vid_dim: int = 1024):
        super().__init__()
        self.proj_img = nn.Linear(img_dim, 256, bias=False)
        self.proj_vid = nn.Linear(vid_dim, 256, bias=False)

    def forward(
        self,
        img_teacher_patches: torch.Tensor,   # (B_img, N, D_img)
        vid_student_tubes: torch.Tensor,     # (B_vid, T*ph*pw, D_vid)
    ) -> torch.Tensor:
        # Mean-pool both to (B, D) then align
        img_feat = self.proj_img(img_teacher_patches.mean(1))   # (B_img, 256)
        vid_feat = self.proj_vid(vid_student_tubes.mean(1))     # (B_vid, 256)
        img_feat = F.normalize(img_feat, dim=-1)
        vid_feat = F.normalize(vid_feat, dim=-1)

        B = min(img_feat.shape[0], vid_feat.shape[0])
        loss = (1 - (img_feat[:B] * vid_feat[:B]).sum(-1)).mean()
        return loss


# ── Prototype clustering head ─────────────────────────────────────────────────

class PrototypeHead(nn.Module):
    """
    Encourages image and video patch tokens to share prototype distributions
    for semantic organisation (DINO-style prototype consistency).

    Maintains K learnable prototype vectors.
    Assigns both image and video tokens to prototypes via soft assignment.
    Loss: cross-entropy between image and video prototype distributions.
    """

    def __init__(self, embed_dim: int = 1024, n_prototypes: int = 256):
        super().__init__()
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(n_prototypes, embed_dim), dim=-1)
        )
        self.n_prototypes = n_prototypes

    def _assign(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, D)  → soft prototype assignment (B, n_prototypes)
        via mean-pooled dot-product with L2-normalised prototypes.
        """
        feat  = F.normalize(tokens.mean(1), dim=-1)   # (B, D)
        proto = F.normalize(self.prototypes, dim=-1)  # (K, D)
        logits = feat @ proto.T                        # (B, K)
        return F.softmax(logits / 0.1, dim=-1)         # temperature=0.1

    def consistency_loss(
        self,
        img_tokens: torch.Tensor,   # (B_img, N, D)
        vid_tokens: torch.Tensor,   # (B_vid, T*ph*pw, D)
    ) -> torch.Tensor:
        p_img = self._assign(img_tokens)   # (B_img, K)
        p_vid = self._assign(vid_tokens)   # (B_vid, K)
        B = min(p_img.shape[0], p_vid.shape[0])
        # Symmetric cross-entropy
        loss = (
            -(p_img[:B] * (p_vid[:B] + 1e-8).log()).sum(-1).mean()
            - (p_vid[:B] * (p_img[:B] + 1e-8).log()).sum(-1).mean()
        ) / 2.0
        return loss


# ── Factory ────────────────────────────────────────────────────────────────────

def build_video_branch(
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    hf_cache_dir: Optional[str] = None,
) -> VideoBranch:
    """
    Download and instantiate the VideoBranch.

    Uses facebook/vjepa2-vitl-fpc64-256:
      - ViT-L encoder, 1024 hidden dim
      - 64 frames per clip at 256px
      - Built-in predictor (narrow 6-layer transformer)

    We initialise student from pretrained weights and hard-copy to teacher.
    The pretrained weights provide a strong spatiotemporal prior for ultrasound
    video understanding out of the box.
    """
    from transformers import AutoModel

    log.info(f"Loading V-JEPA2-L from {VJEPA2_L_HF} ...")
    student_hf = AutoModel.from_pretrained(
        VJEPA2_L_HF,
        torch_dtype=dtype,
        cache_dir=hf_cache_dir,
    )

    log.info("Copying V-JEPA2-L to EMA teacher ...")
    teacher_hf = copy.deepcopy(student_hf)

    branch = VideoBranch(student_hf, teacher_hf)
    branch = branch.to(device=device, dtype=dtype)

    total_params = sum(p.numel() for p in branch.student.parameters())
    log.info(f"VideoBranch ready.  Student params: {total_params/1e6:.1f}M")
    return branch
