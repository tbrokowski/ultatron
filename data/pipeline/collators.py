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

Extended collators (SAM, CLIP, aligned dual-stream)
----------------------------------------------------
SAMPromptCollator / SAMPromptBatch
   Bundles is_promptable samples with boxes, point prompts, text prompts,
   and ground-truth masks for MedSAM-3's iterative agent loop.

CrossModalCollator / CrossModalBatch
   Packages image tensors + tokenised text for CLIP-style image-text
   contrastive training (cross-modal stream).

AlignedDualStreamBatch (replaces DualStreamBatch in Phase 3)
   Extends the dual-stream batch with alignment_pairs linking image
   crops to video tubes that share the same study_id and frame range.
   This is required for the cross-branch patch-to-tube distillation:
     L_cross = Σ_{(i,j)∈A} w_ij · D(z_i^img_teacher, z_j^vid_student)

PairedSSLCollator
   Collates output from PairedSSLDataset into an AlignedDualStreamBatch.

make_aligned_dual_stream / build_alignment_pairs
   Helper to wrap a raw (image_batch, video_batch) pair into an
   AlignedDualStreamBatch by computing cross-branch alignment pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
            # Determine max spatial size across all masks
            max_H = 0
            max_W = 0
            for m in raw_segs:
                if m is None:
                    continue
                if m.ndim == 4:
                    _, _, h, w = m.shape
                elif m.ndim == 3:
                    _, h, w = m.shape
                elif m.ndim == 2:
                    h, w = m.shape
                else:
                    raise ValueError(f"Unexpected seg_mask ndim={m.ndim}")
                max_H = max(max_H, h)
                max_W = max(max_W, w)

            if max_H == 0 or max_W == 0:
                seg_masks = None
            else:
                padded_segs: List[torch.Tensor] = []
                for m in raw_segs:
                    if m is None:
                        padded = torch.zeros(1, 1, max_H, max_W)
                    else:
                        if m.ndim == 4:
                            t = m
                        elif m.ndim == 3:
                            # (1, H, W) -> (1, 1, H, W)
                            t = m.unsqueeze(0)
                        elif m.ndim == 2:
                            # (H, W) -> (1, 1, H, W)
                            t = m.unsqueeze(0).unsqueeze(0)
                        else:
                            raise ValueError(f"Unexpected seg_mask ndim={m.ndim}")

                        _, _, h, w = t.shape
                        padded = torch.zeros(1, 1, max_H, max_W, dtype=t.dtype)
                        padded[:, :, :h, :w] = t
                    padded_segs.append(padded)
                seg_masks = torch.cat(padded_segs, dim=0)
        else:
            seg_masks = None

        return {
            "global_crops":      global_crops,     # B × n_g × 1 × H_max × W_max
            "global_pmasks":     global_pmasks,     # B × n_g × ph_max × pw_max
            "local_crops":       local_crops,       # B × n_l × 1 × h_max × w_max
            "local_pmasks":      local_pmasks,      # B × n_l × ph_max_l × pw_max_l
            "patch_masks":       patch_masks,       # B × ph_max × pw_max  (freq mask)
            "dataset_ids":       [s["dataset_id"]       for s in samples],
            "anatomy_families":  [s["anatomy_family"]   for s in samples],
            "tiers":             torch.tensor([s["tier"]          for s in samples], dtype=torch.long),
            "sample_ids":        [s["sample_id"]        for s in samples],
            "study_ids":         [s.get("study_id", "") for s in samples],
            "source_frame_idxs": [s.get("source_frame_idx", -1) for s in samples],
            "seg_masks":         seg_masks,
            "cls_labels":        torch.tensor([s.get("cls_label", -1) for s in samples], dtype=torch.long),
            "task_types":        [s["task_type"]        for s in samples],
            "is_promptable":     torch.tensor([s.get("is_promptable", False) for s in samples]),
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

        # Build source_frame_indices tensor: (B, max_T) — zero-padded
        src_idx_lists = [s.get("source_frame_indices", []) for s in samples]
        source_frame_indices = torch.zeros(B, max_T, dtype=torch.long)
        for i, idxs in enumerate(src_idx_lists):
            for t, v in enumerate(idxs[:max_T]):
                source_frame_indices[i, t] = int(v)

        return {
            "full_clips":            full_clips,
            "visible_clips":         visible_clips,
            "tube_masks":            tube_masks,
            "padding_masks":         padding_masks,
            "valid_frames":          valid_frames,
            "dataset_ids":           [s["dataset_id"]       for s in samples],
            "anatomy_families":      [s["anatomy_family"]   for s in samples],
            "tiers":                 torch.tensor([s["tier"]     for s in samples], dtype=torch.long),
            "sample_ids":            [s["sample_id"]        for s in samples],
            "study_ids":             [s.get("study_id", "") for s in samples],
            "source_frame_indices":  source_frame_indices,   # (B, max_T) original frame positions
            "fps":                   torch.tensor([s.get("fps", 25.0)   for s in samples]),
            "is_cine":               torch.tensor([s.get("is_cine", False) for s in samples]),
            "task_types":            [s["task_type"]        for s in samples],
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. SAM Prompt Collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SAMPromptBatch:
    """
    One batch for the MedSAM-3 agent loop.

    images        : (B, 1, H, W)  — full-resolution greyscale frames
                    (SAM encoder processes at its native 1024×1024 internally)
    boxes         : list of (N_i, 4) tensors in xyxy format, variable N_i per sample
    point_coords  : list of (N_i, K, 2) tensors  [K points per instance]
    point_labels  : list of (N_i, K)   tensors  [1=fg, 0=bg]
    text_prompts  : list of str — one text prompt per sample (may be empty string)
    gt_masks      : list of (N_i, 1, H, W) tensors — ground truth per instance
    n_instances   : LongTensor (B,) — N_i per sample
    anatomy_families : list[str]
    sample_ids    : list[str]
    dataset_ids   : list[str]
    is_iterative  : BoolTensor (B,) — run full agent refinement loop
    """
    images:           Tensor
    boxes:            List[Tensor]
    point_coords:     List[Tensor]
    point_labels:     List[Tensor]
    text_prompts:     List[str]
    gt_masks:         List[Tensor]
    n_instances:      Tensor
    anatomy_families: List[str]
    sample_ids:       List[str]
    dataset_ids:      List[str]
    is_iterative:     Tensor

    def to(self, device) -> "SAMPromptBatch":
        def _m(x):
            if isinstance(x, Tensor): return x.to(device)
            if isinstance(x, list):
                return [_m(v) for v in x]
            return x
        return SAMPromptBatch(
            images           = self.images.to(device),
            boxes            = [b.to(device) for b in self.boxes],
            point_coords     = [p.to(device) for p in self.point_coords],
            point_labels     = [p.to(device) for p in self.point_labels],
            text_prompts     = self.text_prompts,
            gt_masks         = [m.to(device) for m in self.gt_masks],
            n_instances      = self.n_instances.to(device),
            anatomy_families = self.anatomy_families,
            sample_ids       = self.sample_ids,
            dataset_ids      = self.dataset_ids,
            is_iterative     = self.is_iterative.to(device),
        )


class SAMPromptCollator:
    """
    Collates items from a dataset that returns per-sample dicts in the form:

        {
          "image":          Tensor (1, H, W)
          "boxes":          Tensor (N, 4)          — xyxy, from Instance.box
          "point_coords":   Tensor (N, K, 2)       — optional, may be zeros
          "point_labels":   Tensor (N, K)
          "text_prompt":    str                     — LabelSpec.text_label
          "gt_masks":       Tensor (N, 1, H, W)    — from Instance.mask_path
          "anatomy_family": str
          "sample_id":      str
          "dataset_id":     str
          "is_iterative":   bool
        }

    The dataset should filter to is_promptable=True samples before using
    this collator. Samples with no instances are skipped (filtered upstream).

    Usage
    -----
        dataset = DownstreamDataset(promptable_entries, task_config, ...)
        loader = DataLoader(dataset, collate_fn=SAMPromptCollator(), ...)

    Produces SAMPromptBatch objects ready for the MedSAM-3 interface.
    """

    def __call__(self, samples: List[Dict]) -> SAMPromptBatch:
        images = torch.stack([s["image"] for s in samples])  # B×1×H×W

        boxes        = [s.get("boxes",       torch.zeros(0, 4))     for s in samples]
        point_coords = [s.get("point_coords", torch.zeros(0, 1, 2)) for s in samples]
        point_labels = [s.get("point_labels", torch.zeros(0, 1))    for s in samples]
        gt_masks     = [s.get("gt_masks",     torch.zeros(0, 1, *images.shape[-2:])) for s in samples]

        n_instances = torch.tensor([len(b) for b in boxes], dtype=torch.long)

        text_prompts     = [s.get("text_prompt", "")          for s in samples]
        anatomy_families = [s.get("anatomy_family", "unknown") for s in samples]
        sample_ids       = [s.get("sample_id", "")             for s in samples]
        dataset_ids      = [s.get("dataset_id", "")            for s in samples]
        is_iterative     = torch.tensor([s.get("is_iterative", False) for s in samples])

        return SAMPromptBatch(
            images           = images,
            boxes            = boxes,
            point_coords     = point_coords,
            point_labels     = point_labels,
            text_prompts     = text_prompts,
            gt_masks         = gt_masks,
            n_instances      = n_instances,
            anatomy_families = anatomy_families,
            sample_ids       = sample_ids,
            dataset_ids      = dataset_ids,
            is_iterative     = is_iterative,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CrossModal (CLIP) Collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrossModalBatch:
    """
    One batch for CLIP-style image-text contrastive training.

    images          : (B, 1, H, W)
    input_ids       : (B, L)        — tokenised text (BioMedCLIP / PubMedBERT)
    attention_mask  : (B, L)
    text_raw        : list[str]     — raw text before tokenisation (for logging)
    anatomy_families: list[str]
    sample_ids      : list[str]
    dataset_ids     : list[str]
    """
    images:           Tensor
    input_ids:        Tensor
    attention_mask:   Tensor
    text_raw:         List[str]
    anatomy_families: List[str]
    sample_ids:       List[str]
    dataset_ids:      List[str]

    def to(self, device) -> "CrossModalBatch":
        return CrossModalBatch(
            images           = self.images.to(device),
            input_ids        = self.input_ids.to(device),
            attention_mask   = self.attention_mask.to(device),
            text_raw         = self.text_raw,
            anatomy_families = self.anatomy_families,
            sample_ids       = self.sample_ids,
            dataset_ids      = self.dataset_ids,
        )


class CrossModalCollator:
    """
    Collates items from CrossModalDataset:

        { "image": Tensor (1,H,W), "text": str, "sample_id": str,
          "dataset_id": str, "anatomy_family": str }

    Tokenises text on-the-fly using a Hugging Face tokeniser.

    Parameters
    ----------
    tokenizer_name : str
        HuggingFace model name or local path.
        Default: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        (BioMedCLIP-compatible).
    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        tokenizer_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        max_length: int = 256,
    ):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = None   # lazy init (fork-safe for DataLoader workers)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except ImportError:
                raise RuntimeError(
                    "transformers library required for CrossModalCollator: "
                    "pip install transformers"
                )
        return self._tokenizer

    def __call__(self, samples: List[Dict]) -> CrossModalBatch:
        images   = torch.stack([s["image"] for s in samples])   # B×1×H×W
        texts    = [s.get("text", "") for s in samples]

        tokenizer = self._get_tokenizer()
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return CrossModalBatch(
            images           = images,
            input_ids        = encoded["input_ids"],
            attention_mask   = encoded["attention_mask"],
            text_raw         = texts,
            anatomy_families = [s.get("anatomy_family", "unknown") for s in samples],
            sample_ids       = [s.get("sample_id", "")             for s in samples],
            dataset_ids      = [s.get("dataset_id", "")            for s in samples],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Aligned Dual-Stream Batch  (Gap 1 — cross-branch frame alignment)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlignmentPair:
    """
    One alignment triplet for L_cross computation.

    img_batch_idx : int   — index into AlignedDualStreamBatch.image_batch samples
    vid_batch_idx : int   — index into AlignedDualStreamBatch.video_batch samples
    frame_offset  : int   — temporal position t in the video clip (0-indexed)
    weight        : float — w_ij in L_cross; 1.0 if frames overlap, 0.5 for adjacent
    """
    img_batch_idx: int
    vid_batch_idx: int
    frame_offset:  int
    weight:        float = 1.0


@dataclass
class AlignedDualStreamBatch:
    """
    Extends DualStreamBatch with cross-branch frame alignment information.

    alignment_pairs : List[AlignmentPair]
        Identifies which image batch entry (i) corresponds to which video batch
        entry (j) at temporal position t.  Used by the cross-distillation module:

            for pair in batch.alignment_pairs:
                img_patch_tokens = img_teacher_out[pair.img_batch_idx]  # (N, D)
                vid_tube_tokens  = vid_student_out[pair.vid_batch_idx,
                                                   pair.frame_offset]   # (ph*pw, D)
                loss += pair.weight * D(img_patch_tokens, vid_tube_tokens)

    If no cross-branch pairs exist (disjoint studies in both batches), this list
    is empty and L_cross is skipped for this step. This is expected at the start
    of training when the curriculum draws diverse batches.
    """
    image_batch:      Dict[str, Any]
    video_batch:      Dict[str, Any]
    alignment_pairs:  List[AlignmentPair] = field(default_factory=list)

    @property
    def has_cross_branch_pairs(self) -> bool:
        return len(self.alignment_pairs) > 0

    @property
    def device(self):
        return self.image_batch["global_crops"].device

    def to(self, device) -> "AlignedDualStreamBatch":
        def _move(d):
            return {k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in d.items()}
        return AlignedDualStreamBatch(
            image_batch     = _move(self.image_batch),
            video_batch     = _move(self.video_batch),
            alignment_pairs = self.alignment_pairs,
        )


def build_alignment_pairs(
    image_batch: Dict[str, Any],
    video_batch: Dict[str, Any],
) -> List[AlignmentPair]:
    """
    Compute cross-branch alignment pairs from study_id and frame_index fields.

    The image batch must carry:
        "study_ids"         : list[str]    — one per sample
        "source_frame_idxs" : list[int]    — which frame in the original video

    The video batch must carry:
        "study_ids"              : list[str]     — one per clip
        "source_frame_indices"   : Tensor (B, T) — original frame numbers

    Two samples are paired when:
        image_study_id == video_study_id
        AND image_source_frame_idx falls within [vid_frame_start, vid_frame_end]
    """
    pairs: List[AlignmentPair] = []

    img_study_ids  = image_batch.get("study_ids", [])
    img_frame_idxs = image_batch.get("source_frame_idxs", [])
    vid_study_ids  = video_batch.get("study_ids", [])
    vid_frame_indices = video_batch.get("source_frame_indices")  # (B, T) or None

    if not img_study_ids or not vid_study_ids:
        return pairs

    # Build vid lookup: study_id → list[(vid_batch_idx, frame_positions)]
    vid_lookup: Dict[str, List[Tuple[int, List[int]]]] = {}
    for j, vsid in enumerate(vid_study_ids):
        if vsid not in vid_lookup:
            vid_lookup[vsid] = []
        if vid_frame_indices is not None:
            frames = vid_frame_indices[j].tolist()
        else:
            frames = []
        vid_lookup[vsid].append((j, frames))

    for i, (isid, iframe) in enumerate(zip(img_study_ids, img_frame_idxs)):
        if isid not in vid_lookup:
            continue
        for j, vframes in vid_lookup[isid]:
            if not vframes:
                # No frame index info — add a weak pair at offset 0
                pairs.append(AlignmentPair(i, j, frame_offset=0, weight=0.5))
                continue
            # Find which temporal slot in the video clip is closest to iframe
            vframes_arr = vframes
            if iframe < 0:
                continue  # image from static dataset, no frame index
            # Check overlap
            if iframe in vframes_arr:
                t = vframes_arr.index(iframe)
                pairs.append(AlignmentPair(i, j, frame_offset=t, weight=1.0))
            else:
                # Find nearest frame
                diffs = [abs(f - iframe) for f in vframes_arr]
                t = int(min(range(len(diffs)), key=lambda k: diffs[k]))
                dist = diffs[t]
                # Weight decays with distance (within 5 frames = valid pair)
                if dist <= 5:
                    w = max(0.1, 1.0 - dist * 0.15)
                    pairs.append(AlignmentPair(i, j, frame_offset=t, weight=w))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 4. Aligned combined loader helper
# ─────────────────────────────────────────────────────────────────────────────

class PairedSSLCollator:
    """
    Collates output from PairedSSLDataset into an AlignedDualStreamBatch.

    Each sample in `samples` is ``{"image": img_dict, "video": vid_dict,
    "frame_offset": int}``.  Because both views come from the same source
    clip, alignment_pairs is pre-built trivially:
        pair_i = AlignmentPair(img_batch_idx=i, vid_batch_idx=i,
                               frame_offset=samples[i]["frame_offset"],
                               weight=1.0)

    The image and video sub-dicts are collated with the standard
    ImageSSLCollator and VideoSSLCollator respectively.
    """

    def __init__(self):
        self._img_col = ImageSSLCollator()
        self._vid_col = VideoSSLCollator()

    def __call__(self, samples: List[Dict]) -> "AlignedDualStreamBatch":
        image_batch = self._img_col([s["image"] for s in samples])
        video_batch = self._vid_col([s["video"] for s in samples])
        alignment_pairs = [
            AlignmentPair(
                img_batch_idx=i,
                vid_batch_idx=i,
                frame_offset=s["frame_offset"],
                weight=1.0,
            )
            for i, s in enumerate(samples)
        ]
        return AlignedDualStreamBatch(
            image_batch=image_batch,
            video_batch=video_batch,
            alignment_pairs=alignment_pairs,
        )


def make_aligned_dual_stream(
    image_batch: Dict[str, Any],
    video_batch: Dict[str, Any],
) -> AlignedDualStreamBatch:
    """
    Wraps a raw (image_batch, video_batch) pair into an AlignedDualStreamBatch
    by computing cross-branch alignment pairs.

    Drop-in replacement for the DualStreamBatch constructor in
    USFoundationDataModule.combined_loader():

        for img_b, vid_b in zip(img_loader, vid_loader):
            yield make_aligned_dual_stream(img_b, vid_b)
    """
    pairs = build_alignment_pairs(image_batch, video_batch)
    return AlignedDualStreamBatch(
        image_batch     = image_batch,
        video_batch     = video_batch,
        alignment_pairs = pairs,
    )
